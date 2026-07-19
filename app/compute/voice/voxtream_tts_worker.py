from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TextIO, TypedDict, cast

import numpy as np
from voxtream.config import SpeechGeneratorConfig
from voxtream.generator import SpeechGenerator
from voxtream.utils.generator import set_seed, text_generator
from voxtream.utils.generator.text import build_text_progress_metadata

from app.compute.voice.tts_worker_protocol import (
    CancelTtsCommand,
    FinishTtsCommand,
    ShutdownTtsCommand,
    StartTtsCommand,
    TtsAudioEvent,
    TtsEndEvent,
    TtsWordCommand,
    TtsWordProcessedEvent,
    TtsWorkerCommand,
    TtsWorkerErrorEvent,
    TtsWorkerEvent,
    TtsWorkerReadyEvent,
    VoxtreamTtsFirstAudioMetricsEvent,
    tts_worker_command_adapter,
)
from app.compute.voice.voxtream_alignment import (
    VoxtreamPendingWordBoundary,
    release_started_word_boundaries,
)

logger = logging.getLogger(__name__)
VOXTREAM_WARMUP_TEXT = "Warmup."
PCM_MAXIMUM = 32_767.0


class VoxtreamProgressMetadata(TypedDict):
    time_sec: float
    phone_position: int


@dataclass(frozen=True)
class _QueuedWord:
    sequence_number: int
    text: str
    text_end: int


class _InputMarker(Enum):
    FINISH = auto()
    CANCEL = auto()


_GenerationInput = _QueuedWord | _InputMarker


class VoxtreamWorkerRuntime:
    def __init__(
        self,
        config_path: Path,
        prompt_audio_path: Path,
        compile_model: bool,
        cache_prompt_in_memory: bool,
    ) -> None:
        if not config_path.is_file():
            raise ValueError(f"VoXtream configuration does not exist: {config_path}")
        if not prompt_audio_path.is_file():
            raise ValueError(f"VoXtream prompt audio does not exist: {prompt_audio_path}")
        with config_path.open(encoding="utf-8") as config_stream:
            configuration = SpeechGeneratorConfig(**json.load(config_stream))
        configuration.cache_prompt = True
        self.configuration = configuration
        self.prompt_audio_path = prompt_audio_path
        set_seed()
        self.generator = SpeechGenerator(
            configuration,
            compile=compile_model,
            cache_prompt_in_memory=cache_prompt_in_memory,
        )
        self._warmup()

    @property
    def sample_rate(self) -> int:
        return self.configuration.mimi_sr

    def _warmup(self) -> None:
        stream = self.generator.generate_stream(
            prompt_audio_path=self.prompt_audio_path,
            text=text_generator(VOXTREAM_WARMUP_TEXT),
        )
        for _ in stream:
            pass


class VoxtreamGeneration:
    def __init__(
        self,
        runtime: VoxtreamWorkerRuntime,
        emit: Callable[[TtsWorkerEvent], None],
    ) -> None:
        self.runtime = runtime
        self.emit = emit
        self.input_queue: queue.Queue[_GenerationInput] = queue.Queue()
        self.cancellation_event = threading.Event()
        self.pending_boundaries: deque[VoxtreamPendingWordBoundary] = deque()
        self.accumulated_words: list[str] = []
        self.first_word_received_at: float | None = None
        self.first_text_requested_at: float | None = None
        self.first_word_yielded_at: float | None = None
        self.input_finished = False

    def add_word(self, command: TtsWordCommand) -> None:
        if self.input_finished:
            raise ValueError("Cannot add VoXtream text after input has finished.")
        if self.first_word_received_at is None:
            self.first_word_received_at = time.perf_counter()
        self.input_queue.put(
            _QueuedWord(
                sequence_number=command.sequence_number,
                text=command.text,
                text_end=command.text_end,
            )
        )

    def finish(self) -> None:
        if self.input_finished:
            raise ValueError("VoXtream input may only be finished once.")
        self.input_finished = True
        self.input_queue.put(_InputMarker.FINISH)

    def cancel(self) -> None:
        self.input_finished = True
        self.cancellation_event.set()
        self.input_queue.put(_InputMarker.CANCEL)

    def run(self) -> None:
        generation_started_at = time.perf_counter()
        emitted_sample_count = 0
        first_audio_emitted = False
        stream = self.runtime.generator.generate_stream(
            prompt_audio_path=self.runtime.prompt_audio_path,
            text=self._stream_text(),
            return_progress=True,
        )
        for output in stream:
            if self.cancellation_event.is_set():
                break
            audio_frame, generation_seconds, raw_progress = output
            progress = cast(VoxtreamProgressMetadata, raw_progress)
            pcm_bytes = _float_audio_to_pcm16(audio_frame)
            if not pcm_bytes:
                continue
            if not first_audio_emitted:
                first_audio_emitted = True
                self.emit(
                    self._first_audio_metrics(
                        generation_started_at=generation_started_at,
                        first_frame_generation_seconds=generation_seconds,
                    )
                )
            self._emit_boundaries(
                phone_position=progress["phone_position"],
                start_sample=emitted_sample_count,
            )
            self.emit(
                TtsAudioEvent.from_pcm_bytes(
                    pcm_bytes=pcm_bytes,
                    start_sample=emitted_sample_count,
                )
            )
            emitted_sample_count += len(pcm_bytes) // 2
        self.emit(TtsEndEvent(cancelled=self.cancellation_event.is_set()))

    def _stream_text(self) -> Generator[str, None, None]:
        self.first_text_requested_at = time.perf_counter()
        while not self.cancellation_event.is_set():
            item = self.input_queue.get()
            match item:
                case _QueuedWord():
                    self.accumulated_words.append(item.text)
                    text_metadata = build_text_progress_metadata(
                        " ".join(self.accumulated_words),
                        config=self.runtime.configuration,
                        phone_to_token=self.runtime.generator.ctx.phone_to_token,
                        phonemizer=self.runtime.generator.ctx.phonemizer,
                        max_phone_tokens=self.runtime.configuration.max_phone_tokens,
                    )
                    if not text_metadata.words:
                        raise RuntimeError("VoXtream produced no phonemes for a synthesis word.")
                    self.pending_boundaries.append(
                        VoxtreamPendingWordBoundary(
                            phone_start=text_metadata.words[-1].start,
                            text_offset=item.text_end,
                        )
                    )
                    self.emit(TtsWordProcessedEvent(sequence_number=item.sequence_number))
                    if self.first_word_yielded_at is None:
                        self.first_word_yielded_at = time.perf_counter()
                    yield item.text
                case _InputMarker.FINISH | _InputMarker.CANCEL:
                    return

    def _emit_boundaries(self, phone_position: int, start_sample: int) -> None:
        for boundary in release_started_word_boundaries(
            self.pending_boundaries,
            phone_position,
            start_sample,
        ):
            self.emit(boundary)

    def _first_audio_metrics(
        self,
        generation_started_at: float,
        first_frame_generation_seconds: float,
    ) -> VoxtreamTtsFirstAudioMetricsEvent:
        if self.first_word_received_at is None:
            raise AssertionError("VoXtream emitted audio before receiving text.")
        if self.first_text_requested_at is None:
            raise AssertionError("VoXtream emitted audio before requesting text.")
        return VoxtreamTtsFirstAudioMetricsEvent(
            first_word_to_audio_seconds=time.perf_counter() - self.first_word_received_at,
            prompt_preparation_seconds=(self.first_text_requested_at - generation_started_at),
            first_frame_generation_seconds=first_frame_generation_seconds,
        )


@dataclass(frozen=True)
class ActiveGeneration:
    generation: VoxtreamGeneration
    thread: threading.Thread


class VoxtreamWorkerController:
    def __init__(self, runtime: VoxtreamWorkerRuntime, output_stream: TextIO) -> None:
        self.runtime = runtime
        self.output_stream = output_stream
        self.output_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_generation: ActiveGeneration | None = None

    def run(self) -> None:
        self._send_event(TtsWorkerReadyEvent(sample_rate=self.runtime.sample_rate))
        for command_json in sys.stdin:
            try:
                command = tts_worker_command_adapter.validate_json(command_json)
                if self._handle_command(command):
                    return
            except Exception:
                logger.exception("VoXtream worker command handling failed")

    def _handle_command(self, command: TtsWorkerCommand) -> bool:
        match command:
            case StartTtsCommand():
                self._start()
                return False
            case TtsWordCommand():
                self._require_generation().add_word(command)
                return False
            case FinishTtsCommand():
                self._require_generation().finish()
                return False
            case CancelTtsCommand():
                self._require_generation().cancel()
                return False
            case ShutdownTtsCommand():
                self._shutdown()
                return True

    def _start(self) -> None:
        with self.state_lock:
            if self.active_generation is not None:
                self._send_event(
                    TtsWorkerErrorEvent(message="VoXtream generation is already active.")
                )
                return
            generation = VoxtreamGeneration(self.runtime, self._send_event)
            generation_thread = threading.Thread(
                target=self._run_generation,
                args=(generation,),
                daemon=True,
                name="voxtream-generation",
            )
            self.active_generation = ActiveGeneration(
                generation=generation,
                thread=generation_thread,
            )
            generation_thread.start()

    def _run_generation(self, generation: VoxtreamGeneration) -> None:
        try:
            generation.run()
        except Exception as error:
            logger.exception("VoXtream generation failed")
            self._send_event(TtsWorkerErrorEvent(message=str(error)))
        finally:
            with self.state_lock:
                active_generation = self.active_generation
                if active_generation is not None and active_generation.generation is generation:
                    self.active_generation = None

    def _shutdown(self) -> None:
        with self.state_lock:
            active_generation = self.active_generation
            if active_generation is not None:
                active_generation.generation.cancel()
        if active_generation is not None:
            active_generation.thread.join()

    def _require_generation(self) -> VoxtreamGeneration:
        with self.state_lock:
            active_generation = self.active_generation
            if active_generation is None:
                raise RuntimeError("No VoXtream generation is active.")
            return active_generation.generation

    def _send_event(self, event: TtsWorkerEvent) -> None:
        with self.output_lock:
            self.output_stream.write(event.model_dump_json() + "\n")
            self.output_stream.flush()


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-prompt-in-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    options = parser.parse_args(arguments)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    protocol_output_stream = sys.stdout
    sys.stdout = sys.stderr
    runtime = VoxtreamWorkerRuntime(
        config_path=options.config,
        prompt_audio_path=options.prompt_audio,
        compile_model=options.compile,
        cache_prompt_in_memory=options.cache_prompt_in_memory,
    )
    VoxtreamWorkerController(runtime, protocol_output_stream).run()


def _float_audio_to_pcm16(audio_frame: np.ndarray) -> bytes:
    samples = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
    clipped_samples = np.clip(samples, -1.0, 1.0)
    return (clipped_samples * PCM_MAXIMUM).astype("<i2").tobytes()


if __name__ == "__main__":
    main()
