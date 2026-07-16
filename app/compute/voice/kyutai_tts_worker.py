from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable, Sequence
from enum import Enum, auto
from pathlib import Path
from typing import Final, TextIO

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from moshi.conditioners import ConditionAttributes
from moshi.models.lm import LMGen
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import Entry, TTSModel, script_to_entries

from app.compute.voice.interfaces import SynthesizedAudioChunk, SynthesizedWordBoundary
from app.compute.voice.model_constants import (
    KYUTAI_TTS_MODEL_NAME,
    KYUTAI_TTS_MODEL_REVISION,
)
from app.compute.voice.tts_alignment import TranscriptBoundaryTracker
from app.compute.voice.tts_worker_protocol import (
    TtsAudioEvent,
    TtsEndEvent,
    TtsFirstAudioMetricsEvent,
    TtsWordBoundaryEvent,
    TtsWordCommand,
    TtsWordProcessedEvent,
    TtsWorkerCommandType,
    TtsWorkerErrorEvent,
    TtsWorkerEvent,
    TtsWorkerReadyEvent,
    tts_worker_command_adapter,
)

KYUTAI_TTS_VOICE_REPOSITORY: Final = "kyutai/tts-voices"
KYUTAI_TTS_VOICE_REVISION: Final = "323332d33f997de8394f24a193e1a76df720e01a"
KYUTAI_TTS_VOICE: Final = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
KYUTAI_TTS_CODEBOOK_COUNT: Final = 32
KYUTAI_TTS_TEMPERATURE: Final = 0.6
KYUTAI_TTS_CFG_COEFFICIENT: Final = 2.0
KYUTAI_TTS_WARMUP_TEXT: Final = "Warmup."
KYUTAI_TTS_PROCESS_MAX_STEPS: Final = 512
KYUTAI_TTS_FINALIZE_MAX_STEPS: Final = 2_048
KYUTAI_TTS_MAX_STALLED_STEPS: Final = 192
KYUTAI_TTS_FINAL_FLUSH_MAX_STEPS: Final = 64


class _WorkerMarker(Enum):
    FINISH = auto()


_WorkerInput = TtsWordCommand | _WorkerMarker


class KyutaiTtsWorkerRuntime:
    def __init__(self, output_stream: TextIO) -> None:
        checkpoint = CheckpointInfo.from_hf_repo(
            KYUTAI_TTS_MODEL_NAME,
            revision=KYUTAI_TTS_MODEL_REVISION,
        )
        self.model = TTSModel.from_checkpoint_info(
            checkpoint,
            n_q=KYUTAI_TTS_CODEBOOK_COUNT,
            temp=KYUTAI_TTS_TEMPERATURE,
            device="cuda",
        )
        voice_filename = KYUTAI_TTS_VOICE + self.model.voice_suffix
        voice_path = hf_hub_download(
            KYUTAI_TTS_VOICE_REPOSITORY,
            voice_filename,
            revision=KYUTAI_TTS_VOICE_REVISION,
        )
        attributes = self.model.make_condition_attributes(
            [Path(voice_path)],
            cfg_coef=KYUTAI_TTS_CFG_COEFFICIENT,
        )
        self.generator = _KyutaiStreamingGenerator(
            model=self.model,
            attributes=(attributes,),
        )
        self.model.mimi.streaming_forever(1)
        self.generator.lm_generation.streaming_forever(1)
        _warmup_streaming_generation(self.generator)
        self.output_stream = output_stream
        self.output_lock = threading.Lock()
        self.input_queue: queue.Queue[_WorkerInput] | None = None
        self.cancellation_event: threading.Event | None = None
        self.generation_thread: threading.Thread | None = None

    @property
    def sample_rate(self) -> int:
        return self.model.mimi.sample_rate

    def run(self, input_stream: TextIO) -> None:
        self._send_event(TtsWorkerReadyEvent(sample_rate=self.sample_rate))
        for command_json in input_stream:
            command = tts_worker_command_adapter.validate_json(command_json)
            match command.type:
                case TtsWorkerCommandType.START:
                    self._start_session()
                case TtsWorkerCommandType.WORD:
                    assert isinstance(command, TtsWordCommand)
                    if self.input_queue is None:
                        self._send_error("No TTS session is active.")
                        continue
                    self.input_queue.put(command)
                case TtsWorkerCommandType.FINISH:
                    if self.input_queue is None:
                        self._send_error("No TTS session is active.")
                        continue
                    self.input_queue.put(_WorkerMarker.FINISH)
                case TtsWorkerCommandType.CANCEL:
                    if self.cancellation_event is None or self.input_queue is None:
                        self._send_error("No TTS session is active.")
                        continue
                    self.cancellation_event.set()
                    self.input_queue.put(_WorkerMarker.FINISH)
                case TtsWorkerCommandType.SHUTDOWN:
                    if self.cancellation_event is not None:
                        self.cancellation_event.set()
                    return

    def _start_session(self) -> None:
        if self.generation_thread is not None:
            self.generation_thread.join(timeout=0.1)
            if self.generation_thread.is_alive():
                self._send_error("A TTS session is already active.")
                return
        self.input_queue = queue.Queue()
        self.cancellation_event = threading.Event()
        self.generation_thread = threading.Thread(target=self._run_session, daemon=True)
        self.generation_thread.start()

    def _run_session(self) -> None:
        assert self.input_queue is not None
        assert self.cancellation_event is not None
        cancellation_event = self.cancellation_event
        try:
            with torch.inference_mode():
                self.generator.start_turn(cancellation_event, self._send_synthesis_event)
                first_word = True
                while not cancellation_event.is_set():
                    item = self.input_queue.get()
                    match item:
                        case _WorkerMarker.FINISH:
                            self.generator.process_last()
                            break
                        case TtsWordCommand():
                            if first_word:
                                self.generator.record_first_word_received()
                            tokenization_started_at = time.perf_counter()
                            entries = script_to_entries(
                                self.model.tokenizer,
                                self.model.machine.token_ids,
                                self.model.mimi.frame_rate,
                                [item.text],
                                multi_speaker=first_word and self.model.multi_speaker,
                                padding_between=1,
                            )
                            self.generator.record_tokenization_duration(
                                time.perf_counter() - tokenization_started_at
                            )
                            first_word = False
                            self.generator.append_word(entries, item.text_end)
                            self.generator.process()
                            self._send_event(
                                TtsWordProcessedEvent(sequence_number=item.sequence_number)
                            )
        except Exception as error:
            self._send_error(str(error))
        finally:
            self._send_event(TtsEndEvent(cancelled=cancellation_event.is_set()))

    def _send_synthesis_event(
        self,
        event: SynthesizedAudioChunk | SynthesizedWordBoundary,
    ) -> None:
        match event:
            case SynthesizedAudioChunk():
                if event.start_sample == 0:
                    self._send_event(self.generator.require_first_audio_metrics())
                self._send_event(TtsAudioEvent.from_pcm_bytes(event.pcm_bytes, event.start_sample))
            case SynthesizedWordBoundary():
                self._send_event(
                    TtsWordBoundaryEvent(
                        text_offset=event.text_offset,
                        start_sample=event.start_sample,
                    )
                )

    def _send_error(self, message: str) -> None:
        self._send_event(TtsWorkerErrorEvent(message=message))

    def _send_event(self, event: TtsWorkerEvent) -> None:
        with self.output_lock:
            self.output_stream.write(event.model_dump_json() + "\n")
            self.output_stream.flush()


class _KyutaiStreamingGenerator:
    def __init__(
        self,
        model: TTSModel,
        attributes: Sequence[ConditionAttributes],
    ) -> None:
        if model.cfg_coef != 1.0:
            raise ValueError(
                "The distilled Kyutai TTS checkpoint must use cfg_coef=1.0 at inference."
            )
        if model.lm.condition_provider is None:
            raise ValueError("The Kyutai TTS checkpoint has no condition provider.")
        self.model = model
        self.cancellation_event = threading.Event()
        self.emit: Callable[[SynthesizedAudioChunk | SynthesizedWordBoundary], None] = (
            _discard_output
        )
        self._reset_turn_state()

        condition_tensors = model.lm.condition_provider.prepare_and_provide(attributes)

        def on_text_logits(text_logits: torch.Tensor) -> torch.Tensor:
            if model.padding_bonus:
                text_logits[..., model.machine.token_ids.pad] += model.padding_bonus
            return text_logits

        def on_audio(audio_tokens: torch.Tensor) -> None:
            for codebook_index in range(audio_tokens.shape[1]):
                delay = model.lm.delays[codebook_index + model.lm.audio_offset]
                if self.offset < delay + model.delay_steps:
                    audio_tokens[:, codebook_index] = model.machine.token_ids.zero

        def on_text(text_tokens: torch.Tensor) -> None:
            output_tokens: list[int] = []
            for token in text_tokens.tolist():
                output_token, _ = model.machine.process(self.offset, self.state, token)
                output_tokens.append(output_token)
            text_tokens[:] = torch.tensor(
                output_tokens,
                dtype=torch.long,
                device=text_tokens.device,
            )

        model.lm.dep_q = model.n_q
        self.lm_generation = LMGen(
            model.lm,
            temp=model.temp,
            temp_text=model.temp,
            cfg_coef=model.cfg_coef,
            condition_tensors=condition_tensors,
            on_text_logits_hook=on_text_logits,
            on_text_hook=on_text,
            on_audio_hook=on_audio,
            cfg_is_masked_until=None,
            cfg_is_no_text=True,
        )

    def start_turn(
        self,
        cancellation_event: threading.Event,
        emit: Callable[[SynthesizedAudioChunk | SynthesizedWordBoundary], None],
    ) -> None:
        self.lm_generation.reset_streaming()
        self.model.mimi.reset_streaming()
        self.cancellation_event = cancellation_event
        self.emit = emit
        self._reset_turn_state()

    def _reset_turn_state(self) -> None:
        self.offset = 0
        self.emitted_sample_count = 0
        self.transcript_index = 0
        self.first_word_received_at: float | None = None
        self.tokenization_seconds = 0.0
        self.language_model_step_seconds = 0.0
        self.mimi_decode_seconds = 0.0
        self.model_step_count = 0
        self.first_audio_metrics: TtsFirstAudioMetricsEvent | None = None
        self.boundary_tracker = TranscriptBoundaryTracker()
        self.state = self.model.machine.new_state([])

    def record_first_word_received(self) -> None:
        if self.first_word_received_at is not None:
            raise AssertionError("Kyutai first-word receipt may only be recorded once.")
        self.first_word_received_at = time.perf_counter()

    def record_tokenization_duration(self, duration_seconds: float) -> None:
        if duration_seconds < 0:
            raise ValueError("Kyutai tokenization duration must not be negative.")
        self.tokenization_seconds += duration_seconds

    def require_first_audio_metrics(self) -> TtsFirstAudioMetricsEvent:
        if self.first_audio_metrics is None:
            raise AssertionError("Kyutai emitted first audio without timing metrics.")
        return self.first_audio_metrics

    def append_word(self, entries: Sequence[Entry], text_offset: int) -> None:
        self.state.entries.extend(entries)
        immediate_boundaries = self.boundary_tracker.add_source_word(
            tuple(bool(entry.tokens) for entry in entries),
            text_offset,
        )
        for boundary in immediate_boundaries:
            self.emit(boundary)

    def process(self) -> None:
        self._step_until(
            is_complete=lambda: len(self.state.entries) <= self.model.machine.second_stream_ahead,
            maximum_steps=KYUTAI_TTS_PROCESS_MAX_STEPS,
            error_message="Kyutai generation did not consume the pending text entries.",
        )

    def process_last(self) -> None:
        self._step_until(
            is_complete=lambda: len(self.state.entries) == 0 and self.state.end_step is not None,
            maximum_steps=KYUTAI_TTS_FINALIZE_MAX_STEPS,
            error_message="Kyutai generation did not reach its end marker.",
        )
        additional_steps = self.model.delay_steps + max(self.model.lm.delays) + 8
        for _ in range(additional_steps):
            if self.cancellation_event.is_set():
                return
            self._step()
        self._flush_to_end_step()

    def _step_until(
        self,
        is_complete: Callable[[], bool],
        maximum_steps: int,
        error_message: str,
    ) -> None:
        stalled_steps = 0
        previous_progress = self._progress_signature()
        for _ in range(maximum_steps):
            if self.cancellation_event.is_set() or is_complete():
                return
            self._step()
            current_progress = self._progress_signature()
            if current_progress == previous_progress:
                stalled_steps += 1
                if stalled_steps >= KYUTAI_TTS_MAX_STALLED_STEPS:
                    raise RuntimeError(f"{error_message} Model state stopped advancing.")
            else:
                stalled_steps = 0
                previous_progress = current_progress
        if not is_complete():
            raise RuntimeError(f"{error_message} Step budget exhausted.")

    def _progress_signature(self) -> tuple[int, int | None, int, int]:
        return (
            len(self.state.entries),
            self.state.end_step,
            len(self.state.transcript),
            self.emitted_sample_count,
        )

    def _flush_to_end_step(self) -> None:
        if self.state.end_step is None:
            raise AssertionError("Kyutai generation ended without a model end step.")
        samples_per_frame = round(self.model.mimi.sample_rate / self.model.mimi.frame_rate)
        target_sample_count = self.state.end_step * samples_per_frame
        for _ in range(KYUTAI_TTS_FINAL_FLUSH_MAX_STEPS):
            if self.cancellation_event.is_set() or self.emitted_sample_count >= target_sample_count:
                return
            self._step()
        raise RuntimeError("Kyutai delayed audio did not reach the model end step.")

    def _step(self) -> None:
        if self.cancellation_event.is_set():
            return
        missing_codebooks = self.model.lm.n_q - self.model.lm.dep_q
        input_tokens = torch.full(
            (1, missing_codebooks, 1),
            self.model.machine.token_ids.zero,
            dtype=torch.long,
            device=self.model.lm.device,
        )
        language_model_step_started_at = time.perf_counter()
        frame = self.lm_generation.step(input_tokens)
        self.language_model_step_seconds += time.perf_counter() - language_model_step_started_at
        self.model_step_count += 1
        self.offset += 1
        self._emit_new_boundaries()
        if frame is None or not bool((frame != -1).all()):
            return
        mimi_decode_started_at = time.perf_counter()
        pcm = self.model.mimi.decode(frame[:, 1:, :]).cpu().float().numpy()
        self.mimi_decode_seconds += time.perf_counter() - mimi_decode_started_at
        samples = np.clip(pcm[0, 0], -1.0, 1.0)
        pcm_bytes = (samples * 32_767.0).astype("<i2").tobytes()
        if not pcm_bytes:
            return
        if self.emitted_sample_count == 0:
            if self.first_word_received_at is None:
                raise AssertionError("Kyutai emitted audio before receiving a word.")
            self.first_audio_metrics = TtsFirstAudioMetricsEvent(
                first_word_to_audio_seconds=time.perf_counter() - self.first_word_received_at,
                tokenization_seconds=self.tokenization_seconds,
                language_model_step_seconds=self.language_model_step_seconds,
                mimi_decode_seconds=self.mimi_decode_seconds,
                model_step_count=self.model_step_count,
                first_audio_model_step=self.offset,
            )
        self.emit(
            SynthesizedAudioChunk(
                pcm_bytes=pcm_bytes,
                start_sample=self.emitted_sample_count,
            )
        )
        self.emitted_sample_count += len(pcm_bytes) // 2

    def _emit_new_boundaries(self) -> None:
        new_transcript = self.state.transcript[self.transcript_index :]
        for _, model_step in new_transcript:
            start_sample = round(
                model_step * self.model.mimi.sample_rate / self.model.mimi.frame_rate
            )
            boundary = self.boundary_tracker.consume_transcript_word(start_sample)
            if boundary is not None:
                self.emit(boundary)
        self.transcript_index += len(new_transcript)


def _warmup_streaming_generation(generator: _KyutaiStreamingGenerator) -> None:
    cancellation_event = threading.Event()
    model = generator.model
    with torch.inference_mode():
        generator.start_turn(cancellation_event, _discard_output)
        generator.record_first_word_received()
        tokenization_started_at = time.perf_counter()
        entries = script_to_entries(
            model.tokenizer,
            model.machine.token_ids,
            model.mimi.frame_rate,
            [KYUTAI_TTS_WARMUP_TEXT],
            multi_speaker=model.multi_speaker,
            padding_between=1,
        )
        generator.record_tokenization_duration(time.perf_counter() - tokenization_started_at)
        generator.append_word(entries, len(KYUTAI_TTS_WARMUP_TEXT))
        generator.process_last()
        torch.cuda.synchronize()


def _discard_output(event: SynthesizedAudioChunk | SynthesizedWordBoundary) -> None:
    del event


def main() -> None:
    runtime = KyutaiTtsWorkerRuntime(sys.stdout)
    runtime.run(sys.stdin)


if __name__ == "__main__":
    main()
