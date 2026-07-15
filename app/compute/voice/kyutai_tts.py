from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from enum import Enum, auto
from pathlib import Path
from typing import Final

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from moshi.conditioners import ConditionAttributes
from moshi.models.lm import LMGen
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    Entry,
    TTSModel,
    script_to_entries,
)

from app.compute.voice.interfaces import (
    SpeechSynthesisSession,
    SynthesisEvent,
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
)
from app.compute.voice.thread_queue import get_until_cancelled
from app.compute.voice.tts_alignment import TranscriptBoundaryTracker

KYUTAI_TTS_MODEL_NAME: Final = DEFAULT_DSM_TTS_REPO
KYUTAI_TTS_MODEL_REVISION: Final = "f65439609986c392cb12df63938abcc550c3fb15"
KYUTAI_TTS_VOICE_REPOSITORY: Final = "kyutai/tts-voices"
KYUTAI_TTS_VOICE_REVISION: Final = "323332d33f997de8394f24a193e1a76df720e01a"
KYUTAI_TTS_VOICE: Final = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
KYUTAI_TTS_CODEBOOK_COUNT: Final = 32
KYUTAI_TTS_TEMPERATURE: Final = 0.6
KYUTAI_TTS_CFG_COEFFICIENT: Final = 2.0
KYUTAI_TTS_INPUT_QUEUE_SIZE: Final = 64
KYUTAI_TTS_OUTPUT_QUEUE_SIZE: Final = 32
KYUTAI_TTS_QUEUE_POLL_SECONDS: Final = 0.1


class _StreamMarker(Enum):
    FINISH = auto()
    END = auto()


_InputItem = SynthesisWord | _StreamMarker
_OutputItem = SynthesisEvent | Exception | _StreamMarker


class KyutaiSpeechSynthesizer:
    def __init__(self) -> None:
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
        self.attributes = self.model.make_condition_attributes(
            [Path(voice_path)],
            cfg_coef=KYUTAI_TTS_CFG_COEFFICIENT,
        )
        self.generation_lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self.model.mimi.sample_rate

    def start_session(self) -> SpeechSynthesisSession:
        return KyutaiSpeechSynthesisSession(
            model=self.model,
            attributes=self.attributes,
            generation_lock=self.generation_lock,
        )


class KyutaiSpeechSynthesisSession:
    def __init__(
        self,
        model: TTSModel,
        attributes: ConditionAttributes,
        generation_lock: threading.Lock,
    ) -> None:
        self.model = model
        self.attributes = attributes
        self.generation_lock = generation_lock
        self.input_queue: queue.Queue[_InputItem] = queue.Queue(maxsize=KYUTAI_TTS_INPUT_QUEUE_SIZE)
        self.output_queue: queue.Queue[_OutputItem] = queue.Queue(
            maxsize=KYUTAI_TTS_OUTPUT_QUEUE_SIZE
        )
        self.cancellation_event = threading.Event()
        self.input_finished = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    async def add_word(self, word: SynthesisWord) -> None:
        if self.input_finished:
            raise ValueError("Cannot add a word after synthesis input has finished.")
        if not word.text or any(character.isspace() for character in word.text):
            raise ValueError("Synthesis words must be non-empty and contain no whitespace.")
        await asyncio.to_thread(self._put_input, word)

    async def finish_input(self) -> None:
        if self.input_finished:
            raise ValueError("Synthesis input may only be finished once.")
        self.input_finished = True
        await asyncio.to_thread(self._put_input, _StreamMarker.FINISH)

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        while True:
            item = await asyncio.to_thread(self.output_queue.get)
            match item:
                case _StreamMarker.END:
                    return
                case Exception():
                    raise item
                case SynthesizedAudioChunk() | SynthesizedWordBoundary():
                    yield item
                case _:
                    raise AssertionError(f"Unexpected synthesis output item: {item!r}")

    async def cancel(self) -> None:
        self.cancellation_event.set()
        if not self.input_finished:
            self.input_finished = True
            try:
                self.input_queue.put_nowait(_StreamMarker.FINISH)
            except queue.Full:
                pass
        await asyncio.to_thread(self.thread.join)

    def _put_input(self, item: _InputItem) -> None:
        while not self.cancellation_event.is_set():
            try:
                self.input_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _put_output(self, item: _OutputItem) -> None:
        while not self.cancellation_event.is_set():
            try:
                self.output_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        try:
            while not self.cancellation_event.is_set():
                if self.generation_lock.acquire(timeout=KYUTAI_TTS_QUEUE_POLL_SECONDS):
                    break
            else:
                return
            try:
                self._generate()
            finally:
                self.generation_lock.release()
        except Exception as error:
            self._put_output(error)
        finally:
            if self.cancellation_event.is_set():
                try:
                    self.output_queue.put_nowait(_StreamMarker.END)
                except queue.Full:
                    pass
            else:
                self.output_queue.put(_StreamMarker.END)

    def _generate(self) -> None:
        with torch.inference_mode(), self.model.mimi.streaming(1):
            generator = _KyutaiStreamingGenerator(
                model=self.model,
                attributes=(self.attributes,),
                cancellation_event=self.cancellation_event,
                emit=self._put_output,
            )
            first_word = True
            while not self.cancellation_event.is_set():
                item = get_until_cancelled(
                    self.input_queue,
                    self.cancellation_event,
                    KYUTAI_TTS_QUEUE_POLL_SECONDS,
                )
                match item:
                    case None:
                        break
                    case _StreamMarker.FINISH:
                        generator.process_last()
                        break
                    case SynthesisWord():
                        entries = script_to_entries(
                            self.model.tokenizer,
                            self.model.machine.token_ids,
                            self.model.mimi.frame_rate,
                            [item.text],
                            multi_speaker=first_word and self.model.multi_speaker,
                            padding_between=1,
                        )
                        first_word = False
                        generator.append_word(entries, item.text_end)
                        generator.process()


class _KyutaiStreamingGenerator:
    def __init__(
        self,
        model: TTSModel,
        attributes: Sequence[ConditionAttributes],
        cancellation_event: threading.Event,
        emit: Callable[[_OutputItem], None],
    ) -> None:
        if model.cfg_coef != 1.0:
            raise ValueError(
                "The distilled Kyutai TTS checkpoint must use cfg_coef=1.0 at inference."
            )
        if model.lm.condition_provider is None:
            raise ValueError("The Kyutai TTS checkpoint has no condition provider.")
        self.model = model
        self.cancellation_event = cancellation_event
        self.emit = emit
        self.offset = 0
        self.emitted_sample_count = 0
        self.transcript_index = 0
        self.boundary_tracker = TranscriptBoundaryTracker()
        self.state = model.machine.new_state([])
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
        self.lm_generation.streaming_forever(1)

    def append_word(self, entries: Sequence[Entry], text_offset: int) -> None:
        for entry in entries:
            self.state.entries.append(entry)
        immediate_boundaries = self.boundary_tracker.add_source_word(
            tuple(bool(entry.tokens) for entry in entries),
            text_offset,
        )
        for boundary in immediate_boundaries:
            self.emit(boundary)

    def process(self) -> None:
        while len(self.state.entries) > self.model.machine.second_stream_ahead:
            self._step()

    def process_last(self) -> None:
        while (len(self.state.entries) > 0 or self.state.end_step is None) and not (
            self.cancellation_event.is_set()
        ):
            self._step()
        additional_steps = self.model.delay_steps + max(self.model.lm.delays) + 8
        for _ in range(additional_steps):
            if self.cancellation_event.is_set():
                return
            self._step()

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
        frame = self.lm_generation.step(input_tokens)
        self.offset += 1
        self._emit_new_boundaries()
        if frame is None or not bool((frame != -1).all()):
            return
        pcm = self.model.mimi.decode(frame[:, 1:, :]).cpu().float().numpy()
        samples = np.clip(pcm[0, 0], -1.0, 1.0)
        pcm_bytes = (samples * 32_767.0).astype("<i2").tobytes()
        if not pcm_bytes:
            return
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
            # Kyutai defines transcript steps at Mimi's frame rate. The conversion is the protocol
            # anchor; confirm codec priming behavior on the target GPU deployment.
            start_sample = round(
                model_step * self.model.mimi.sample_rate / self.model.mimi.frame_rate
            )
            boundary = self.boundary_tracker.consume_transcript_word(start_sample)
            if boundary is not None:
                self.emit(boundary)
        self.transcript_index += len(new_transcript)
