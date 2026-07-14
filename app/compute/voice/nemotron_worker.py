from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import Final, TextIO, cast

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingForRNNT,
)
from transformers.models.nemotron_asr_streaming.processing_nemotron_asr_streaming import (
    NemotronAsrStreamingProcessor,
)

from app.compute.voice.asr_worker_protocol import (
    AsrAudioCommand,
    AsrWorkerCommandType,
    AsrWorkerErrorEvent,
    AsrWorkerReadyEvent,
    FinalAsrEvent,
    PartialAsrEvent,
    asr_worker_command_adapter,
)

MODEL_NAME: Final = "nvidia/nemotron-speech-streaming-en-0.6b"
MODEL_REVISION: Final = "df1f0fe9dfdf05152936192b4c8c7653d53bf557"
INPUT_SAMPLE_RATE: Final = 16_000
PCM_BYTES_PER_SAMPLE: Final = 2
NUM_LOOKAHEAD_TOKENS: Final = int(os.environ.get("VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS", "1"))


@dataclass(frozen=True)
class AudioRead:
    samples: np.ndarray
    reached_end: bool


class StreamingAudioBuffer:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.pcm_bytes = bytearray()
        self.closed = False

    def append(self, pcm_bytes: bytes) -> None:
        with self.condition:
            self.pcm_bytes.extend(pcm_bytes)
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def read(self, sample_count: int) -> AudioRead:
        byte_count = sample_count * PCM_BYTES_PER_SAMPLE
        with self.condition:
            self.condition.wait_for(lambda: len(self.pcm_bytes) >= byte_count or self.closed)
            consumed_byte_count = min(byte_count, len(self.pcm_bytes))
            pcm_bytes = bytes(self.pcm_bytes[:consumed_byte_count])
            del self.pcm_bytes[:consumed_byte_count]
            reached_end = self.closed and not self.pcm_bytes
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32_768.0
        return AudioRead(samples=samples, reached_end=reached_end)


class NemotronRecognition:
    def __init__(
        self,
        processor: NemotronAsrStreamingProcessor,
        model: NemotronAsrStreamingForRNNT,
        audio_buffer: StreamingAudioBuffer,
        output_stream: TextIO,
        output_lock: threading.Lock,
    ) -> None:
        self.processor = processor
        self.model = model
        self.audio_buffer = audio_buffer
        self.output_stream = output_stream
        self.output_lock = output_lock

    def run(self) -> None:
        try:
            first_audio = self.audio_buffer.read(self.processor.num_samples_first_audio_chunk)
            if not first_audio.samples.size:
                self._send_event(FinalAsrEvent(text=""))
                return
            padded_first_audio = _pad_samples(
                first_audio.samples,
                self.processor.num_samples_first_audio_chunk,
            )
            first_inputs = self.processor(
                padded_first_audio,
                sampling_rate=INPUT_SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=True,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)
            input_features = self._input_features(
                first_audio=padded_first_audio,
                first_input_features=first_inputs.input_features,
                first_audio_reached_end=first_audio.reached_end,
            )
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_special_tokens=True,
            )
            generation_errors: list[Exception] = []
            generate = partial(
                self.model.generate,
                input_features=input_features,
                attention_mask=first_inputs.attention_mask,
                num_lookahead_tokens=first_inputs.num_lookahead_tokens,
                streamer=streamer,
            )

            def run_generation() -> None:
                try:
                    generate()
                except Exception as error:
                    generation_errors.append(error)
                    streamer.on_finalized_text("", stream_end=True)

            generation_thread = threading.Thread(target=run_generation, daemon=True)
            generation_thread.start()
            transcript_parts: list[str] = []
            for text_delta in streamer:
                transcript_parts.append(text_delta)
                self._send_event(PartialAsrEvent(text="".join(transcript_parts).strip()))
            generation_thread.join()
            if generation_errors:
                raise generation_errors[0]
            self._send_event(FinalAsrEvent(text="".join(transcript_parts).strip()))
        except Exception as error:
            self._send_event(AsrWorkerErrorEvent(message=str(error)))

    def _input_features(
        self,
        first_audio: np.ndarray,
        first_input_features: torch.Tensor,
        first_audio_reached_end: bool,
    ) -> Iterator[torch.Tensor]:
        yield first_input_features[:, : self.processor.num_mel_frames_first_audio_chunk, :]
        if first_audio_reached_end:
            return
        overlap_sample_count = self.processor.feature_extractor.n_fft // 2
        previous_tail = first_audio[-overlap_sample_count:]
        new_sample_count = self.processor.num_samples_per_audio_chunk - overlap_sample_count
        while True:
            audio_read = self.audio_buffer.read(new_sample_count)
            if not audio_read.samples.size:
                return
            audio_chunk = np.concatenate((previous_tail, audio_read.samples))
            audio_chunk = _pad_samples(audio_chunk, self.processor.num_samples_per_audio_chunk)
            inputs = self.processor(
                audio_chunk,
                sampling_rate=INPUT_SAMPLE_RATE,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)
            yield inputs.input_features
            previous_tail = audio_chunk[-overlap_sample_count:]
            if audio_read.reached_end:
                return

    def _send_event(self, event: PartialAsrEvent | FinalAsrEvent | AsrWorkerErrorEvent) -> None:
        with self.output_lock:
            self.output_stream.write(event.model_dump_json() + "\n")
            self.output_stream.flush()


def main() -> None:
    processor = cast(
        NemotronAsrStreamingProcessor,
        AutoProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION),
    )
    processor.set_num_lookahead_tokens(NUM_LOOKAHEAD_TOKENS)
    model = cast(
        NemotronAsrStreamingForRNNT,
        AutoModelForRNNT.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
        ).to("cuda"),
    )
    output_lock = threading.Lock()
    with output_lock:
        sys.stdout.write(AsrWorkerReadyEvent().model_dump_json() + "\n")
        sys.stdout.flush()
    audio_buffer: StreamingAudioBuffer | None = None
    recognition_thread: threading.Thread | None = None

    for command_json in sys.stdin:
        command = asr_worker_command_adapter.validate_json(command_json)
        match command.type:
            case AsrWorkerCommandType.START:
                if recognition_thread is not None and recognition_thread.is_alive():
                    _send_controller_error("An ASR session is already active.", output_lock)
                    continue
                audio_buffer = StreamingAudioBuffer()
                recognition = NemotronRecognition(
                    processor=processor,
                    model=model,
                    audio_buffer=audio_buffer,
                    output_stream=sys.stdout,
                    output_lock=output_lock,
                )
                recognition_thread = threading.Thread(target=recognition.run, daemon=True)
                recognition_thread.start()
            case AsrWorkerCommandType.AUDIO:
                assert isinstance(command, AsrAudioCommand)
                if audio_buffer is None:
                    _send_controller_error("No ASR session is active.", output_lock)
                    continue
                audio_buffer.append(command.pcm_bytes())
            case AsrWorkerCommandType.FINISH:
                if audio_buffer is None or recognition_thread is None:
                    _send_controller_error("No ASR session is active.", output_lock)
                    continue
                audio_buffer.close()
                recognition_thread.join()
                audio_buffer = None
                recognition_thread = None
            case AsrWorkerCommandType.SHUTDOWN:
                if audio_buffer is not None:
                    audio_buffer.close()
                if recognition_thread is not None:
                    recognition_thread.join()
                return


def _pad_samples(samples: np.ndarray, sample_count: int) -> np.ndarray:
    if len(samples) >= sample_count:
        return samples
    return np.pad(samples, (0, sample_count - len(samples)))


def _send_controller_error(message: str, output_lock: threading.Lock) -> None:
    with output_lock:
        sys.stdout.write(AsrWorkerErrorEvent(message=message).model_dump_json() + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
