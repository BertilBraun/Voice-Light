from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

import librosa
from transformers import AutoModelForTDT, AutoProcessor

from app.compute.asr.chunking import (
    ParakeetAudioChunk,
    global_chunk_words,
    parakeet_audio_chunks,
)
from app.compute.asr.models.base import ModelTranscription, cuda_device, load_time_seconds
from app.compute.asr.models.parsing import (
    words_from_parakeet_timestamps,
)
from app.shared.asr import PARAKEET_IDENTIFIER, PARAKEET_REVISION, AsrModelId, TimestampedWord


class ParakeetAsrModel:
    model_id = AsrModelId.PARAKEET_TDT
    package_names = ("torch", "transformers", "librosa")

    def __init__(self) -> None:
        self.inference_lock = Lock()
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            PARAKEET_IDENTIFIER,
            revision=PARAKEET_REVISION,
        )
        self.model = AutoModelForTDT.from_pretrained(
            PARAKEET_IDENTIFIER,
            revision=PARAKEET_REVISION,
            dtype="auto",
        )
        self.device = cuda_device()
        self.model.to(self.device)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def transcribe(self, audio_path: Path) -> ModelTranscription:
        queue_start = time.perf_counter()
        with self.inference_lock:
            inference_queue_time_seconds = time.perf_counter() - queue_start
            audio_loading_start = time.perf_counter()
            audio, _sample_rate = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            audio_loading_time_seconds = time.perf_counter() - audio_loading_start
            model_execution_start = time.perf_counter()
            words = tuple(
                word
                for chunk in parakeet_audio_chunks(audio, self.sample_rate)
                for word in self._transcribe_chunk(chunk)
            )
            model_execution_time_seconds = time.perf_counter() - model_execution_start
        return ModelTranscription(
            words=words,
            inference_queue_time_seconds=inference_queue_time_seconds,
            audio_loading_time_seconds=audio_loading_time_seconds,
            model_execution_time_seconds=model_execution_time_seconds,
        )

    def _transcribe_chunk(self, chunk: ParakeetAudioChunk) -> tuple[TimestampedWord, ...]:
        inputs = self.processor([chunk.samples], sampling_rate=self.sample_rate)
        inputs.to(self.model.device, dtype=self.model.dtype)
        output = self.model.generate(**inputs, return_dict_in_generate=True)
        _decoded_output, decoded_timestamps = self.processor.decode(
            output.sequences,
            durations=output.durations,
            skip_special_tokens=True,
        )
        return global_chunk_words(
            chunk=chunk,
            words=tuple(words_from_parakeet_timestamps(decoded_timestamps)),
        )
