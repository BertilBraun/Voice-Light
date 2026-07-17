from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

import librosa
from torch import Tensor
from transformers import AutoModelForRNNT, AutoProcessor

from app.compute.asr.models.base import ModelTranscription, cuda_device, load_time_seconds
from app.shared.asr import (
    NEMOTRON_3_5_IDENTIFIER,
    NEMOTRON_3_5_REVISION,
    AsrModelId,
    TimestampedWord,
)


class NemotronAsrModel:
    model_id = AsrModelId.NEMOTRON_3_5
    package_names = ("torch", "transformers", "librosa")

    def __init__(self) -> None:
        self.inference_lock = Lock()
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            NEMOTRON_3_5_IDENTIFIER,
            revision=NEMOTRON_3_5_REVISION,
        )
        self.model = AutoModelForRNNT.from_pretrained(
            NEMOTRON_3_5_IDENTIFIER,
            revision=NEMOTRON_3_5_REVISION,
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
            inputs = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                language="en-US",
                return_tensors="pt",
            )
            inputs.to(self.model.device, dtype=self.model.dtype)
            output = self.model.generate(**inputs, return_dict_in_generate=True)
            decoded_text = self.decoded_text(output.sequences)
            duration_seconds = float(librosa.get_duration(path=str(audio_path)))
            model_execution_time_seconds = time.perf_counter() - model_execution_start
        words = (
            ()
            if not decoded_text.strip()
            else (
                TimestampedWord(
                    text=decoded_text.strip(),
                    start_seconds=0.0,
                    end_seconds=duration_seconds,
                ),
            )
        )
        return ModelTranscription(
            words=words,
            inference_queue_time_seconds=inference_queue_time_seconds,
            audio_loading_time_seconds=audio_loading_time_seconds,
            model_execution_time_seconds=model_execution_time_seconds,
        )

    def decoded_text(self, sequences: Tensor) -> str:
        decoded = self.processor.decode(sequences, skip_special_tokens=True)
        if isinstance(decoded, str):
            return decoded
        return str(decoded[0])
