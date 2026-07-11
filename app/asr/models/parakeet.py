from __future__ import annotations

from pathlib import Path
from threading import Lock

import librosa
import torch
from transformers import AutoModelForTDT, AutoProcessor

from app.analyses.asr.runners import PARAKEET_IDENTIFIER, words_from_parakeet_timestamps
from app.asr.models.base import load_time_seconds, timestamped_word_from_word
from app.asr.schemas import AsrModelId, TimestampedWord


class ParakeetAsrModel:
    model_id = AsrModelId.PARAKEET_TDT
    package_names = ("torch", "transformers", "librosa")

    def __init__(self) -> None:
        self.inference_lock = Lock()
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(PARAKEET_IDENTIFIER)
        self.model = AutoModelForTDT.from_pretrained(PARAKEET_IDENTIFIER, dtype="auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def transcribe(self, audio_path: Path) -> tuple[TimestampedWord, ...]:
        with self.inference_lock:
            audio, _sample_rate = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            inputs = self.processor([audio], sampling_rate=self.sample_rate)
            inputs.to(self.model.device, dtype=self.model.dtype)
            output = self.model.generate(**inputs, return_dict_in_generate=True)
            _decoded_output, decoded_timestamps = self.processor.decode(
                output.sequences,
                durations=output.durations,
                skip_special_tokens=True,
            )
        return tuple(
            timestamped_word_from_word(word)
            for word in words_from_parakeet_timestamps(decoded_timestamps)
        )
