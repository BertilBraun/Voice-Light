from __future__ import annotations

from pathlib import Path
from threading import Lock

import librosa
from torch import Tensor
from transformers import AutoModelForRNNT, AutoProcessor

from app.asr.models.base import cuda_device, load_time_seconds, timestamped_word_from_word
from app.asr.models.parsing import NEMOTRON_3_5_IDENTIFIER
from app.asr.models.text_alignment import WhisperxTextAligner
from app.asr.schemas import AsrModelId, TimestampedWord


class NemotronAsrModel:
    model_id = AsrModelId.NEMOTRON_3_5
    package_names = ("torch", "transformers", "librosa", "whisperx")

    def __init__(self) -> None:
        self.inference_lock = Lock()
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(NEMOTRON_3_5_IDENTIFIER)
        self.model = AutoModelForRNNT.from_pretrained(NEMOTRON_3_5_IDENTIFIER, dtype="auto")
        self.device = cuda_device()
        self.model.to(self.device)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self.aligner = WhisperxTextAligner(device=self.device)

    def transcribe(self, audio_path: Path) -> tuple[TimestampedWord, ...]:
        with self.inference_lock:
            audio, _sample_rate = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            inputs = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                language="en-US",
                return_tensors="pt",
            )
            inputs.to(self.model.device, dtype=self.model.dtype)
            output = self.model.generate(**inputs, return_dict_in_generate=True)
            decoded_text = self.decoded_text(output.sequences)
            words = self.aligner.align_text(
                audio_path=audio_path,
                text=decoded_text,
                language_code="en",
                audio_duration_seconds=float(librosa.get_duration(path=str(audio_path))),
            )
        return tuple(timestamped_word_from_word(word) for word in words)

    def decoded_text(self, sequences: Tensor) -> str:
        decoded = self.processor.decode(sequences, skip_special_tokens=True)
        if isinstance(decoded, str):
            return decoded
        return str(decoded[0])
