from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import whisperx
from torch import nn

from app.asr.models.parsing import words_from_whisperx_output
from app.asr.transcript import Word

AlignmentMetadata = dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class AlignmentResources:
    align_model: nn.Module
    metadata: AlignmentMetadata


class WhisperxTextAligner:
    def __init__(self, device: str) -> None:
        self.alignment_load_lock = Lock()
        self.alignment_models: dict[str, AlignmentResources] = {}
        self.device = device

    def align_text(
        self,
        audio_path: Path,
        text: str,
        language_code: str,
        audio_duration_seconds: float,
    ) -> tuple[Word, ...]:
        alignment_resources = self.alignment_model_for_language(language_code=language_code)
        audio = whisperx.load_audio(str(audio_path))
        aligned = whisperx.align(
            [
                {
                    "start": 0.0,
                    "end": audio_duration_seconds,
                    "text": text,
                }
            ],
            alignment_resources.align_model,
            alignment_resources.metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        return tuple(words_from_whisperx_output(aligned))

    def align_segments(
        self,
        audio_path: Path,
        segments: list[dict[str, str | float]],
        language_code: str,
    ) -> tuple[Word, ...]:
        alignment_resources = self.alignment_model_for_language(language_code=language_code)
        audio = whisperx.load_audio(str(audio_path))
        aligned = whisperx.align(
            segments,
            alignment_resources.align_model,
            alignment_resources.metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        return tuple(words_from_whisperx_output(aligned))

    def alignment_model_for_language(self, language_code: str) -> AlignmentResources:
        with self.alignment_load_lock:
            cached_alignment_model = self.alignment_models.get(language_code)
            if cached_alignment_model is not None:
                return cached_alignment_model
            align_model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
            )
            alignment_resources = AlignmentResources(align_model=align_model, metadata=metadata)
            self.alignment_models[language_code] = alignment_resources
            return alignment_resources
