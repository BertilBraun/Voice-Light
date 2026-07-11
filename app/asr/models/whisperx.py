from __future__ import annotations

from pathlib import Path
from threading import Lock

import whisperx
from faster_whisper import WhisperModel
from torch import nn

from app.asr.models.base import cuda_device, load_time_seconds, timestamped_word_from_word
from app.asr.models.parsing import words_from_whisperx_output
from app.asr.schemas import AsrModelId, TimestampedWord

AlignmentMetadata = dict[str, str | int | float | bool | None]
AlignmentResources = tuple[nn.Module, AlignmentMetadata]


class WhisperxAsrModel:
    model_id = AsrModelId.WHISPERX
    package_names = ("torch", "whisperx", "faster-whisper")

    def __init__(self) -> None:
        self.alignment_load_lock = Lock()
        self.inference_lock = Lock()
        self.alignment_models: dict[str, AlignmentResources] = {}
        self.model_loading_time_seconds = load_time_seconds(self.load)

    def load(self) -> None:
        self.device = cuda_device()
        self.model = WhisperModel("large-v3", device=self.device, compute_type="float16")

    def transcribe(self, audio_path: Path) -> tuple[TimestampedWord, ...]:
        with self.inference_lock:
            segments, transcription_info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=False,
                vad_filter=False,
            )
            transcription_segments = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ]
            language_code = str(transcription_info.language)
            align_model, metadata = self.alignment_model_for_language(language_code=language_code)
            audio = whisperx.load_audio(str(audio_path))
            aligned = whisperx.align(
                transcription_segments,
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
        return tuple(
            timestamped_word_from_word(word) for word in words_from_whisperx_output(aligned)
        )

    def alignment_model_for_language(self, language_code: str) -> AlignmentResources:
        with self.alignment_load_lock:
            cached_alignment_model = self.alignment_models.get(language_code)
            if cached_alignment_model is not None:
                return cached_alignment_model
            align_model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
            )
            alignment_resources = (align_model, metadata)
            self.alignment_models[language_code] = alignment_resources
            return alignment_resources
