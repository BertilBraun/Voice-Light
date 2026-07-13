from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

from pydantic import Field

from app.shared.base_model import FrozenBaseModel

JsonScalar: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, JsonScalar]


class SpeakerTrack(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"


class Word(FrozenBaseModel):
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


class ReferenceTranscript(FrozenBaseModel):
    audio_path: str
    words: tuple[Word, ...]


class RuntimeStats(FrozenBaseModel):
    audio_duration_seconds: float
    processing_time_seconds: float
    model_loading_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    real_time_factor: float | None = None
    peak_gpu_memory_mb: float | None = None


class TranscriptionResult(FrozenBaseModel):
    model_name: str
    audio_path: str
    track: SpeakerTrack
    audio_duration_seconds: float
    processing_time_seconds: float
    words: tuple[Word, ...]
    raw_output: JsonObject = Field(default_factory=dict)
    model_identifier: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    model_loading_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    real_time_factor: float | None = None
    peak_gpu_memory_mb: float | None = None
    error: str | None = None
