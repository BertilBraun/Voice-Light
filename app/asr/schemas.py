from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from app.frozen_base_config import FrozenBaseModel


class AsrModelId(StrEnum):
    PARAKEET_TDT = "parakeet_tdt_0_6b_v3"
    WHISPERX = "whisperx_large_v3"


class TimestampedWord(FrozenBaseModel):
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


class AsrRuntimeStats(FrozenBaseModel):
    processing_time_seconds: float
    model_loading_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    real_time_factor: float | None = None
    peak_gpu_memory_mb: float | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)


class AsrTranscriptResult(FrozenBaseModel):
    model_id: AsrModelId
    text: str
    words: tuple[TimestampedWord, ...]
    processing_time_seconds: float | None = None
    error: str | None = None
    runtime: AsrRuntimeStats | None = None


class CachedAsrRequest(FrozenBaseModel):
    audio_path: str
    models: tuple[AsrModelId, ...] = Field(min_length=1)


class CachedAsrResponse(FrozenBaseModel):
    audio_sha256: str
    results: tuple[AsrTranscriptResult, ...]


class RemoteAsrRequest(FrozenBaseModel):
    audio_sha256: str
    audio_base64: str
    models: tuple[AsrModelId, ...] = Field(min_length=1)


class RemoteAsrResponse(FrozenBaseModel):
    results: tuple[AsrTranscriptResult, ...]
