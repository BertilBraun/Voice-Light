from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AsrModelId(StrEnum):
    PARAKEET_TDT = "parakeet_tdt_0_6b_v3"
    WHISPERX = "whisperx_large_v3"


class TimestampedWord(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


class AsrRuntimeStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    processing_time_seconds: float
    model_loading_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    real_time_factor: float | None = None
    peak_gpu_memory_mb: float | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)


class AsrTranscriptResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: AsrModelId
    text: str
    words: tuple[TimestampedWord, ...]
    processing_time_seconds: float | None = None
    error: str | None = None
    runtime: AsrRuntimeStats | None = None


class CachedAsrRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_path: str
    models: tuple[AsrModelId, ...] = Field(min_length=1)


class CachedAsrResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_sha256: str
    results: tuple[AsrTranscriptResult, ...]


class RemoteAsrRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_sha256: str
    audio_base64: str
    models: tuple[AsrModelId, ...] = Field(min_length=1)


class RemoteAsrResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: tuple[AsrTranscriptResult, ...]
