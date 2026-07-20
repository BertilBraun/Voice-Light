from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from app.shared.asr import AsrModelId, AsrTranscriptResult, TimestampedWord
from app.shared.audio.s3 import S3AudioSource
from app.shared.base_model import FrozenBaseModel
from app.shared.language import LanguageProbeWindow, TrackLanguageStatus
from app.shared.quality import AudioMetadata, QualityResult, TrackVadResult


class HealthStatus(StrEnum):
    LIVE = "live"
    READY = "ready"
    NOT_READY = "not_ready"


class ModelStageStatus(StrEnum):
    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ModelStage(FrozenBaseModel):
    name: str
    status: ModelStageStatus
    load_time_seconds: float | None = None
    error: str | None = None


class GpuMemory(FrozenBaseModel):
    current_allocated_mb: float | None
    peak_allocated_mb: float | None


class LivenessResponse(FrozenBaseModel):
    status: Literal[HealthStatus.LIVE] = HealthStatus.LIVE


class ReadinessResponse(FrozenBaseModel):
    status: Literal[HealthStatus.READY, HealthStatus.NOT_READY]
    stages: tuple[ModelStage, ...]
    gpu_memory: GpuMemory


class QualityAnalysisUpload(FrozenBaseModel):
    sample_id: str
    speaker1_original_metadata: AudioMetadata | None
    speaker2_original_metadata: AudioMetadata | None


class QualityAnalysisResponse(FrozenBaseModel):
    speaker1_metadata: AudioMetadata
    speaker2_metadata: AudioMetadata
    quality_result: QualityResult


class MaterializedAudio(FrozenBaseModel):
    source: S3AudioSource
    filename: str
    content_sha256: str
    metadata: AudioMetadata


class DatasetLanguageRequest(FrozenBaseModel):
    speaker1: S3AudioSource
    speaker2: S3AudioSource


class DatasetLanguageTrackResponse(FrozenBaseModel):
    audio: MaterializedAudio
    status: TrackLanguageStatus
    language_code: str | None
    confidence: float | None
    transcript_word_count: int
    transcript_text: str
    probe_windows: tuple[LanguageProbeWindow, ...]
    error: str | None


class DatasetLanguageResponse(FrozenBaseModel):
    speaker1: DatasetLanguageTrackResponse
    speaker2: DatasetLanguageTrackResponse


class DatasetAsrRequest(FrozenBaseModel):
    source: S3AudioSource
    models: tuple[AsrModelId, ...] = Field(min_length=1)


class DatasetAsrResponse(FrozenBaseModel):
    audio: MaterializedAudio
    prepared_audio_sha256: str
    prepared_duration_seconds: float
    results: tuple[AsrTranscriptResult, ...]


class DatasetTrackTranscripts(FrozenBaseModel):
    parakeet: AsrTranscriptResult
    canary: AsrTranscriptResult


class DatasetQualityRequest(FrozenBaseModel):
    sample_id: str
    speaker1: S3AudioSource
    speaker2: S3AudioSource
    speaker1_transcripts: DatasetTrackTranscripts
    speaker2_transcripts: DatasetTrackTranscripts


class DatasetQualityResponse(FrozenBaseModel):
    speaker1_audio: MaterializedAudio
    speaker2_audio: MaterializedAudio
    vad_version: str
    speaker1_vad: TrackVadResult
    speaker2_vad: TrackVadResult
    speaker1_filtered_words: tuple[TimestampedWord, ...]
    speaker2_filtered_words: tuple[TimestampedWord, ...]
    quality_result: QualityResult
