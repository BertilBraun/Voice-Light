from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.frozen_base_config import FrozenBaseModel


class DatasetStorageKind(StrEnum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


class TrackSide(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AsrRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetCreate(FrozenBaseModel):
    name: str
    storage_kind: DatasetStorageKind = DatasetStorageKind.LOCAL
    root_uri: str
    description: str = ""


class DatasetRecord(FrozenBaseModel):
    id: UUID
    name: str
    storage_kind: DatasetStorageKind
    root_uri: str
    description: str
    created_at: datetime
    updated_at: datetime


class SampleRecord(FrozenBaseModel):
    id: UUID
    dataset_id: UUID
    external_id: str
    duration_seconds: float | None
    quality_score: float | None
    quality_flags: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


class SampleTrackRecord(FrozenBaseModel):
    id: UUID
    sample_id: UUID
    side: TrackSide
    speaker_index: int
    storage_uri: str
    access_uri: str
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    sample_count: int | None
    created_at: datetime
    updated_at: datetime


class AudioMetadataRecord(FrozenBaseModel):
    id: UUID
    sample_track_id: UUID
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_count: int | None
    payload: dict[str, object]
    created_at: datetime
    updated_at: datetime


class QualityResultRecord(FrozenBaseModel):
    id: UUID
    sample_id: UUID
    metric_version: str
    status: str
    total_quality_score: float | None
    raw_quality_score: float | None
    calibrated_quality_score: float | None
    interaction_density_score: float | None
    timing_reliability_score: float | None
    audio_quality_score: float | None
    speech_ratio: float | None
    silence_ratio: float | None
    overlap_ratio: float | None
    duration_mismatch_seconds: float | None
    track_correlation: float | None
    energy_envelope_correlation: float | None
    flags: tuple[str, ...]
    payload: dict[str, object]
    created_at: datetime
    updated_at: datetime


class AsrRunRecord(FrozenBaseModel):
    id: UUID
    sample_id: UUID
    model_name: str
    status: AsrRunStatus
    runtime_seconds: float | None
    real_time_factor: float | None
    error: str | None
    payload: dict[str, object]
    created_at: datetime
    updated_at: datetime


class AsrWordRecord(FrozenBaseModel):
    id: UUID
    asr_run_id: UUID
    side: TrackSide
    word_index: int
    text: str
    start_seconds: float | None
    end_seconds: float | None
    confidence: float | None
    created_at: datetime


class AsrEvaluationRecord(FrozenBaseModel):
    id: UUID
    asr_run_id: UUID
    wer: float | None
    substitutions: int
    insertions: int
    deletions: int
    reference_word_count: int
    start_median_absolute_error: float | None
    start_mean_absolute_error: float | None
    start_p90_absolute_error: float | None
    end_median_absolute_error: float | None
    end_mean_absolute_error: float | None
    end_p90_absolute_error: float | None
    payload: dict[str, object]
    created_at: datetime
    updated_at: datetime


class IngestionJobRecord(FrozenBaseModel):
    id: UUID
    dataset_id: UUID | None
    status: JobStatus
    source_uri: str
    message: str
    total_samples: int
    processed_samples: int
    failed_samples: int
    error: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


class SampleListFilter(FrozenBaseModel):
    dataset_id: UUID | None = None
    quality_min: float | None = None
    quality_max: float | None = None
    duration_min: float | None = None
    duration_max: float | None = None
    flag: str | None = None
    speech_ratio_min: float | None = None
    speech_ratio_max: float | None = None
    overlap_ratio_min: float | None = None
    overlap_ratio_max: float | None = None
    silence_ratio_min: float | None = None
    silence_ratio_max: float | None = None
    asr_model: str | None = None
    wer_min: float | None = None
    wer_max: float | None = None
    timestamp_p90_max: float | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class DashboardSample(FrozenBaseModel):
    sample: SampleRecord
    tracks: tuple[SampleTrackRecord, ...]
    latest_quality: QualityResultRecord | None
    latest_asr_run: AsrRunRecord | None
    latest_asr_evaluation: AsrEvaluationRecord | None
