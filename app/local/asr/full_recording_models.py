from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.local.db.models import TrackSide
from app.shared.asr import AsrModelId, AsrRuntimeStats, TimestampedWord
from app.shared.base_model import FrozenBaseModel


class FullRecordingAsrSampleScope(StrEnum):
    QUALITY_ANALYZED = "quality_analyzed"
    ALL_DATASET_SAMPLES = "all_dataset_samples"


class FullRecordingAsrBatchStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"


class FullRecordingAsrItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FullRecordingAsrBatchRequest(FrozenBaseModel):
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9._-]+$")
    dataset_id: UUID
    sample_scope: FullRecordingAsrSampleScope
    models: tuple[AsrModelId, ...] = Field(min_length=1)


class FullRecordingAsrBatchRecord(FrozenBaseModel):
    id: UUID
    idempotency_key: str
    dataset_id: UUID
    sample_scope: FullRecordingAsrSampleScope
    models: tuple[AsrModelId, ...]
    status: FullRecordingAsrBatchStatus
    total_track_count: int
    pending_track_count: int
    running_track_count: int
    completed_track_count: int
    failed_track_count: int
    recent_errors: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class FullRecordingAsrBatchItem(FrozenBaseModel):
    id: UUID
    batch_id: UUID
    sample_track_id: UUID
    sample_external_id: str
    side: TrackSide
    access_uri: str
    models: tuple[AsrModelId, ...]
    attempt_count: int


class FullRecordingAsrTranscriptRecord(FrozenBaseModel):
    id: UUID
    sample_track_id: UUID
    sample_id: UUID
    sample_external_id: str
    side: TrackSide
    source_audio_sha256: str
    prepared_audio_sha256: str
    audio_filename: str
    model_id: AsrModelId
    transcript_text: str
    words: tuple[TimestampedWord, ...]
    source_duration_seconds: float
    prepared_duration_seconds: float
    processing_time_seconds: float | None
    runtime: AsrRuntimeStats | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class FullRecordingAsrTranscriptPair(FrozenBaseModel):
    sample_id: UUID
    sample_external_id: str
    model_id: AsrModelId
    speaker1: FullRecordingAsrTranscriptRecord
    speaker2: FullRecordingAsrTranscriptRecord
