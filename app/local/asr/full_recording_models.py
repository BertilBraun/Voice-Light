from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.local.db.models import TrackSide
from app.shared.asr import AsrModelId, AsrRuntimeStats, TimestampedWord
from app.shared.base_model import FrozenBaseModel


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
