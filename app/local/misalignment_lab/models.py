from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.local.training_samples.models import PreviewWaveformPoint
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import (
    AnnotationPoint,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    SegmentAnnotationTarget,
    SpeakerSide,
)


class MisalignmentJudgment(StrEnum):
    PLAUSIBLY_ALIGNED = "plausibly_aligned"
    LIKELY_MISALIGNED = "likely_misaligned"
    UNSURE = "unsure"


class InteractionWindowMetrics(FrozenBaseModel):
    alternating_speaker_boundaries: int
    rapid_speaker_boundaries: int
    turn_count: int
    backchannel_count: int
    interruption_count: int
    both_speakers_active_seconds: float
    interaction_score: float


class MisalignmentCandidateSummary(FrozenBaseModel):
    candidate_id: UUID
    sample_id: UUID
    external_id: str
    window_start_seconds: float
    window_end_seconds: float
    seconds_from_recording_end: float
    interaction: InteractionWindowMetrics
    suspicion_score: float
    audit_anomaly_score: float | None
    audit_late_shift_seconds: float | None
    duration_mismatch_seconds: float | None
    sampling_weight: float


class MisalignmentWindowAnnotation(FrozenBaseModel):
    side: SpeakerSide
    speech_segments: tuple[AnnotationSpan, ...]
    pauses: tuple[AnnotationSpan, ...]
    backchannels: tuple[AnnotationSpan, ...]
    turns: tuple[AnnotationPoint, ...]
    interruptions: tuple[AnnotationPoint, ...]
    segment_targets: tuple[SegmentAnnotationTarget, ...]
    connection_targets: tuple[ConnectionAnnotationTarget, ...]


class MisalignmentCandidatePreview(FrozenBaseModel):
    candidate: MisalignmentCandidateSummary
    annotation_version: str
    speaker1_waveform: tuple[PreviewWaveformPoint, ...]
    speaker2_waveform: tuple[PreviewWaveformPoint, ...]
    speaker1: MisalignmentWindowAnnotation
    speaker2: MisalignmentWindowAnnotation


class MisalignmentLabProgress(FrozenBaseModel):
    reviewed_snippet_count: int
    plausibly_aligned_count: int
    likely_misaligned_count: int
    unsure_count: int
    quarantined_session_count: int
    eligible_session_count: int


class MisalignmentQueueResponse(FrozenBaseModel):
    seed: str
    requested_count: int
    candidates: tuple[MisalignmentCandidateSummary, ...]
    progress: MisalignmentLabProgress


class MisalignmentJudgmentRequest(FrozenBaseModel):
    candidate_id: UUID
    sample_id: UUID
    window_start_seconds: float
    window_end_seconds: float
    judgment: MisalignmentJudgment
    queue_seed: str = Field(min_length=1, max_length=100)


class MisalignmentStoredJudgment(FrozenBaseModel):
    candidate_id: UUID
    sample_id: UUID
    external_id: str
    window_start_seconds: float
    window_end_seconds: float
    judgment: MisalignmentJudgment
    queue_seed: str
    created_at: datetime
    updated_at: datetime


class MisalignmentJudgmentResponse(FrozenBaseModel):
    stored: MisalignmentStoredJudgment
    session_quarantined: bool
    progress: MisalignmentLabProgress
