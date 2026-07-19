from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.shared.base_model import FrozenBaseModel


class OffsetPattern(StrEnum):
    CONSTANT = "constant"
    VARIABLE = "variable"
    UNCERTAIN = "uncertain"


class SynchronizationAuditKind(StrEnum):
    TEMPORAL_CHANGE = "temporal_change"
    STABLE_OFFSET = "stable_offset"
    UNCERTAIN = "uncertain"


class SynchronizationEvidenceSource(StrEnum):
    AUDIO_ACTIVITY = "audio_activity"
    CONVERSATION_ANNOTATION = "conversation_annotation"
    PARAKEET = "parakeet"
    CANARY = "canary"


class GainMeasurementBasis(StrEnum):
    WHOLE_TRACK_ESTIMATE = "whole_track_estimate"
    ANNOTATED_SPEECH = "annotated_speech"


class AlignmentEstimateOrigin(StrEnum):
    PREDICTED = "predicted"
    REVIEWED = "reviewed"
    UNRESOLVED = "unresolved"


class SynchronizationEvidence(FrozenBaseModel):
    source: SynchronizationEvidenceSource
    estimated_b_shift_seconds: float
    bad_state_improvement: float
    overlap_reduction: float
    silence_reduction: float


class SynchronizationWindowEstimate(FrozenBaseModel):
    source: SynchronizationEvidenceSource
    start_seconds: float
    end_seconds: float
    estimated_b_shift_seconds: float
    bad_state_improvement: float
    overlap_reduction: float
    silence_reduction: float
    meaningful: bool


class TrackGainNormalization(FrozenBaseModel):
    measurement_basis: GainMeasurementBasis
    measured_rms_dbfs: float | None
    estimated_active_rms_dbfs: float | None
    measured_speech_duration_seconds: float | None
    target_active_rms_dbfs: float
    default_gain: float


class SynchronizationCandidate(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    likelihood_score: float
    is_offset_candidate: bool
    estimated_b_shift_seconds: float
    full_recording_estimated_b_shift_seconds: float
    alignment_estimate_origin: AlignmentEstimateOrigin
    offset_confidence_score: float
    offset_pattern: OffsetPattern
    static_offset_valid: bool
    drift_warning: str | None
    duration_mismatch_seconds: float | None
    source_agreement: bool
    evidence: tuple[SynchronizationEvidence, ...]
    window_estimates: tuple[SynchronizationWindowEstimate, ...]
    overlap_silence_cycle_count: int
    speaker1_gain: TrackGainNormalization
    speaker2_gain: TrackGainNormalization


class SynchronizationCandidateListResponse(FrozenBaseModel):
    candidates: tuple[SynchronizationCandidate, ...]
    analyzed_session_count: int
    offset_candidate_count: int
    excluded_session_count: int


class SynchronizationAuditWindow(FrozenBaseModel):
    start_seconds: float
    end_seconds: float
    estimated_b_shift_seconds: float
    confidence_score: float
    bad_state_improvement: float
    competing_margin: float
    basin_width_seconds: float
    persistence_window_count: int
    agreeing_transcript_sources: tuple[SynchronizationEvidenceSource, ...]
    accepted: bool
    maximum_lag_boundary: bool


class SynchronizationAuditResult(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    kind: SynchronizationAuditKind
    anomaly_score: float
    strongest_window_start_seconds: float
    strongest_window_end_seconds: float
    strongest_shift_seconds: float
    temporal_shift_range_seconds: float
    summary: str
    windows: tuple[SynchronizationAuditWindow, ...]


class SynchronizationAuditReport(FrozenBaseModel):
    estimator_version: str
    window_duration_seconds: float
    window_hop_seconds: float
    analyzed_session_count: int
    generated_at: str
    results: tuple[SynchronizationAuditResult, ...]


class SynchronizationGainResponse(FrozenBaseModel):
    sample_id: UUID
    speaker1_gain: TrackGainNormalization
    speaker2_gain: TrackGainNormalization


class SynchronizationReviewRequest(FrozenBaseModel):
    speaker2_shift_seconds: float = Field(ge=-12.0, le=12.0)


class SynchronizationReviewSaveResponse(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    speaker2_shift_seconds: float
