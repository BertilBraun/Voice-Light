from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from app.shared.base_model import FrozenBaseModel


class OffsetPattern(StrEnum):
    CONSTANT = "constant"
    VARIABLE = "variable"
    UNCERTAIN = "uncertain"


class SynchronizationEvidenceSource(StrEnum):
    CONVERSATION_ANNOTATION = "conversation_annotation"
    PARAKEET = "parakeet"
    CANARY = "canary"


class GainMeasurementBasis(StrEnum):
    WHOLE_TRACK_ESTIMATE = "whole_track_estimate"
    ANNOTATED_SPEECH = "annotated_speech"


class SynchronizationEvidence(FrozenBaseModel):
    source: SynchronizationEvidenceSource
    estimated_b_shift_seconds: float
    bad_state_improvement: float
    overlap_reduction: float
    silence_reduction: float


class SynchronizationWindowEstimate(FrozenBaseModel):
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
    estimated_b_shift_seconds: float
    full_recording_estimated_b_shift_seconds: float
    offset_pattern: OffsetPattern
    source_agreement: bool
    evidence: tuple[SynchronizationEvidence, ...]
    window_estimates: tuple[SynchronizationWindowEstimate, ...]
    overlap_silence_cycle_count: int
    speaker1_gain: TrackGainNormalization
    speaker2_gain: TrackGainNormalization


class SynchronizationCandidateListResponse(FrozenBaseModel):
    candidates: tuple[SynchronizationCandidate, ...]
    analyzed_session_count: int
    excluded_session_count: int


class SynchronizationGainResponse(FrozenBaseModel):
    sample_id: UUID
    speaker1_gain: TrackGainNormalization
    speaker2_gain: TrackGainNormalization
