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


class MisalignmentRepairJudgment(StrEnum):
    PLAUSIBLE = "plausible"
    NOT_PLAUSIBLE = "not_plausible"
    UNSURE = "unsure"


class MisalignmentRepairScope(StrEnum):
    GLOBAL_OFFSET = "global_offset"
    AFTER_CHANGE_POINT = "after_change_point"


class MisalignmentGlobalCountercheckJudgment(StrEnum):
    NEEDS_TRANSITION = "needs_transition"
    GLOBAL_OFFSET_CONFIRMED = "global_offset_confirmed"
    NOT_REPAIRABLE = "not_repairable"
    UNSURE = "unsure"


class AlignmentReviewCategory(StrEnum):
    LIKELY_ALIGNED = "likely_aligned"
    LIKELY_CONSTANT_OFFSET = "likely_constant_offset"
    NON_CONSTANT_OR_UNCERTAIN = "non_constant_or_uncertain"


class MisalignmentOffsetRecommendation(FrozenBaseModel):
    estimator_version: str
    repair_scope: MisalignmentRepairScope
    shift_seconds: float
    confidence_score: float
    supporting_window_count: int
    summary: str


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
    review_category: AlignmentReviewCategory
    alignment_likelihood_score: float
    review_category_summary: str
    offset_recommendation: MisalignmentOffsetRecommendation | None


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
    predicted_speaker2_waveform: tuple[PreviewWaveformPoint, ...] | None
    speaker1: MisalignmentWindowAnnotation
    speaker2: MisalignmentWindowAnnotation
    predicted_speaker2: MisalignmentWindowAnnotation | None
    repair_estimate: MisalignmentRepairEstimate | None


class MisalignmentRepairEstimate(FrozenBaseModel):
    estimator_version: str
    first_part_shift_seconds: float
    predicted_second_part_shift_seconds: float
    shift_change_seconds: float
    change_interval_start_seconds: float
    change_interval_end_seconds: float
    conservative_first_part_end_seconds: float
    conservative_second_part_start_seconds: float
    stable_second_part_start_seconds: float
    stable_second_part_end_seconds: float
    stable_second_part_duration_seconds: float
    supporting_window_count: int
    shift_spread_seconds: float
    confidence_score: float


class MisalignmentTransitionPreview(FrozenBaseModel):
    sample_id: UUID
    candidate_id: UUID
    external_id: str
    window_start_seconds: float
    window_end_seconds: float
    estimated_change_point_seconds: float
    change_interval_start_seconds: float
    change_interval_end_seconds: float
    search_start_seconds: float
    search_end_seconds: float
    first_part_shift_seconds: float
    second_part_shift_seconds: float
    speaker1_waveform: tuple[PreviewWaveformPoint, ...]
    speaker2_raw_waveform: tuple[PreviewWaveformPoint, ...]
    speaker2_first_alignment_waveform: tuple[PreviewWaveformPoint, ...]
    speaker2_second_alignment_waveform: tuple[PreviewWaveformPoint, ...]
    speaker1: MisalignmentWindowAnnotation
    speaker2_raw: MisalignmentWindowAnnotation
    speaker2_first_alignment: MisalignmentWindowAnnotation
    speaker2_second_alignment: MisalignmentWindowAnnotation


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


class MisalignmentRepairStoredJudgment(FrozenBaseModel):
    sample_id: UUID
    candidate_id: UUID
    external_id: str
    predicted_shift_seconds: float
    estimator_version: str
    repair_scope: MisalignmentRepairScope
    first_part_shift_seconds: float | None
    change_point_seconds: float | None
    change_interval_start_seconds: float | None
    change_interval_end_seconds: float | None
    transition_confirmed: bool
    judgment: MisalignmentRepairJudgment
    created_at: datetime
    updated_at: datetime


class MisalignmentRepairCandidate(FrozenBaseModel):
    candidate: MisalignmentCandidateSummary
    repair_estimate: MisalignmentRepairEstimate
    stored_judgment: MisalignmentRepairStoredJudgment | None


class MisalignmentRepairProgress(FrozenBaseModel):
    quarantined_session_count: int
    repair_candidate_count: int
    reviewed_repair_count: int
    plausible_repair_count: int
    rejected_repair_count: int
    unsure_repair_count: int


class MisalignmentRepairQueueResponse(FrozenBaseModel):
    candidates: tuple[MisalignmentRepairCandidate, ...]
    progress: MisalignmentRepairProgress


class MisalignmentGlobalCountercheckStored(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    beginning_candidate_id: UUID
    ending_candidate_id: UUID
    beginning_start_seconds: float
    beginning_end_seconds: float
    ending_start_seconds: float
    ending_end_seconds: float
    predicted_shift_seconds: float
    estimator_version: str
    judgment: MisalignmentGlobalCountercheckJudgment
    created_at: datetime
    updated_at: datetime


class MisalignmentGlobalCountercheckCandidate(FrozenBaseModel):
    beginning: MisalignmentCandidateSummary
    ending: MisalignmentCandidateSummary
    provisional_repair: MisalignmentRepairStoredJudgment
    stored_judgment: MisalignmentGlobalCountercheckStored | None


class MisalignmentGlobalCountercheckProgress(FrozenBaseModel):
    candidate_count: int
    reviewed_count: int
    needs_transition_count: int
    global_offset_confirmed_count: int
    not_repairable_count: int
    unsure_count: int


class MisalignmentGlobalCountercheckQueueResponse(FrozenBaseModel):
    candidates: tuple[MisalignmentGlobalCountercheckCandidate, ...]
    progress: MisalignmentGlobalCountercheckProgress


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


class MisalignmentRepairJudgmentRequest(FrozenBaseModel):
    sample_id: UUID
    candidate_id: UUID
    predicted_shift_seconds: float
    estimator_version: str = Field(min_length=1, max_length=200)
    repair_scope: MisalignmentRepairScope
    first_part_shift_seconds: float | None
    change_point_seconds: float | None
    change_interval_start_seconds: float | None
    change_interval_end_seconds: float | None
    transition_confirmed: bool
    judgment: MisalignmentRepairJudgment


class MisalignmentRepairJudgmentResponse(FrozenBaseModel):
    stored: MisalignmentRepairStoredJudgment
    progress: MisalignmentRepairProgress


class MisalignmentGlobalCountercheckRequest(FrozenBaseModel):
    sample_id: UUID
    beginning_candidate_id: UUID
    ending_candidate_id: UUID
    predicted_shift_seconds: float
    estimator_version: str = Field(min_length=1, max_length=200)
    judgment: MisalignmentGlobalCountercheckJudgment


class MisalignmentGlobalCountercheckResponse(FrozenBaseModel):
    stored: MisalignmentGlobalCountercheckStored
    progress: MisalignmentGlobalCountercheckProgress
