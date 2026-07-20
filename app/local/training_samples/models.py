from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from app.local.conversation_regions.models import ConversationRegionAnalysis
from app.local.db.models import TrackSide
from app.shared.audio.gain import TrackGainNormalization
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import ConnectionAnnotationTarget, SegmentAnnotationTarget


class PreviewEventType(StrEnum):
    USER_SPEECH = "user_speech"
    USER_PAUSE = "user_pause"
    USER_END_OF_TURN = "user_end_of_turn"
    USER_BACKCHANNEL = "user_backchannel"
    USER_INTERRUPTION = "user_interruption"
    ASSISTANT_SPEECH = "assistant_speech"
    ASSISTANT_PAUSE = "assistant_pause"
    ASSISTANT_END_OF_TURN = "assistant_end_of_turn"
    ASSISTANT_BACKCHANNEL = "assistant_backchannel"
    ASSISTANT_INTERRUPTION = "assistant_interruption"


class CandidateSource(StrEnum):
    CONNECTION = "connection"
    SEGMENT_END = "segment_end"
    OVERLAP_ONSET = "overlap_onset"
    CENSORED = "censored"


class SupervisionMaskReason(StrEnum):
    BURN_IN = "Burn-in recurrent-state warm-up"
    AMBIGUOUS_ANNOTATION = "Annotation confidence is ambiguous"
    CENSORED_ANNOTATION = "Required future annotation is unavailable"
    NO_EVENT_ANCHOR = "No interaction event is anchored to this frame"
    FUTURE_HORIZON_CENSORED = "Future activity horizon exceeds annotated audio"


class TrainingSampleSelectionMode(StrEnum):
    RANDOM = "random"
    INTERESTING = "interesting"


class EventTargetDistribution(FrozenBaseModel):
    turn_completion: float
    continuation_pause: float
    non_floor_feedback: float
    floor_take: float


class FutureActivityTarget(FrozenBaseModel):
    start_milliseconds: int
    end_milliseconds: int
    occupancy: float | None
    valid: bool
    mask_reason: SupervisionMaskReason | None


class PreviewWaveformPoint(FrozenBaseModel):
    minimum_amplitude: float
    maximum_amplitude: float


class PreviewSpan(FrozenBaseModel):
    event_type: PreviewEventType
    start_seconds: float
    end_seconds: float
    text: str | None


class PreviewPoint(FrozenBaseModel):
    event_type: PreviewEventType
    time_seconds: float
    confidence: float | None
    text: str | None


class TrainingFramePreview(FrozenBaseModel):
    frame_index: int
    time_seconds: float
    relative_time_seconds: float
    supervised: bool
    assistant_speaking_input: bool
    candidate: bool
    candidate_source: CandidateSource | None
    seconds_since_speech_offset: float | None
    user_yield_target: float | None
    user_yield_valid: bool
    user_yield_mask_reason: SupervisionMaskReason | None
    user_has_floor_target: float | None
    user_has_floor_valid: bool
    user_has_floor_mask_reason: SupervisionMaskReason | None
    interaction_event_distribution: EventTargetDistribution | None
    interaction_event_valid: bool
    interaction_event_mask_reason: SupervisionMaskReason | None
    future_activity: tuple[FutureActivityTarget, ...]


class TrainingSampleQuality(FrozenBaseModel):
    total_score: float | None
    interaction_density_score: float | None
    timing_reliability_score: float | None
    audio_quality_score: float | None
    conversation_quality_score: float | None
    usable_event_count: int | None
    events_per_hour: float | None
    flags: tuple[str, ...]


class TrainingSamplePreview(FrozenBaseModel):
    dataset_id: UUID
    sample_id: UUID
    external_id: str
    user_side: TrackSide
    assistant_side: TrackSide
    user_audio_sha256: str
    assistant_audio_sha256: str
    user_gain: TrackGainNormalization
    assistant_gain: TrackGainNormalization
    annotation_version: str
    annotation_generated_at: datetime
    quality_metric_version: str
    quality: TrainingSampleQuality
    represented_duration_seconds: float
    annotated_duration_seconds: float
    eligible_duration_seconds: float
    start_seconds: float
    end_seconds: float
    burn_in_end_seconds: float
    input_duration_seconds: float
    supervised_duration_seconds: float
    frame_seconds: float
    waveform_sample_rate: int
    assistant_waveform_sample_rate: int
    user_waveform: tuple[PreviewWaveformPoint, ...]
    assistant_waveform: tuple[PreviewWaveformPoint, ...]
    user_spans: tuple[PreviewSpan, ...]
    assistant_spans: tuple[PreviewSpan, ...]
    user_points: tuple[PreviewPoint, ...]
    assistant_points: tuple[PreviewPoint, ...]
    user_segment_targets: tuple[SegmentAnnotationTarget, ...]
    assistant_segment_targets: tuple[SegmentAnnotationTarget, ...]
    user_connection_targets: tuple[ConnectionAnnotationTarget, ...]
    assistant_connection_targets: tuple[ConnectionAnnotationTarget, ...]
    recording_user_spans: tuple[PreviewSpan, ...]
    recording_assistant_spans: tuple[PreviewSpan, ...]
    recording_user_points: tuple[PreviewPoint, ...]
    recording_assistant_points: tuple[PreviewPoint, ...]
    conversation_regions: ConversationRegionAnalysis | None
    frames: tuple[TrainingFramePreview, ...]


class TrainingSampleOption(FrozenBaseModel):
    dataset_id: UUID
    sample_id: UUID
    external_id: str
    represented_duration_seconds: float
    usable_event_count: int
    quality_score: float | None
