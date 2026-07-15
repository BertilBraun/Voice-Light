from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from app.local.db.models import TrackSide
from app.shared.base_model import FrozenBaseModel


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
    CENSORED = "censored"


class ReliabilitySource(StrEnum):
    ANNOTATED = "annotated"
    UNMEASURED = "unmeasured"


class EventTargetDistribution(FrozenBaseModel):
    turn_completion: float
    continuation_pause: float
    backchannel: float
    interruption: float
    other: float


class FutureActivityTarget(FrozenBaseModel):
    start_milliseconds: int
    end_milliseconds: int
    active: bool | None
    valid: bool


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
    yield_probability: float | None
    hold_probability: float | None
    primary_reliability: float | None
    primary_reliability_source: ReliabilitySource | None
    primary_valid: bool
    event_distribution: EventTargetDistribution | None
    event_reliability: float | None
    event_reliability_source: ReliabilitySource | None
    event_valid: bool
    future_activity: tuple[FutureActivityTarget, ...]


class TrainingSamplePreview(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    user_side: TrackSide
    assistant_side: TrackSide
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
    user_waveform: tuple[PreviewWaveformPoint, ...]
    user_spans: tuple[PreviewSpan, ...]
    assistant_spans: tuple[PreviewSpan, ...]
    user_points: tuple[PreviewPoint, ...]
    assistant_points: tuple[PreviewPoint, ...]
    frames: tuple[TrainingFramePreview, ...]


class TrainingSampleOption(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    represented_duration_seconds: float
    usable_event_count: int
