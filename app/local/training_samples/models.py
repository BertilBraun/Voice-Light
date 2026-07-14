from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from app.local.db.models import TrackSide
from app.shared.base_model import FrozenBaseModel


class AgentActionLabel(StrEnum):
    LISTEN = "listen"
    SPEAK = "speak"


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
    agent_action: AgentActionLabel
    assistant_playback_active: bool
    user_turn_active: bool
    user_speech_active: bool
    assistant_turn_active: bool
    assistant_speech_active: bool
    user_pause: bool
    user_end_of_turn: bool
    user_end_within_0_5_seconds: bool
    user_end_within_1_second: bool
    user_end_within_2_seconds: bool
    user_backchannel: bool
    user_interruption: bool
    assistant_backchannel: bool


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
