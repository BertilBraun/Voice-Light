from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.compute.voice.conversation import ConversationRole
from app.compute.voice.errors import VoiceComponent, VoiceOperation
from app.shared.base_model import FrozenBaseModel


class CausalSource(StrEnum):
    ENERGY_VAD = "energy_vad"
    SILERO_VAD = "silero_vad"
    NEMOTRON_ASR = "nemotron_asr"
    TURN_ADAPTER = "turn_adapter"
    LEXICAL_CLASSIFIER = "lexical_classifier"
    RESPONSE_RANKER = "response_ranker"
    FLOOR_POLICY = "floor_policy"
    PLAYBACK_ENGINE = "playback_engine"
    USER_COMMAND = "user_command"


class TraceStamp(FrozenBaseModel):
    event_id: str
    parent_event_ids: tuple[str, ...]
    monotonic_time_ns: int = Field(ge=0)
    input_sample_position: int = Field(ge=0)
    output_sample_position: int | None = Field(default=None, ge=0)
    transcript_revision_id: int | None = Field(default=None, ge=0)
    source: CausalSource
    model_name: str | None
    model_revision: str | None


class ActivityHorizon(FrozenBaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    probability: float = Field(ge=0.0, le=1.0)


class PlaybackState(StrEnum):
    IDLE = "idle"
    QUEUED = "queued"
    SPEAKING = "speaking"
    DUCKING = "ducking"
    PAUSED_BUFFERED = "paused_buffered"
    RESUMING = "resuming"
    DRAINING_TO_BOUNDARY = "draining_to_boundary"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class InteractionPrediction(FrozenBaseModel):
    type: Literal["interaction.prediction"] = "interaction.prediction"
    stamp: TraceStamp
    p_user_speech: float = Field(ge=0.0, le=1.0)
    p_user_yield: float = Field(ge=0.0, le=1.0)
    p_user_backchannel: float = Field(ge=0.0, le=1.0)
    p_user_interruption: float = Field(ge=0.0, le=1.0)
    future_user_activity_horizons: tuple[ActivityHorizon, ...]
    assistant_playback_state: PlaybackState
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptRevision(FrozenBaseModel):
    type: Literal["transcript.revision"] = "transcript.revision"
    stamp: TraceStamp
    revision_id: int = Field(ge=0)
    supersedes_revision_id: int | None = Field(default=None, ge=0)
    stable_prefix: str
    volatile_suffix: str
    audio_sample_position: int = Field(ge=0)
    stable_prefix_end_sample: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class VoiceClientEventType(StrEnum):
    SESSION_START = "session.start"
    SESSION_STOP = "session.stop"
    PLAYBACK_STARTED = "playback.started"
    PLAYBACK_COMPLETE = "playback.complete"
    PLAYBACK_PROGRESS = "playback.progress"
    PLAYBACK_STOPPED = "playback.stopped"


class SessionStartEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.SESSION_START] = VoiceClientEventType.SESSION_START
    input_sample_rate: int = Field(gt=0)


class SessionStopEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.SESSION_STOP] = VoiceClientEventType.SESSION_STOP


class PlaybackCompleteEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_COMPLETE] = VoiceClientEventType.PLAYBACK_COMPLETE
    generation_id: int = Field(gt=0)


class PlaybackStartedEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_STARTED] = VoiceClientEventType.PLAYBACK_STARTED
    generation_id: int = Field(gt=0)


class PlaybackProgressEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_PROGRESS] = VoiceClientEventType.PLAYBACK_PROGRESS
    generation_id: int = Field(gt=0)
    text_offset: int = Field(gt=0)
    boundary_start_sample: int = Field(ge=0)
    played_sample_count: int = Field(gt=0)


class PlaybackStoppedEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_STOPPED] = VoiceClientEventType.PLAYBACK_STOPPED
    generation_id: int = Field(gt=0)
    text_offset: int = Field(ge=0)
    played_sample_count: int = Field(ge=0)


VoiceClientEvent = Annotated[
    SessionStartEvent
    | SessionStopEvent
    | PlaybackStartedEvent
    | PlaybackCompleteEvent
    | PlaybackProgressEvent
    | PlaybackStoppedEvent,
    Field(discriminator="type"),
]
voice_client_event_adapter: TypeAdapter[VoiceClientEvent] = TypeAdapter(VoiceClientEvent)


class VoiceServerEventType(StrEnum):
    SESSION_READY = "session.ready"
    VAD_STARTED = "vad.started"
    VAD_STOPPED = "vad.stopped"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    TURN_COMMITTED = "turn.committed"
    LLM_HISTORY = "llm.history"
    ASSISTANT_TEXT_DELTA = "assistant.text.delta"
    ASSISTANT_AUDIO_START = "assistant.audio.start"
    ASSISTANT_AUDIO_END = "assistant.audio.end"
    ASSISTANT_AUDIO_TEXT_BOUNDARY = "assistant.audio.text_boundary"
    ASSISTANT_CANCEL = "assistant.cancel"
    ERROR = "error"


class SessionReadyEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.SESSION_READY] = VoiceServerEventType.SESSION_READY
    session_id: str
    input_sample_rate: int = Field(gt=0)
    output_sample_rate: int = Field(gt=0)


class SpeechStateEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.VAD_STARTED, VoiceServerEventType.VAD_STOPPED]
    audio_time_ms: int = Field(ge=0)


class TranscriptEvent(FrozenBaseModel):
    type: Literal[
        VoiceServerEventType.TRANSCRIPT_PARTIAL,
        VoiceServerEventType.TRANSCRIPT_FINAL,
        VoiceServerEventType.TURN_COMMITTED,
    ]
    text: str


class LlmHistoryMessage(FrozenBaseModel):
    role: ConversationRole
    content: str


class LlmHistoryEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.LLM_HISTORY] = VoiceServerEventType.LLM_HISTORY
    generation_id: int = Field(gt=0)
    messages: tuple[LlmHistoryMessage, ...]


class AssistantTextDeltaEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ASSISTANT_TEXT_DELTA] = (
        VoiceServerEventType.ASSISTANT_TEXT_DELTA
    )
    generation_id: int = Field(gt=0)
    text: str


class AssistantAudioBoundaryEvent(FrozenBaseModel):
    type: Literal[
        VoiceServerEventType.ASSISTANT_AUDIO_START,
        VoiceServerEventType.ASSISTANT_AUDIO_END,
        VoiceServerEventType.ASSISTANT_CANCEL,
    ]
    generation_id: int = Field(gt=0)


class AssistantAudioTextBoundaryEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ASSISTANT_AUDIO_TEXT_BOUNDARY] = (
        VoiceServerEventType.ASSISTANT_AUDIO_TEXT_BOUNDARY
    )
    generation_id: int = Field(gt=0)
    text_offset: int = Field(gt=0)
    start_sample: int = Field(ge=0)


class ErrorEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ERROR] = VoiceServerEventType.ERROR
    component: VoiceComponent
    operation: VoiceOperation
    generation_id: int | None = Field(default=None, gt=0)
    retryable: bool
    message: str


VoiceServerEvent = Annotated[
    SessionReadyEvent
    | SpeechStateEvent
    | TranscriptEvent
    | LlmHistoryEvent
    | AssistantTextDeltaEvent
    | AssistantAudioBoundaryEvent
    | AssistantAudioTextBoundaryEvent
    | ErrorEvent,
    Field(discriminator="type"),
]
