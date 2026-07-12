from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.frozen_base_config import FrozenBaseModel


class ClientEventType(StrEnum):
    SESSION_START = "session.start"
    SESSION_STOP = "session.stop"


class SessionStartEvent(FrozenBaseModel):
    type: Literal[ClientEventType.SESSION_START]
    input_sample_rate: int = Field(gt=0)


class SessionStopEvent(FrozenBaseModel):
    type: Literal[ClientEventType.SESSION_STOP]


ClientEvent = Annotated[SessionStartEvent | SessionStopEvent, Field(discriminator="type")]
client_event_adapter: TypeAdapter[ClientEvent] = TypeAdapter(ClientEvent)


class ServerEventType(StrEnum):
    SESSION_READY = "session.ready"
    VAD_STARTED = "vad.started"
    VAD_STOPPED = "vad.stopped"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    TURN_COMMITTED = "turn.committed"
    ASSISTANT_TEXT_DELTA = "assistant.text.delta"
    ASSISTANT_AUDIO_START = "assistant.audio.start"
    ASSISTANT_AUDIO_END = "assistant.audio.end"
    ASSISTANT_CANCEL = "assistant.cancel"
    ERROR = "error"


class SessionReadyEvent(FrozenBaseModel):
    type: Literal[ServerEventType.SESSION_READY] = ServerEventType.SESSION_READY
    session_id: str
    input_sample_rate: int
    output_sample_rate: int


class SpeechStateEvent(FrozenBaseModel):
    type: Literal[ServerEventType.VAD_STARTED, ServerEventType.VAD_STOPPED]
    audio_time_ms: int = Field(ge=0)


class TranscriptEvent(FrozenBaseModel):
    type: Literal[
        ServerEventType.TRANSCRIPT_PARTIAL,
        ServerEventType.TRANSCRIPT_FINAL,
        ServerEventType.TURN_COMMITTED,
    ]
    text: str


class AssistantTextDeltaEvent(FrozenBaseModel):
    type: Literal[ServerEventType.ASSISTANT_TEXT_DELTA] = ServerEventType.ASSISTANT_TEXT_DELTA
    generation_id: int = Field(ge=0)
    text: str


class AssistantAudioBoundaryEvent(FrozenBaseModel):
    type: Literal[
        ServerEventType.ASSISTANT_AUDIO_START,
        ServerEventType.ASSISTANT_AUDIO_END,
        ServerEventType.ASSISTANT_CANCEL,
    ]
    generation_id: int = Field(ge=0)


class ErrorEvent(FrozenBaseModel):
    type: Literal[ServerEventType.ERROR] = ServerEventType.ERROR
    message: str
