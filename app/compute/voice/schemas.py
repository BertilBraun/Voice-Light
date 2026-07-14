from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.shared.base_model import FrozenBaseModel


class VoiceClientEventType(StrEnum):
    SESSION_START = "session.start"
    SESSION_STOP = "session.stop"
    PLAYBACK_COMPLETE = "playback.complete"
    PLAYBACK_PROGRESS = "playback.progress"


class SessionStartEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.SESSION_START] = VoiceClientEventType.SESSION_START
    input_sample_rate: int = Field(gt=0)


class SessionStopEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.SESSION_STOP] = VoiceClientEventType.SESSION_STOP


class PlaybackCompleteEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_COMPLETE] = VoiceClientEventType.PLAYBACK_COMPLETE
    generation_id: int = Field(gt=0)


class PlaybackProgressEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_PROGRESS] = VoiceClientEventType.PLAYBACK_PROGRESS
    generation_id: int = Field(gt=0)
    sentence_id: int = Field(ge=0)
    text_offset: int = Field(gt=0)


VoiceClientEvent = Annotated[
    SessionStartEvent | SessionStopEvent | PlaybackCompleteEvent | PlaybackProgressEvent,
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
    ASSISTANT_TEXT_DELTA = "assistant.text.delta"
    ASSISTANT_AUDIO_START = "assistant.audio.start"
    ASSISTANT_AUDIO_END = "assistant.audio.end"
    ASSISTANT_AUDIO_SENTENCE = "assistant.audio.sentence"
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


class AssistantAudioSentenceEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ASSISTANT_AUDIO_SENTENCE] = (
        VoiceServerEventType.ASSISTANT_AUDIO_SENTENCE
    )
    generation_id: int = Field(gt=0)
    sentence_id: int = Field(ge=0)
    text_start: int = Field(ge=0)
    text_end: int = Field(gt=0)
    sample_count: int = Field(gt=0)


class ErrorEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ERROR] = VoiceServerEventType.ERROR
    message: str


VoiceServerEvent = Annotated[
    SessionReadyEvent
    | SpeechStateEvent
    | TranscriptEvent
    | AssistantTextDeltaEvent
    | AssistantAudioBoundaryEvent
    | AssistantAudioSentenceEvent
    | ErrorEvent,
    Field(discriminator="type"),
]
