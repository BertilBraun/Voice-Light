from __future__ import annotations

from base64 import b64decode, b64encode
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.shared.base_model import FrozenBaseModel


class TtsWorkerCommandType(str, Enum):
    START = "start"
    WORD = "word"
    FINISH = "finish"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


class TtsWorkerEventType(str, Enum):
    READY = "ready"
    AUDIO = "audio"
    FIRST_AUDIO_METRICS = "first_audio_metrics"
    WORD_BOUNDARY = "word_boundary"
    WORD_PROCESSED = "word_processed"
    END = "end"
    ERROR = "error"


class StartTtsCommand(FrozenBaseModel):
    type: Literal[TtsWorkerCommandType.START] = TtsWorkerCommandType.START


class TtsWordCommand(FrozenBaseModel):
    type: Literal[TtsWorkerCommandType.WORD] = TtsWorkerCommandType.WORD
    sequence_number: int
    text: str
    text_start: int
    text_end: int


class FinishTtsCommand(FrozenBaseModel):
    type: Literal[TtsWorkerCommandType.FINISH] = TtsWorkerCommandType.FINISH


class CancelTtsCommand(FrozenBaseModel):
    type: Literal[TtsWorkerCommandType.CANCEL] = TtsWorkerCommandType.CANCEL


class ShutdownTtsCommand(FrozenBaseModel):
    type: Literal[TtsWorkerCommandType.SHUTDOWN] = TtsWorkerCommandType.SHUTDOWN


TtsWorkerCommand = Annotated[
    StartTtsCommand | TtsWordCommand | FinishTtsCommand | CancelTtsCommand | ShutdownTtsCommand,
    Field(discriminator="type"),
]
tts_worker_command_adapter: TypeAdapter[TtsWorkerCommand] = TypeAdapter(TtsWorkerCommand)


class TtsWorkerReadyEvent(FrozenBaseModel):
    type: Literal[TtsWorkerEventType.READY] = TtsWorkerEventType.READY
    sample_rate: int


class TtsAudioEvent(FrozenBaseModel):
    type: Literal[TtsWorkerEventType.AUDIO] = TtsWorkerEventType.AUDIO
    pcm_base64: str
    start_sample: int

    @classmethod
    def from_pcm_bytes(cls, pcm_bytes: bytes, start_sample: int) -> TtsAudioEvent:
        return cls(
            pcm_base64=b64encode(pcm_bytes).decode("ascii"),
            start_sample=start_sample,
        )

    def pcm_bytes(self) -> bytes:
        return b64decode(self.pcm_base64, validate=True)


class TtsWordBoundaryEvent(FrozenBaseModel):
    type: Literal[TtsWorkerEventType.WORD_BOUNDARY] = TtsWorkerEventType.WORD_BOUNDARY
    text_offset: int
    start_sample: int


class TtsFirstAudioMetricsEvent(FrozenBaseModel):
    type: Literal[TtsWorkerEventType.FIRST_AUDIO_METRICS] = TtsWorkerEventType.FIRST_AUDIO_METRICS
    first_word_to_audio_seconds: float = Field(ge=0)
    tokenization_seconds: float = Field(ge=0)
    language_model_step_seconds: float = Field(ge=0)
    mimi_decode_seconds: float = Field(ge=0)
    model_step_count: int = Field(gt=0)
    first_audio_model_step: int = Field(gt=0)


class TtsWordProcessedEvent(FrozenBaseModel):
    type: Literal[TtsWorkerEventType.WORD_PROCESSED] = TtsWorkerEventType.WORD_PROCESSED
    sequence_number: int


class TtsEndEvent(FrozenBaseModel):
    type: Literal[TtsWorkerEventType.END] = TtsWorkerEventType.END
    cancelled: bool


class TtsWorkerErrorEvent(FrozenBaseModel):
    type: Literal[TtsWorkerEventType.ERROR] = TtsWorkerEventType.ERROR
    message: str


TtsWorkerEvent = Annotated[
    TtsWorkerReadyEvent
    | TtsAudioEvent
    | TtsFirstAudioMetricsEvent
    | TtsWordBoundaryEvent
    | TtsWordProcessedEvent
    | TtsEndEvent
    | TtsWorkerErrorEvent,
    Field(discriminator="type"),
]
tts_worker_event_adapter: TypeAdapter[TtsWorkerEvent] = TypeAdapter(TtsWorkerEvent)
