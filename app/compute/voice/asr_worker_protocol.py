from __future__ import annotations

from base64 import b64decode, b64encode
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.shared.base_model import FrozenBaseModel


class AsrWorkerCommandType(str, Enum):
    START = "start"
    AUDIO = "audio"
    FINISH = "finish"
    SHUTDOWN = "shutdown"


class AsrWorkerEventType(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    FINAL = "final"
    ERROR = "error"


class StartAsrCommand(FrozenBaseModel):
    type: Literal[AsrWorkerCommandType.START] = AsrWorkerCommandType.START


class AsrAudioCommand(FrozenBaseModel):
    type: Literal[AsrWorkerCommandType.AUDIO] = AsrWorkerCommandType.AUDIO
    pcm_base64: str

    @classmethod
    def from_pcm_bytes(cls, pcm_bytes: bytes) -> AsrAudioCommand:
        return cls(pcm_base64=b64encode(pcm_bytes).decode("ascii"))

    def pcm_bytes(self) -> bytes:
        return b64decode(self.pcm_base64, validate=True)


class FinishAsrCommand(FrozenBaseModel):
    type: Literal[AsrWorkerCommandType.FINISH] = AsrWorkerCommandType.FINISH


class ShutdownAsrCommand(FrozenBaseModel):
    type: Literal[AsrWorkerCommandType.SHUTDOWN] = AsrWorkerCommandType.SHUTDOWN


AsrWorkerCommand = Annotated[
    StartAsrCommand | AsrAudioCommand | FinishAsrCommand | ShutdownAsrCommand,
    Field(discriminator="type"),
]
asr_worker_command_adapter: TypeAdapter[AsrWorkerCommand] = TypeAdapter(AsrWorkerCommand)


class AsrWorkerReadyEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.READY] = AsrWorkerEventType.READY


class PartialAsrEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.PARTIAL] = AsrWorkerEventType.PARTIAL
    text: str


class FinalAsrEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.FINAL] = AsrWorkerEventType.FINAL
    text: str


class AsrWorkerErrorEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.ERROR] = AsrWorkerEventType.ERROR
    message: str


AsrWorkerEvent = Annotated[
    AsrWorkerReadyEvent | PartialAsrEvent | FinalAsrEvent | AsrWorkerErrorEvent,
    Field(discriminator="type"),
]
asr_worker_event_adapter: TypeAdapter[AsrWorkerEvent] = TypeAdapter(AsrWorkerEvent)
