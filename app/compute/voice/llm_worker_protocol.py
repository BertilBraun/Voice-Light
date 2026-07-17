from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.compute.voice.conversation import ConversationRole
from app.shared.base_model import FrozenBaseModel


class LlmWorkerCommandType(StrEnum):
    START = "start"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


class LlmWorkerEventType(StrEnum):
    READY = "ready"
    TEXT_DELTA = "text_delta"
    END = "end"
    CANCELLED = "cancelled"
    ERROR = "error"


class LlmWorkerMessage(FrozenBaseModel):
    role: ConversationRole
    content: str


class StartLlmCommand(FrozenBaseModel):
    type: Literal[LlmWorkerCommandType.START] = LlmWorkerCommandType.START
    generation_id: int = Field(gt=0)
    messages: tuple[LlmWorkerMessage, ...]


class CancelLlmCommand(FrozenBaseModel):
    type: Literal[LlmWorkerCommandType.CANCEL] = LlmWorkerCommandType.CANCEL
    generation_id: int = Field(gt=0)


class ShutdownLlmCommand(FrozenBaseModel):
    type: Literal[LlmWorkerCommandType.SHUTDOWN] = LlmWorkerCommandType.SHUTDOWN


LlmWorkerCommand = Annotated[
    StartLlmCommand | CancelLlmCommand | ShutdownLlmCommand,
    Field(discriminator="type"),
]
llm_worker_command_adapter: TypeAdapter[LlmWorkerCommand] = TypeAdapter(LlmWorkerCommand)


class LlmWorkerReadyEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.READY] = LlmWorkerEventType.READY


class LlmTextDeltaEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.TEXT_DELTA] = LlmWorkerEventType.TEXT_DELTA
    generation_id: int = Field(gt=0)
    text: str
    cumulative_token_count: int = Field(ge=0)


class LlmEndEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.END] = LlmWorkerEventType.END
    generation_id: int = Field(gt=0)


class LlmCancelledEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.CANCELLED] = LlmWorkerEventType.CANCELLED
    generation_id: int = Field(gt=0)


class LlmWorkerErrorEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.ERROR] = LlmWorkerEventType.ERROR
    generation_id: int = Field(gt=0)
    message: str


LlmWorkerEvent = Annotated[
    LlmWorkerReadyEvent | LlmTextDeltaEvent | LlmEndEvent | LlmCancelledEvent | LlmWorkerErrorEvent,
    Field(discriminator="type"),
]
llm_worker_event_adapter: TypeAdapter[LlmWorkerEvent] = TypeAdapter(LlmWorkerEvent)
