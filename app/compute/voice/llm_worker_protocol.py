from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.compute.voice.tools import (
    SerializedToolCall,
    ToolCall,
    ToolCallFailure,
    ToolExecutionFailure,
    ToolSpecification,
    ToolSuccess,
)
from app.shared.base_model import FrozenBaseModel


class LlmWorkerCommandType(StrEnum):
    START = "start"
    GENERATE_TEXT = "generate_text"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


class LlmWorkerEventType(StrEnum):
    READY = "ready"
    SPOKEN_TEXT_DELTA = "spoken_text_delta"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL = "tool_call"
    TOOL_CALL_FAILURE = "tool_call_failure"
    END = "end"
    CANCELLED = "cancelled"
    ERROR = "error"


class LlmUserMessage(FrozenBaseModel):
    role: Literal["user"] = "user"
    content: str


class LlmAssistantMessage(FrozenBaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class LlmToolMessage(FrozenBaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: ToolSuccess | ToolExecutionFailure


LlmWorkerMessage = Annotated[
    LlmUserMessage | LlmAssistantMessage | LlmToolMessage,
    Field(discriminator="role"),
]


class StartLlmCommand(FrozenBaseModel):
    type: Literal[LlmWorkerCommandType.START] = LlmWorkerCommandType.START
    invocation_id: int = Field(gt=0)
    assistant_generation_id: int = Field(gt=0)
    messages: tuple[LlmWorkerMessage, ...]
    tools: tuple[ToolSpecification, ...]


class GenerateTextLlmCommand(FrozenBaseModel):
    type: Literal[LlmWorkerCommandType.GENERATE_TEXT] = LlmWorkerCommandType.GENERATE_TEXT
    invocation_id: int = Field(gt=0)
    system_prompt: str = Field(min_length=1, max_length=4_000)
    user_prompt: str = Field(min_length=1, max_length=8_000)
    max_new_tokens: int = Field(gt=0, le=256)


class CancelLlmCommand(FrozenBaseModel):
    type: Literal[LlmWorkerCommandType.CANCEL] = LlmWorkerCommandType.CANCEL
    invocation_id: int = Field(gt=0)


class ShutdownLlmCommand(FrozenBaseModel):
    type: Literal[LlmWorkerCommandType.SHUTDOWN] = LlmWorkerCommandType.SHUTDOWN


LlmWorkerCommand = Annotated[
    StartLlmCommand | GenerateTextLlmCommand | CancelLlmCommand | ShutdownLlmCommand,
    Field(discriminator="type"),
]
llm_worker_command_adapter: TypeAdapter[LlmWorkerCommand] = TypeAdapter(LlmWorkerCommand)


class LlmWorkerReadyEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.READY] = LlmWorkerEventType.READY


class LlmSpokenTextDeltaEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.SPOKEN_TEXT_DELTA] = LlmWorkerEventType.SPOKEN_TEXT_DELTA
    invocation_id: int = Field(gt=0)
    text: str
    cumulative_token_count: int = Field(ge=0)


class LlmToolCallStartedEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.TOOL_CALL_STARTED] = LlmWorkerEventType.TOOL_CALL_STARTED
    invocation_id: int = Field(gt=0)
    call_id: str
    cumulative_token_count: int = Field(ge=0)


class LlmToolCallEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.TOOL_CALL] = LlmWorkerEventType.TOOL_CALL
    invocation_id: int = Field(gt=0)
    request: SerializedToolCall
    cumulative_token_count: int = Field(ge=0)


class LlmToolCallFailureEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.TOOL_CALL_FAILURE] = LlmWorkerEventType.TOOL_CALL_FAILURE
    invocation_id: int = Field(gt=0)
    failure: ToolCallFailure
    cumulative_token_count: int = Field(ge=0)


class LlmEndEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.END] = LlmWorkerEventType.END
    invocation_id: int = Field(gt=0)
    cumulative_token_count: int = Field(ge=0)


class LlmCancelledEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.CANCELLED] = LlmWorkerEventType.CANCELLED
    invocation_id: int = Field(gt=0)


class LlmWorkerErrorEvent(FrozenBaseModel):
    type: Literal[LlmWorkerEventType.ERROR] = LlmWorkerEventType.ERROR
    invocation_id: int = Field(gt=0)
    message: str


LlmWorkerEvent = Annotated[
    LlmWorkerReadyEvent
    | LlmSpokenTextDeltaEvent
    | LlmToolCallStartedEvent
    | LlmToolCallEvent
    | LlmToolCallFailureEvent
    | LlmEndEvent
    | LlmCancelledEvent
    | LlmWorkerErrorEvent,
    Field(discriminator="type"),
]
llm_worker_event_adapter: TypeAdapter[LlmWorkerEvent] = TypeAdapter(LlmWorkerEvent)
