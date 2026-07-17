from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.compute.voice.tools import ToolCall, ToolExecutionFailure, ToolSuccess


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ConversationMessage:
    role: ConversationRole
    content: str


@dataclass(frozen=True)
class ModelUserMessage:
    content: str


@dataclass(frozen=True)
class ModelAssistantMessage:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ModelToolMessage:
    tool_call_id: str
    outcome: ToolSuccess | ToolExecutionFailure


ModelMessage = ModelUserMessage | ModelAssistantMessage | ModelToolMessage


def model_message_from_conversation(message: ConversationMessage) -> ModelMessage:
    match message.role:
        case ConversationRole.USER:
            return ModelUserMessage(content=message.content)
        case ConversationRole.ASSISTANT:
            return ModelAssistantMessage(content=message.content)
