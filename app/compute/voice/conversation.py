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


@dataclass(frozen=True)
class ModelTurnIdentity:
    assistant_generation_id: int
    user_turn_id: int
    transcript_revision_id: int | None

    def __post_init__(self) -> None:
        if self.assistant_generation_id <= 0:
            raise ValueError("The assistant generation ID must be positive.")
        if self.user_turn_id <= 0:
            raise ValueError("The user turn ID must be positive.")
        if self.transcript_revision_id is not None and self.transcript_revision_id <= 0:
            raise ValueError("The transcript revision ID must be positive.")


@dataclass(frozen=True)
class StagedAssistantToolCall:
    identity: ModelTurnIdentity
    invocation_id: int
    message: ModelAssistantMessage

    def __post_init__(self) -> None:
        if self.invocation_id <= 0:
            raise ValueError("The model invocation ID must be positive.")
        if len(self.message.tool_calls) != 1:
            raise ValueError("A staged assistant tool call must contain exactly one call.")


@dataclass(frozen=True)
class CommittedToolExchange:
    identity: ModelTurnIdentity
    invocation_id: int
    assistant_message: ModelAssistantMessage
    tool_message: ModelToolMessage

    def __post_init__(self) -> None:
        if self.invocation_id <= 0:
            raise ValueError("The model invocation ID must be positive.")
        if len(self.assistant_message.tool_calls) != 1:
            raise ValueError("A committed tool exchange must contain exactly one call.")
        call = self.assistant_message.tool_calls[0]
        if call.id != self.tool_message.tool_call_id:
            raise ValueError("The assistant call and tool result must use the same call ID.")
        if call.id != self.tool_message.outcome.call_id:
            raise ValueError("The tool outcome must use the assistant call ID.")


@dataclass
class ModelConversationTurn:
    identity: ModelTurnIdentity
    user_message: ModelUserMessage
    audible_assistant_content: str = ""
    staged_tool_call: StagedAssistantToolCall | None = None
    committed_tool_exchange: CommittedToolExchange | None = None

    def stage_tool_call(
        self,
        invocation_id: int,
        message: ModelAssistantMessage,
    ) -> StagedAssistantToolCall:
        if self.staged_tool_call is not None or self.committed_tool_exchange is not None:
            raise ValueError("The model turn already has a staged or committed tool exchange.")
        stage = StagedAssistantToolCall(
            identity=self.identity,
            invocation_id=invocation_id,
            message=message,
        )
        self.staged_tool_call = stage
        return stage

    def commit_tool_result(
        self,
        outcome: ToolSuccess | ToolExecutionFailure,
    ) -> CommittedToolExchange:
        stage = self.staged_tool_call
        if stage is None:
            raise ValueError("A tool result cannot be committed without a staged assistant call.")
        exchange = CommittedToolExchange(
            identity=self.identity,
            invocation_id=stage.invocation_id,
            assistant_message=stage.message,
            tool_message=ModelToolMessage(
                tool_call_id=outcome.call_id,
                outcome=outcome,
            ),
        )
        self.staged_tool_call = None
        self.committed_tool_exchange = exchange
        return exchange

    def discard_staged_tool_call(self) -> None:
        self.staged_tool_call = None

    def discard_tool_exchange(self) -> None:
        self.staged_tool_call = None
        self.committed_tool_exchange = None

    def update_audible_assistant_content(self, content: str) -> None:
        if not content:
            raise ValueError("Audible assistant content cannot be empty.")
        self.audible_assistant_content = content

    def messages(self) -> tuple[ModelMessage, ...]:
        exchange = self.committed_tool_exchange
        if exchange is None:
            if not self.audible_assistant_content:
                return (self.user_message,)
            return (
                self.user_message,
                ModelAssistantMessage(content=self.audible_assistant_content),
            )
        continuation = _audible_tool_continuation(
            audible_content=self.audible_assistant_content,
            bridge_content=exchange.assistant_message.content,
        )
        if not continuation:
            return (
                self.user_message,
                exchange.assistant_message,
                exchange.tool_message,
            )
        return (
            self.user_message,
            exchange.assistant_message,
            exchange.tool_message,
            ModelAssistantMessage(content=continuation),
        )


class PrivateModelContext:
    def __init__(self) -> None:
        self._turns: list[ModelConversationTurn] = []

    @property
    def turns(self) -> tuple[ModelConversationTurn, ...]:
        return tuple(self._turns)

    def commit_turn(self, turn: ModelConversationTurn) -> None:
        if self._turns and turn.identity.user_turn_id <= self._turns[-1].identity.user_turn_id:
            raise ValueError("Model-context user turn IDs must increase monotonically.")
        self._turns.append(turn)

    def snapshot(self) -> tuple[ModelMessage, ...]:
        return tuple(message for turn in self._turns for message in turn.messages())

    def clear(self) -> None:
        self._turns.clear()


def model_message_from_conversation(message: ConversationMessage) -> ModelMessage:
    match message.role:
        case ConversationRole.USER:
            return ModelUserMessage(content=message.content)
        case ConversationRole.ASSISTANT:
            return ModelAssistantMessage(content=message.content)


def _audible_tool_continuation(
    audible_content: str,
    bridge_content: str,
) -> str:
    if not audible_content.startswith(bridge_content):
        return ""
    return audible_content[len(bridge_content) :].strip()
