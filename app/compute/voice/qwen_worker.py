from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TextIO, TypedDict

from app.compute.voice.hermes_tool_parser import (
    HermesParserEvent,
    HermesSpokenText,
    HermesToolCallCompleted,
    HermesToolCallFailed,
    HermesToolCallParser,
    HermesToolCallStarted,
)
from app.compute.voice.llm_worker_protocol import (
    CancelLlmCommand,
    GenerateTextLlmCommand,
    LlmAssistantMessage,
    LlmCancelledEvent,
    LlmEndEvent,
    LlmSpokenTextDeltaEvent,
    LlmToolCallEvent,
    LlmToolCallFailureEvent,
    LlmToolCallStartedEvent,
    LlmToolMessage,
    LlmUserMessage,
    LlmWorkerCommand,
    LlmWorkerErrorEvent,
    LlmWorkerEvent,
    LlmWorkerReadyEvent,
    ShutdownLlmCommand,
    StartLlmCommand,
    llm_worker_command_adapter,
)
from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class QwenSystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class QwenUserMessage(TypedDict):
    role: Literal["user"]
    content: str


class QwenToolCallFunction(TypedDict):
    name: str
    arguments: str


class QwenToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: QwenToolCallFunction


class QwenAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str
    tool_calls: NotRequired[list[QwenToolCall]]


class QwenToolMessage(TypedDict):
    role: Literal["tool"]
    content: str
    tool_call_id: str


QwenChatMessage = QwenSystemMessage | QwenUserMessage | QwenAssistantMessage | QwenToolMessage
QwenGenerationCommand = StartLlmCommand | GenerateTextLlmCommand


class QwenChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[QwenChatMessage],
        *,
        tokenize: Literal[False],
        add_generation_prompt: Literal[True],
        tools: list[dict[str, object]],
        enable_thinking: Literal[False],
    ) -> str: ...


@dataclass(frozen=True)
class GeneratedTextDelta:
    text: str
    cumulative_token_count: int


class QwenTextRuntime(Protocol):
    def stream_text(
        self,
        command: QwenGenerationCommand,
    ) -> AsyncIterator[GeneratedTextDelta]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ActiveWorkerInvocation:
    invocation_id: int
    task: asyncio.Task[None]


class QwenWorkerController:
    def __init__(
        self,
        runtime: QwenTextRuntime,
        output_stream: TextIO = sys.stdout,
    ) -> None:
        self.runtime = runtime
        self.output_stream = output_stream
        self.active_invocation: ActiveWorkerInvocation | None = None

    async def run(self) -> None:
        self._send_event(LlmWorkerReadyEvent())
        while line := await asyncio.to_thread(sys.stdin.readline):
            try:
                command = llm_worker_command_adapter.validate_json(line)
                if await self._handle_command(command):
                    return
            except Exception:
                logger.exception("Qwen worker command handling failed")
        await self._shutdown()

    async def _handle_command(self, command: LlmWorkerCommand) -> bool:
        match command:
            case StartLlmCommand() | GenerateTextLlmCommand():
                self._start(command)
                return False
            case CancelLlmCommand():
                await self._cancel(command)
                return False
            case ShutdownLlmCommand():
                await self._shutdown()
                return True

    def _start(self, command: QwenGenerationCommand) -> None:
        if self.active_invocation is not None:
            self._send_event(
                LlmWorkerErrorEvent(
                    invocation_id=command.invocation_id,
                    message="Qwen worker already has an active invocation.",
                )
            )
            return
        invocation_task = asyncio.create_task(
            self._generate(command),
            name=f"qwen-stream-{command.invocation_id}",
        )
        self.active_invocation = ActiveWorkerInvocation(
            invocation_id=command.invocation_id,
            task=invocation_task,
        )

    async def _cancel(self, command: CancelLlmCommand) -> None:
        active_invocation = self.active_invocation
        if active_invocation is None or active_invocation.invocation_id != command.invocation_id:
            logger.warning(
                "ignoring Qwen cancellation for inactive invocation: invocation=%d",
                command.invocation_id,
            )
            return
        active_invocation.task.cancel()
        await active_invocation.task

    async def _shutdown(self) -> None:
        active_invocation = self.active_invocation
        if active_invocation is not None:
            active_invocation.task.cancel()
            await active_invocation.task
        self.runtime.close()

    async def _generate(
        self,
        command: QwenGenerationCommand,
    ) -> None:
        terminal_event: LlmWorkerEvent | None = None
        cumulative_token_count = 0
        try:
            match command:
                case StartLlmCommand():
                    parser = HermesToolCallParser(command.invocation_id)
                    async for delta in self.runtime.stream_text(command):
                        cumulative_token_count = delta.cumulative_token_count
                        for parser_event in parser.add_text(delta.text):
                            self._send_parser_event(
                                command.invocation_id,
                                cumulative_token_count,
                                parser_event,
                            )
                    for parser_event in parser.finish():
                        self._send_parser_event(
                            command.invocation_id,
                            cumulative_token_count,
                            parser_event,
                        )
                case GenerateTextLlmCommand():
                    async for delta in self.runtime.stream_text(command):
                        cumulative_token_count = delta.cumulative_token_count
                        self._send_event(
                            LlmSpokenTextDeltaEvent(
                                invocation_id=command.invocation_id,
                                text=delta.text,
                                cumulative_token_count=cumulative_token_count,
                            )
                        )
            terminal_event = LlmEndEvent(
                invocation_id=command.invocation_id,
                cumulative_token_count=cumulative_token_count,
            )
        except asyncio.CancelledError:
            terminal_event = LlmCancelledEvent(invocation_id=command.invocation_id)
        except Exception as error:
            logger.exception("Qwen generation failed: invocation=%d", command.invocation_id)
            terminal_event = LlmWorkerErrorEvent(
                invocation_id=command.invocation_id,
                message=str(error),
            )
        finally:
            active_invocation = self.active_invocation
            if (
                active_invocation is not None
                and active_invocation.invocation_id == command.invocation_id
            ):
                self.active_invocation = None
            if terminal_event is not None:
                self._send_event(terminal_event)

    def _send_parser_event(
        self,
        invocation_id: int,
        cumulative_token_count: int,
        parser_event: HermesParserEvent,
    ) -> None:
        match parser_event:
            case HermesSpokenText():
                self._send_event(
                    LlmSpokenTextDeltaEvent(
                        invocation_id=invocation_id,
                        text=parser_event.text,
                        cumulative_token_count=cumulative_token_count,
                    )
                )
            case HermesToolCallStarted():
                self._send_event(
                    LlmToolCallStartedEvent(
                        invocation_id=invocation_id,
                        call_id=parser_event.call_id,
                        cumulative_token_count=cumulative_token_count,
                    )
                )
            case HermesToolCallCompleted():
                self._send_event(
                    LlmToolCallEvent(
                        invocation_id=invocation_id,
                        request=parser_event.request,
                        cumulative_token_count=cumulative_token_count,
                    )
                )
            case HermesToolCallFailed():
                self._send_event(
                    LlmToolCallFailureEvent(
                        invocation_id=invocation_id,
                        failure=parser_event.failure,
                        cumulative_token_count=cumulative_token_count,
                    )
                )

    def _send_event(self, event: LlmWorkerEvent) -> None:
        self.output_stream.write(event.model_dump_json() + "\n")
        self.output_stream.flush()


def _qwen_chat_message(
    message: LlmUserMessage | LlmAssistantMessage | LlmToolMessage,
) -> QwenChatMessage:
    match message:
        case LlmUserMessage():
            return {"role": "user", "content": message.content}
        case LlmAssistantMessage():
            assistant_message: QwenAssistantMessage = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments.model_dump_json(),
                        },
                    }
                    for call in message.tool_calls
                ]
            return assistant_message
        case LlmToolMessage():
            return {
                "role": "tool",
                "content": message.content.model_dump_json(),
                "tool_call_id": message.tool_call_id,
            }


def qwen_chat_messages(command: StartLlmCommand) -> list[QwenChatMessage]:
    return [
        {"role": "system", "content": LANGUAGE_MODEL_SYSTEM_PROMPT},
        *(_qwen_chat_message(message) for message in command.messages),
    ]


def render_qwen_prompt(
    tokenizer: QwenChatTemplateTokenizer,
    command: QwenGenerationCommand,
) -> str:
    match command:
        case StartLlmCommand():
            messages = qwen_chat_messages(command)
            tools = [specification.model_dump(mode="json") for specification in command.tools]
        case GenerateTextLlmCommand(system_prompt=system_prompt, user_prompt=user_prompt):
            messages = [
                QwenSystemMessage(role="system", content=system_prompt),
                QwenUserMessage(role="user", content=user_prompt),
            ]
            tools = []
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        enable_thinking=False,
    )
