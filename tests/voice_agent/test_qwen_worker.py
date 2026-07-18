from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Literal

from app.compute.voice.llm_worker_protocol import (
    LlmAssistantMessage,
    LlmEndEvent,
    LlmSpokenTextDeltaEvent,
    LlmToolMessage,
    LlmUserMessage,
    LlmWorkerEvent,
    StartLlmCommand,
)
from app.compute.voice.qwen_worker import (
    LANGUAGE_MODEL_SYSTEM_PROMPT,
    QwenChatMessage,
    QwenRuntime,
    QwenWorkerController,
    qwen_chat_messages,
    render_qwen_prompt,
)
from app.compute.voice.tools import (
    SearchArguments,
    SearchToolCallFunction,
    ToolCall,
    ToolName,
    ToolSuccess,
    create_runtime_tool_registry,
)


class ImmediateQwenRuntime(QwenRuntime):
    def __init__(self) -> None:
        return

    def stream_text(
        self,
        command: StartLlmCommand,
        cancellation_event: threading.Event,
    ) -> Iterator[str]:
        del command, cancellation_event
        yield "ready"

    def count_response_tokens(self, text: str) -> int:
        return len(text)


class RecordingQwenWorkerController(QwenWorkerController):
    def __init__(self) -> None:
        super().__init__(ImmediateQwenRuntime())
        self.events: queue.Queue[LlmWorkerEvent] = queue.Queue()

    def _send_event(self, event: LlmWorkerEvent) -> None:
        self.events.put(event)


class RecordingChatTemplateTokenizer:
    def __init__(self) -> None:
        self.conversation: list[QwenChatMessage] | None = None
        self.tools: list[dict[str, object]] | None = None
        self.tokenize: bool | None = None
        self.add_generation_prompt: bool | None = None
        self.enable_thinking: bool | None = None

    def apply_chat_template(
        self,
        conversation: list[QwenChatMessage],
        *,
        tokenize: Literal[False],
        add_generation_prompt: Literal[True],
        tools: list[dict[str, object]],
        enable_thinking: Literal[False],
    ) -> str:
        self.conversation = conversation
        self.tools = tools
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.enable_thinking = enable_thinking
        return "rendered prompt"


def test_tool_prompt_requires_spoken_bridge_before_call() -> None:
    assert "always begin with exactly one short, natural bridge sentence" in (
        LANGUAGE_MODEL_SYSTEM_PROMPT
    )
    assert "never begin with the tool call" in LANGUAGE_MODEL_SYSTEM_PROMPT
    assert "another provided tool call is genuinely needed" in LANGUAGE_MODEL_SYSTEM_PROMPT
    assert "Let me look that up." in LANGUAGE_MODEL_SYSTEM_PROMPT
    assert (
        '<tool_call>{"name":"search","arguments":{"query":"latest Mars mission"}}</tool_call>'
        in LANGUAGE_MODEL_SYSTEM_PROMPT
    )


def test_qwen_template_preserves_sequential_tool_exchanges_before_later_user_turn() -> None:
    tool_call = ToolCall(
        id="qwen-41-tool-1",
        function=SearchToolCallFunction(
            arguments=SearchArguments(query="current weather in London"),
        ),
    )
    tool_outcome = ToolSuccess(
        call_id=tool_call.id,
        tool_name=ToolName.SEARCH,
        result="London is 12 degrees and lightly cloudy.",
    )
    second_tool_call = ToolCall(
        id="qwen-42-tool-1",
        function=SearchToolCallFunction(
            arguments=SearchArguments(query="current weather in Berlin"),
        ),
    )
    second_tool_outcome = ToolSuccess(
        call_id=second_tool_call.id,
        tool_name=ToolName.SEARCH,
        result="Berlin is 18 degrees and clear.",
    )
    command = StartLlmCommand(
        invocation_id=43,
        assistant_generation_id=12,
        messages=(
            LlmUserMessage(content="What is the weather in London?"),
            LlmAssistantMessage(
                content="Let me check that.",
                tool_calls=(tool_call,),
            ),
            LlmToolMessage(
                tool_call_id=tool_call.id,
                content=tool_outcome,
            ),
            LlmAssistantMessage(
                content="London is cool; I will compare Berlin.",
                tool_calls=(second_tool_call,),
            ),
            LlmToolMessage(
                tool_call_id=second_tool_call.id,
                content=second_tool_outcome,
            ),
            LlmAssistantMessage(content="Berlin is warmer at 18 degrees."),
            LlmUserMessage(content="Should I take a coat?"),
        ),
        tools=create_runtime_tool_registry().specifications,
    )
    tokenizer = RecordingChatTemplateTokenizer()

    assert render_qwen_prompt(tokenizer, command) == "rendered prompt"

    assert tokenizer.conversation == qwen_chat_messages(command)
    assert tokenizer.conversation == [
        {"role": "system", "content": LANGUAGE_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": "What is the weather in London?"},
        {
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [
                {
                    "id": "qwen-41-tool-1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query":"current weather in London"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": tool_outcome.model_dump_json(),
            "tool_call_id": "qwen-41-tool-1",
        },
        {
            "role": "assistant",
            "content": "London is cool; I will compare Berlin.",
            "tool_calls": [
                {
                    "id": "qwen-42-tool-1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query":"current weather in Berlin"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": second_tool_outcome.model_dump_json(),
            "tool_call_id": "qwen-42-tool-1",
        },
        {
            "role": "assistant",
            "content": "Berlin is warmer at 18 degrees.",
        },
        {"role": "user", "content": "Should I take a coat?"},
    ]
    assert tokenizer.tools == [
        specification.model_dump(mode="json")
        for specification in create_runtime_tool_registry().specifications
    ]
    assert tokenizer.tokenize is False
    assert tokenizer.add_generation_prompt is True
    assert tokenizer.enable_thinking is False


def test_terminal_event_means_worker_accepts_next_invocation() -> None:
    controller = RecordingQwenWorkerController()

    controller._start(
        StartLlmCommand(
            invocation_id=1,
            assistant_generation_id=9,
            messages=(),
            tools=(),
        )
    )
    assert controller.events.get(timeout=1) == LlmSpokenTextDeltaEvent(
        invocation_id=1,
        text="ready",
        cumulative_token_count=5,
    )
    assert controller.events.get(timeout=1) == LlmEndEvent(
        invocation_id=1,
        cumulative_token_count=5,
    )

    controller._start(
        StartLlmCommand(
            invocation_id=2,
            assistant_generation_id=9,
            messages=(),
            tools=(),
        )
    )
    assert controller.events.get(timeout=1) == LlmSpokenTextDeltaEvent(
        invocation_id=2,
        text="ready",
        cumulative_token_count=5,
    )
    assert controller.events.get(timeout=1) == LlmEndEvent(
        invocation_id=2,
        cumulative_token_count=5,
    )
