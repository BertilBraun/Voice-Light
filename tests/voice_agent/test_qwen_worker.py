from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import pytest
import torch

from app.compute.voice.llm_worker_protocol import (
    LlmEndEvent,
    LlmSpokenTextDeltaEvent,
    LlmWorkerEvent,
    StartLlmCommand,
)
from app.compute.voice.qwen_worker import (
    LANGUAGE_MODEL_SYSTEM_PROMPT,
    QwenRuntime,
    QwenWorkerController,
    SpokenBridgeBeforeToolCall,
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


def test_tool_prompt_requires_spoken_bridge_before_call() -> None:
    assert "always begin with exactly one short, natural bridge sentence" in (
        LANGUAGE_MODEL_SYSTEM_PROMPT
    )
    assert "never begin with the tool call" in LANGUAGE_MODEL_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("generated_text", "tool_call_is_blocked"),
    (
        ("", True),
        ("\n", True),
        ("Let me check that", True),
        ("Let me check that.", False),
        ("I will check!", False),
        ("Could I check?", False),
    ),
)
def test_tool_call_opener_waits_for_complete_spoken_bridge(
    generated_text: str,
    tool_call_is_blocked: bool,
) -> None:
    tool_call_token_id = 3
    processor = SpokenBridgeBeforeToolCall(
        prompt_token_count=2,
        tool_call_token_id=tool_call_token_id,
        decode_generated_tokens=lambda token_ids: generated_text if token_ids else "",
    )
    input_ids = torch.tensor([[10, 11, 12]])
    scores = torch.zeros((1, 6))

    processed_scores = processor(input_ids, scores)

    assert torch.isneginf(processed_scores[0, tool_call_token_id]).item() is (tool_call_is_blocked)


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
