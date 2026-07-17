from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from app.compute.voice.llm_worker_protocol import (
    LlmEndEvent,
    LlmTextDeltaEvent,
    LlmWorkerEvent,
    StartLlmCommand,
)
from app.compute.voice.qwen_worker import QwenRuntime, QwenWorkerController


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


def test_terminal_event_means_worker_accepts_next_generation() -> None:
    controller = RecordingQwenWorkerController()

    controller._start(StartLlmCommand(generation_id=1, messages=()))
    assert controller.events.get(timeout=1) == LlmTextDeltaEvent(
        generation_id=1,
        text="ready",
        cumulative_token_count=5,
    )
    assert controller.events.get(timeout=1) == LlmEndEvent(generation_id=1)

    controller._start(StartLlmCommand(generation_id=2, messages=()))
    assert controller.events.get(timeout=1) == LlmTextDeltaEvent(
        generation_id=2,
        text="ready",
        cumulative_token_count=5,
    )
    assert controller.events.get(timeout=1) == LlmEndEvent(generation_id=2)
