from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import app.compute.voice.models as language_models
from app.compute.voice.conversation import ConversationMessage, ConversationRole
from app.compute.voice.interfaces import LanguageModelTextDelta
from app.compute.voice.llm_worker_protocol import (
    CancelLlmCommand,
    LlmCancelledEvent,
    LlmEndEvent,
    LlmTextDeltaEvent,
    LlmWorkerCommand,
    LlmWorkerCommandType,
    LlmWorkerErrorEvent,
    LlmWorkerEvent,
    StartLlmCommand,
)
from app.compute.voice.models import (
    QwenGenerationSession,
    QwenWorker,
    QwenWorkerLease,
    RestartingQwenWorkerManager,
)

CONVERSATION = (ConversationMessage(role=ConversationRole.USER, content="Hello"),)


class FakeQwenWorker:
    def __init__(
        self,
        start_events: tuple[LlmWorkerEvent, ...] = (),
        cancel_completes: bool = True,
        failing_command_type: LlmWorkerCommandType | None = None,
    ) -> None:
        self.start_events = start_events
        self.cancel_completes = cancel_completes
        self.failing_command_type = failing_command_type
        self.commands: list[LlmWorkerCommand] = []
        self.events: queue.Queue[LlmWorkerEvent] = queue.Queue()

    def send(self, command: LlmWorkerCommand) -> None:
        if command.type is self.failing_command_type:
            raise BrokenPipeError("synthetic Qwen send failure")
        self.commands.append(command)
        match command:
            case StartLlmCommand():
                for event in self.start_events:
                    self.events.put(event)
            case CancelLlmCommand():
                if self.cancel_completes:
                    self.events.put(LlmCancelledEvent(generation_id=command.generation_id))
            case _:
                pass

    def read_event(self) -> LlmWorkerEvent:
        return self.events.get(timeout=1)

    def terminate(self) -> None:
        return


class FakeQwenWorkerManager:
    def __init__(self, worker: FakeQwenWorker) -> None:
        self.worker = worker
        self.lock = asyncio.Lock()
        self.next_generation_id = 1
        self.replacement_count = 0

    async def acquire(self) -> QwenWorkerLease:
        await self.lock.acquire()
        lease = QwenWorkerLease(
            worker=self.worker,
            generation_id=self.next_generation_id,
        )
        self.next_generation_id += 1
        return lease

    def release(self) -> None:
        self.lock.release()

    async def replace(self, failed_worker: QwenWorker) -> None:
        assert failed_worker is self.worker
        self.replacement_count += 1
        failed_worker.terminate()


def test_qwen_worker_exception_is_surfaced_and_worker_is_replaced() -> None:
    async def run_generation() -> None:
        worker = FakeQwenWorker(
            start_events=(LlmWorkerErrorEvent(generation_id=1, message="CUDA failed"),)
        )
        worker_manager = FakeQwenWorkerManager(worker)
        session = create_session(worker_manager)

        with pytest.raises(RuntimeError, match="Qwen worker failed: CUDA failed"):
            await collect_text(session.stream_text())

        assert worker_manager.replacement_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_generation())


def test_qwen_no_progress_timeout_replaces_worker() -> None:
    async def run_generation() -> None:
        worker = FakeQwenWorker()
        worker_manager = FakeQwenWorkerManager(worker)
        session = create_session(worker_manager, progress_timeout_seconds=0.01)

        with pytest.raises(RuntimeError, match="stopped making progress"):
            await collect_text(session.stream_text())

        assert worker_manager.replacement_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_generation())


def test_qwen_cancel_timeout_replaces_worker() -> None:
    async def run_generation() -> None:
        worker = FakeQwenWorker(
            start_events=(
                LlmTextDeltaEvent(
                    generation_id=1,
                    text="Hello",
                    cumulative_token_count=1,
                ),
            ),
            cancel_completes=False,
        )
        worker_manager = FakeQwenWorkerManager(worker)
        session = create_session(worker_manager, cancellation_timeout_seconds=0.01)
        stream = session.stream_text()
        assert await anext(stream) == LanguageModelTextDelta(
            text="Hello",
            cumulative_token_count=1,
        )

        with pytest.raises(RuntimeError, match="cancellation timed out"):
            await session.cancel()
        await stream.aclose()

        assert worker_manager.replacement_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_generation())


def test_qwen_worker_is_reused_after_completed_generation() -> None:
    async def run_generations() -> None:
        worker = FakeQwenWorker(
            start_events=(
                LlmTextDeltaEvent(
                    generation_id=1,
                    text="Hello",
                    cumulative_token_count=1,
                ),
                LlmEndEvent(generation_id=1),
            )
        )
        worker_manager = FakeQwenWorkerManager(worker)
        first_session = create_session(worker_manager)
        assert await collect_text(first_session.stream_text()) == "Hello"
        assert not worker_manager.lock.locked()

        worker.start_events = (
            LlmTextDeltaEvent(
                generation_id=2,
                text="Again",
                cumulative_token_count=1,
            ),
            LlmEndEvent(generation_id=2),
        )
        second_session = create_session(worker_manager)
        assert await collect_text(second_session.stream_text()) == "Again"

        start_commands = [
            command for command in worker.commands if isinstance(command, StartLlmCommand)
        ]
        assert [command.generation_id for command in start_commands] == [1, 2]
        assert worker_manager.replacement_count == 0
        assert not worker_manager.lock.locked()

    asyncio.run(run_generations())


@pytest.mark.parametrize(
    "failing_command_type",
    [LlmWorkerCommandType.START, LlmWorkerCommandType.CANCEL],
)
def test_qwen_send_failure_replaces_worker_and_releases_lock(
    failing_command_type: LlmWorkerCommandType,
) -> None:
    async def run_generation() -> None:
        start_events: tuple[LlmWorkerEvent, ...] = ()
        if failing_command_type is LlmWorkerCommandType.CANCEL:
            start_events = (
                LlmTextDeltaEvent(
                    generation_id=1,
                    text="Hello",
                    cumulative_token_count=1,
                ),
            )
        worker = FakeQwenWorker(
            start_events=start_events,
            failing_command_type=failing_command_type,
        )
        worker_manager = FakeQwenWorkerManager(worker)
        session = create_session(worker_manager)

        with pytest.raises(RuntimeError, match="Failed to"):
            if failing_command_type is LlmWorkerCommandType.START:
                await anext(session.stream_text())
            else:
                stream = session.stream_text()
                assert await anext(stream) == LanguageModelTextDelta(
                    text="Hello",
                    cumulative_token_count=1,
                )
                await session.cancel()

        assert worker_manager.replacement_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_generation())


def test_qwen_spawn_failure_releases_manager_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_count = 0

    def create_worker(python_path: Path) -> FakeQwenWorker:
        nonlocal construction_count
        del python_path
        construction_count += 1
        if construction_count > 1:
            raise RuntimeError("synthetic Qwen spawn failure")
        return FakeQwenWorker()

    monkeypatch.setattr(language_models, "QwenWorkerProcess", create_worker)
    worker_manager = RestartingQwenWorkerManager(Path("python"))
    worker_manager.worker = None

    async def acquire_worker() -> None:
        with pytest.raises(RuntimeError, match="synthetic Qwen spawn failure"):
            await worker_manager.acquire()
        assert not worker_manager.lock.locked()

    asyncio.run(acquire_worker())


def create_session(
    worker_manager: FakeQwenWorkerManager,
    progress_timeout_seconds: float = 1.0,
    cancellation_timeout_seconds: float = 1.0,
) -> QwenGenerationSession:
    return QwenGenerationSession(
        worker_manager=worker_manager,
        conversation=CONVERSATION,
        progress_timeout_seconds=progress_timeout_seconds,
        cancellation_timeout_seconds=cancellation_timeout_seconds,
    )


async def collect_text(stream: AsyncIterator[LanguageModelTextDelta]) -> str:
    return "".join([delta.text async for delta in stream])
