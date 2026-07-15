from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Final, Protocol, TextIO

from app.compute.voice.conversation import ConversationMessage
from app.compute.voice.llm_worker_protocol import (
    CancelLlmCommand,
    LlmCancelledEvent,
    LlmEndEvent,
    LlmTextDeltaEvent,
    LlmWorkerCommand,
    LlmWorkerErrorEvent,
    LlmWorkerEvent,
    LlmWorkerMessage,
    LlmWorkerReadyEvent,
    ShutdownLlmCommand,
    StartLlmCommand,
    llm_worker_event_adapter,
)

logger = logging.getLogger(__name__)
QWEN_PYTHON_PATH: Final = Path(sys.executable)
QWEN_GENERATION_PROGRESS_TIMEOUT_SECONDS: Final = 10.0
QWEN_CANCELLATION_TIMEOUT_SECONDS: Final = 2.0
QWEN_WORKER_STOP_TIMEOUT_SECONDS: Final = 5.0


class QwenWorker(Protocol):
    def send(self, command: LlmWorkerCommand) -> None: ...

    def read_event(self) -> LlmWorkerEvent: ...

    def terminate(self) -> None: ...


@dataclass(frozen=True)
class QwenWorkerLease:
    worker: QwenWorker
    generation_id: int


class QwenWorkerManager(Protocol):
    async def acquire(self) -> QwenWorkerLease: ...

    def release(self) -> None: ...

    async def replace(self, failed_worker: QwenWorker) -> None: ...


class QwenWorkerProcess:
    def __init__(self, python_path: Path) -> None:
        self.process = subprocess.Popen(
            [python_path.as_posix(), "-m", "app.compute.voice.qwen_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.input_stream: TextIO = self.process.stdin
        self.output_stream: TextIO = self.process.stdout
        try:
            ready_event = self.read_event()
            if not isinstance(ready_event, LlmWorkerReadyEvent):
                raise RuntimeError("The Qwen worker failed to initialize.")
        except BaseException:
            self.terminate()
            raise

    def send(self, command: LlmWorkerCommand) -> None:
        self.input_stream.write(command.model_dump_json() + "\n")
        self.input_stream.flush()

    def read_event(self) -> LlmWorkerEvent:
        event_json = self.output_stream.readline()
        if not event_json:
            raise RuntimeError("The Qwen worker stopped unexpectedly.")
        return llm_worker_event_adapter.validate_json(event_json)

    def close(self) -> None:
        if self.process.poll() is not None:
            self._close_streams()
            return
        try:
            self.send(ShutdownLlmCommand())
            self.process.wait(timeout=QWEN_WORKER_STOP_TIMEOUT_SECONDS)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.terminate()
            return
        self._close_streams()

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=QWEN_WORKER_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=QWEN_WORKER_STOP_TIMEOUT_SECONDS)
        self._close_streams()

    def _close_streams(self) -> None:
        self.input_stream.close()
        self.output_stream.close()


class RestartingQwenWorkerManager:
    def __init__(self, python_path: Path) -> None:
        self.python_path = python_path
        self.worker: QwenWorkerProcess | None = QwenWorkerProcess(python_path)
        self.lock = asyncio.Lock()
        self.next_generation_id = 1

    async def acquire(self) -> QwenWorkerLease:
        await self.lock.acquire()
        try:
            if self.worker is None:
                self.worker = await asyncio.to_thread(QwenWorkerProcess, self.python_path)
            lease = QwenWorkerLease(
                worker=self.worker,
                generation_id=self.next_generation_id,
            )
            self.next_generation_id += 1
            return lease
        except BaseException:
            self.lock.release()
            raise

    def release(self) -> None:
        if not self.lock.locked():
            raise AssertionError("Cannot release an unowned Qwen worker.")
        self.lock.release()

    async def replace(self, failed_worker: QwenWorker) -> None:
        if not self.lock.locked():
            raise AssertionError("Cannot replace an unowned Qwen worker.")
        if failed_worker is not self.worker:
            raise AssertionError("Cannot replace a Qwen worker owned by another generation.")
        logger.error("replacing failed Qwen worker process")
        self.worker = None
        await asyncio.to_thread(failed_worker.terminate)

    def close(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.close()


class TransformersLanguageModel:
    def __init__(self, python_path: Path = QWEN_PYTHON_PATH) -> None:
        self.worker_manager = RestartingQwenWorkerManager(python_path)

    async def stream_response(
        self,
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]:
        session = QwenGenerationSession(
            worker_manager=self.worker_manager,
            conversation=conversation,
            progress_timeout_seconds=QWEN_GENERATION_PROGRESS_TIMEOUT_SECONDS,
            cancellation_timeout_seconds=QWEN_CANCELLATION_TIMEOUT_SECONDS,
        )
        try:
            async for text_delta in session.stream_text():
                yield text_delta
        finally:
            await session.cancel()

    def close(self) -> None:
        self.worker_manager.close()


class _GenerationMarker(Enum):
    END = auto()
    CANCELLED = auto()


@dataclass(frozen=True)
class _GenerationFailure:
    error: Exception


_GenerationOutput = str | _GenerationFailure | _GenerationMarker


class QwenGenerationSession:
    def __init__(
        self,
        worker_manager: QwenWorkerManager,
        conversation: tuple[ConversationMessage, ...],
        progress_timeout_seconds: float,
        cancellation_timeout_seconds: float,
    ) -> None:
        self.worker_manager = worker_manager
        self.conversation = conversation
        self.progress_timeout_seconds = progress_timeout_seconds
        self.cancellation_timeout_seconds = cancellation_timeout_seconds
        self.worker: QwenWorker | None = None
        self.generation_id: int | None = None
        self.output_queue: asyncio.Queue[_GenerationOutput] = asyncio.Queue()
        self.output_task: asyncio.Task[None] | None = None
        self.start_lock = asyncio.Lock()
        self.finalization_lock = asyncio.Lock()
        self.reader_error: Exception | None = None
        self.worker_finalized = False

    async def stream_text(self) -> AsyncIterator[str]:
        await self._ensure_started()
        while True:
            try:
                item = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=self.progress_timeout_seconds,
                )
            except TimeoutError as error:
                failure = RuntimeError("Qwen generation stopped making progress.")
                await self._fail_worker(failure)
                await self._stop_output_task()
                raise failure from error
            match item:
                case _GenerationMarker.END:
                    await self._finalize_worker(replace=False)
                    await self._stop_output_task()
                    return
                case _GenerationMarker.CANCELLED:
                    failure = RuntimeError("Qwen generation was cancelled unexpectedly.")
                    await self._finalize_worker(replace=False)
                    await self._stop_output_task()
                    raise failure
                case _GenerationFailure(error=error):
                    await self._fail_worker(error)
                    await self._stop_output_task()
                    raise error
                case str():
                    yield item

    async def cancel(self) -> None:
        if self.worker is None or self.worker_finalized:
            return
        output_task = self.output_task
        assert output_task is not None
        if output_task.done():
            if self.reader_error is not None:
                error = self.reader_error
                await self._fail_worker(error)
                raise error
            await self._finalize_worker(replace=False)
            return
        worker = self._require_worker()
        generation_id = self._require_generation_id()
        try:
            worker.send(CancelLlmCommand(generation_id=generation_id))
        except Exception as error:
            await self._fail_worker(error)
            await self._stop_output_task()
            raise RuntimeError("Failed to cancel Qwen generation.") from error
        try:
            await asyncio.wait_for(
                asyncio.shield(output_task),
                timeout=self.cancellation_timeout_seconds,
            )
        except TimeoutError as error:
            failure = RuntimeError("Qwen generation cancellation timed out.")
            await self._fail_worker(failure)
            await self._stop_output_task()
            raise failure from error
        if self.reader_error is not None:
            error = self.reader_error
            await self._fail_worker(error)
            await self._stop_output_task()
            raise error
        await self._finalize_worker(replace=False)
        await self._stop_output_task()

    async def _ensure_started(self) -> None:
        async with self.start_lock:
            if self.worker is not None:
                return
            if self.worker_finalized:
                raise RuntimeError("The Qwen generation session has already ended.")
            lease = await self.worker_manager.acquire()
            self.worker = lease.worker
            self.generation_id = lease.generation_id
            try:
                self.worker.send(
                    StartLlmCommand(
                        generation_id=lease.generation_id,
                        messages=tuple(
                            LlmWorkerMessage(role=message.role, content=message.content)
                            for message in self.conversation
                        ),
                    )
                )
            except Exception as error:
                await self._fail_worker(error)
                raise RuntimeError("Failed to start Qwen generation.") from error
            self.output_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> None:
        try:
            while True:
                event = await asyncio.to_thread(self._require_worker().read_event)
                match event:
                    case LlmWorkerReadyEvent():
                        raise RuntimeError("The Qwen worker sent an unexpected ready event.")
                    case LlmTextDeltaEvent():
                        self._validate_generation_id(event.generation_id)
                        await self.output_queue.put(event.text)
                    case LlmEndEvent():
                        self._validate_generation_id(event.generation_id)
                        await self.output_queue.put(_GenerationMarker.END)
                        return
                    case LlmCancelledEvent():
                        self._validate_generation_id(event.generation_id)
                        await self.output_queue.put(_GenerationMarker.CANCELLED)
                        return
                    case LlmWorkerErrorEvent():
                        self._validate_generation_id(event.generation_id)
                        raise RuntimeError(f"Qwen worker failed: {event.message}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.reader_error = error
            await self.output_queue.put(_GenerationFailure(error))

    async def _fail_worker(self, error: Exception) -> None:
        self.reader_error = error
        await self._finalize_worker(replace=True)

    async def _finalize_worker(self, replace: bool) -> None:
        async with self.finalization_lock:
            if self.worker_finalized:
                return
            worker = self._require_worker()
            try:
                if replace:
                    await self.worker_manager.replace(worker)
            finally:
                self.worker_manager.release()
                self.worker = None
                self.worker_finalized = True

    async def _stop_output_task(self) -> None:
        output_task = self.output_task
        if output_task is None or output_task is asyncio.current_task() or output_task.done():
            return
        output_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await output_task

    def _validate_generation_id(self, generation_id: int) -> None:
        if generation_id != self._require_generation_id():
            raise RuntimeError("The Qwen worker returned an event for another generation.")

    def _require_worker(self) -> QwenWorker:
        if self.worker is None:
            raise AssertionError("The Qwen generation session does not own a worker.")
        return self.worker

    def _require_generation_id(self) -> int:
        if self.generation_id is None:
            raise AssertionError("The Qwen generation session has no generation ID.")
        return self.generation_id
