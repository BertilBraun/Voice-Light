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

from app.compute.voice.conversation import (
    ModelAssistantMessage,
    ModelMessage,
    ModelToolMessage,
    ModelUserMessage,
)
from app.compute.voice.interfaces import (
    LanguageModelCompleted,
    LanguageModelEvent,
    LanguageModelFailed,
    LanguageModelRequest,
    LanguageModelTextDelta,
    LanguageModelToolCall,
    LanguageModelToolCallFailure,
    LanguageModelToolCallStarted,
    TextGenerationRequest,
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
    LlmWorkerMessage,
    LlmWorkerReadyEvent,
    ShutdownLlmCommand,
    StartLlmCommand,
    llm_worker_event_adapter,
)
from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
    SEARCH_SUMMARIZER_MODEL_NAME,
    SEARCH_SUMMARIZER_MODEL_REVISION,
)
from app.compute.voice.subprocess_start import read_worker_start_event

logger = logging.getLogger(__name__)
QWEN_PYTHON_PATH: Final = Path(sys.executable)
QWEN_GENERATION_PROGRESS_TIMEOUT_SECONDS: Final = 10.0
QWEN_CANCELLATION_TIMEOUT_SECONDS: Final = 2.0
QWEN_WORKER_STOP_TIMEOUT_SECONDS: Final = 5.0
QWEN_WORKER_START_TIMEOUT_SECONDS: Final = 180.0


class QwenWorker(Protocol):
    def send(self, command: LlmWorkerCommand) -> None: ...

    def read_event(self) -> LlmWorkerEvent: ...

    def terminate(self) -> None: ...


@dataclass(frozen=True)
class QwenWorkerLease:
    worker: QwenWorker
    invocation_id: int


@dataclass(frozen=True)
class QwenWorkerConfiguration:
    model_name: str
    model_revision: str
    component_name: str


class QwenWorkerManager(Protocol):
    async def acquire(self) -> QwenWorkerLease: ...

    def release(self) -> None: ...

    async def replace(self, failed_worker: QwenWorker) -> None: ...


class QwenWorkerProcess:
    def __init__(self, python_path: Path, configuration: QwenWorkerConfiguration) -> None:
        self.process = subprocess.Popen(
            [
                python_path.as_posix(),
                "-m",
                "app.compute.voice.qwen_worker",
                "--model",
                configuration.model_name,
                "--revision",
                configuration.model_revision,
            ],
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
            ready_event = read_worker_start_event(
                self.read_event,
                self.terminate,
                QWEN_WORKER_START_TIMEOUT_SECONDS,
                configuration.component_name,
            )
            if not isinstance(ready_event, LlmWorkerReadyEvent):
                raise RuntimeError(
                    f"The {configuration.component_name} worker failed to initialize."
                )
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
    def __init__(
        self,
        python_path: Path,
        configuration: QwenWorkerConfiguration,
        inference_lock: asyncio.Lock | None = None,
    ) -> None:
        self.python_path = python_path
        self.configuration = configuration
        self.worker: QwenWorkerProcess | None = QwenWorkerProcess(python_path, configuration)
        self.lock = inference_lock if inference_lock is not None else asyncio.Lock()
        self.next_invocation_id = 1

    async def acquire(self) -> QwenWorkerLease:
        await self.lock.acquire()
        try:
            if self.worker is None:
                self.worker = await asyncio.to_thread(
                    QwenWorkerProcess,
                    self.python_path,
                    self.configuration,
                )
            lease = QwenWorkerLease(
                worker=self.worker,
                invocation_id=self.next_invocation_id,
            )
            self.next_invocation_id += 1
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
            raise AssertionError("Cannot replace a Qwen worker owned by another invocation.")
        logger.error("replacing failed Qwen worker process")
        self.worker = None
        await asyncio.to_thread(failed_worker.terminate)

    def close(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.close()


class TransformersLanguageModel:
    def __init__(
        self,
        inference_lock: asyncio.Lock,
        python_path: Path = QWEN_PYTHON_PATH,
    ) -> None:
        self.worker_manager = RestartingQwenWorkerManager(
            python_path,
            QwenWorkerConfiguration(
                model_name=LANGUAGE_MODEL_NAME,
                model_revision=LANGUAGE_MODEL_REVISION,
                component_name="Qwen language model",
            ),
            inference_lock,
        )

    async def stream_response(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelEvent]:
        session = QwenInvocationSession(
            worker_manager=self.worker_manager,
            request=request,
            progress_timeout_seconds=QWEN_GENERATION_PROGRESS_TIMEOUT_SECONDS,
            cancellation_timeout_seconds=QWEN_CANCELLATION_TIMEOUT_SECONDS,
        )
        try:
            async for event in session.stream_events():
                yield event
        finally:
            await session.cancel()

    async def generate_text(self, request: TextGenerationRequest) -> str:
        return await _generate_text(self.worker_manager, request)

    def close(self) -> None:
        self.worker_manager.close()


class TransformersTextGenerator:
    def __init__(
        self,
        inference_lock: asyncio.Lock,
        python_path: Path = QWEN_PYTHON_PATH,
    ) -> None:
        self.worker_manager = RestartingQwenWorkerManager(
            python_path,
            QwenWorkerConfiguration(
                model_name=SEARCH_SUMMARIZER_MODEL_NAME,
                model_revision=SEARCH_SUMMARIZER_MODEL_REVISION,
                component_name="Qwen search summarizer",
            ),
            inference_lock,
        )

    async def generate_text(self, request: TextGenerationRequest) -> str:
        return await _generate_text(self.worker_manager, request)

    def close(self) -> None:
        self.worker_manager.close()


async def _generate_text(
    worker_manager: QwenWorkerManager,
    request: TextGenerationRequest,
) -> str:
    session = QwenInvocationSession(
        worker_manager=worker_manager,
        request=request,
        progress_timeout_seconds=QWEN_GENERATION_PROGRESS_TIMEOUT_SECONDS,
        cancellation_timeout_seconds=QWEN_CANCELLATION_TIMEOUT_SECONDS,
    )
    generated_parts: list[str] = []
    try:
        async for event in session.stream_events():
            match event:
                case LanguageModelTextDelta(text=text):
                    generated_parts.append(text)
                case LanguageModelCompleted():
                    pass
                case LanguageModelFailed(message=message):
                    raise RuntimeError(message)
                case (
                    LanguageModelToolCallStarted()
                    | LanguageModelToolCall()
                    | LanguageModelToolCallFailure()
                ):
                    raise RuntimeError("Qwen emitted a tool event during bounded text generation.")
    finally:
        await session.cancel()
    return "".join(generated_parts)


class _InvocationMarker(Enum):
    CANCELLED = auto()


@dataclass(frozen=True)
class _InvocationFailure:
    error: Exception


_InvocationOutput = LanguageModelEvent | _InvocationFailure | _InvocationMarker


class QwenInvocationSession:
    def __init__(
        self,
        worker_manager: QwenWorkerManager,
        request: LanguageModelRequest | TextGenerationRequest,
        progress_timeout_seconds: float,
        cancellation_timeout_seconds: float,
    ) -> None:
        self.worker_manager = worker_manager
        self.request = request
        self.progress_timeout_seconds = progress_timeout_seconds
        self.cancellation_timeout_seconds = cancellation_timeout_seconds
        self.worker: QwenWorker | None = None
        self.invocation_id: int | None = None
        self.output_queue: asyncio.Queue[_InvocationOutput] = asyncio.Queue()
        self.output_task: asyncio.Task[None] | None = None
        self.start_lock = asyncio.Lock()
        self.finalization_lock = asyncio.Lock()
        self.reader_error: Exception | None = None
        self.worker_finalized = False

    async def stream_events(self) -> AsyncIterator[LanguageModelEvent]:
        await self._ensure_started()
        while True:
            try:
                item = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=self.progress_timeout_seconds,
                )
            except TimeoutError:
                failure = RuntimeError("Qwen invocation stopped making progress.")
                await self._fail_worker(failure)
                await self._stop_output_task()
                yield LanguageModelFailed(
                    invocation_id=self._require_invocation_id(),
                    message=str(failure),
                )
                return
            match item:
                case _InvocationMarker.CANCELLED:
                    failure = RuntimeError("Qwen invocation was cancelled unexpectedly.")
                    await self._finalize_worker(replace=False)
                    await self._stop_output_task()
                    yield LanguageModelFailed(
                        invocation_id=self._require_invocation_id(),
                        message=str(failure),
                    )
                    return
                case _InvocationFailure(error=error):
                    await self._fail_worker(error)
                    await self._stop_output_task()
                    yield LanguageModelFailed(
                        invocation_id=self._require_invocation_id(),
                        message=str(error),
                    )
                    return
                case (
                    LanguageModelTextDelta()
                    | LanguageModelToolCallStarted()
                    | LanguageModelToolCall()
                    | LanguageModelToolCallFailure()
                    | LanguageModelCompleted()
                    | LanguageModelFailed()
                ):
                    yield item
                    if isinstance(item, LanguageModelCompleted):
                        await self._finalize_worker(replace=False)
                        await self._stop_output_task()
                        return

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
        invocation_id = self._require_invocation_id()
        try:
            worker.send(CancelLlmCommand(invocation_id=invocation_id))
        except Exception as error:
            await self._fail_worker(error)
            await self._stop_output_task()
            raise RuntimeError("Failed to cancel Qwen invocation.") from error
        try:
            await asyncio.wait_for(
                asyncio.shield(output_task),
                timeout=self.cancellation_timeout_seconds,
            )
        except TimeoutError as error:
            failure = RuntimeError("Qwen invocation cancellation timed out.")
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
                raise RuntimeError("The Qwen invocation session has already ended.")
            lease = await self.worker_manager.acquire()
            self.worker = lease.worker
            self.invocation_id = lease.invocation_id
            try:
                match self.request:
                    case LanguageModelRequest():
                        command: LlmWorkerCommand = StartLlmCommand(
                            invocation_id=lease.invocation_id,
                            assistant_generation_id=self.request.assistant_generation_id,
                            messages=tuple(
                                _worker_message(message) for message in self.request.messages
                            ),
                            tools=self.request.tools,
                        )
                    case TextGenerationRequest():
                        command = GenerateTextLlmCommand(
                            invocation_id=lease.invocation_id,
                            system_prompt=self.request.system_prompt,
                            user_prompt=self.request.user_prompt,
                            max_new_tokens=self.request.max_new_tokens,
                        )
                self.worker.send(command)
            except Exception as error:
                await self._fail_worker(error)
                raise RuntimeError("Failed to start Qwen invocation.") from error
            self.output_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> None:
        try:
            while True:
                event = await asyncio.to_thread(self._require_worker().read_event)
                match event:
                    case LlmWorkerReadyEvent():
                        raise RuntimeError("The Qwen worker sent an unexpected ready event.")
                    case LlmSpokenTextDeltaEvent():
                        self._validate_invocation_id(event.invocation_id)
                        await self.output_queue.put(
                            LanguageModelTextDelta(
                                invocation_id=event.invocation_id,
                                text=event.text,
                                cumulative_token_count=event.cumulative_token_count,
                            )
                        )
                    case LlmToolCallStartedEvent():
                        self._validate_invocation_id(event.invocation_id)
                        await self.output_queue.put(
                            LanguageModelToolCallStarted(
                                invocation_id=event.invocation_id,
                                call_id=event.call_id,
                                cumulative_token_count=event.cumulative_token_count,
                            )
                        )
                    case LlmToolCallEvent():
                        self._validate_invocation_id(event.invocation_id)
                        await self.output_queue.put(
                            LanguageModelToolCall(
                                invocation_id=event.invocation_id,
                                request=event.request,
                                cumulative_token_count=event.cumulative_token_count,
                            )
                        )
                    case LlmToolCallFailureEvent():
                        self._validate_invocation_id(event.invocation_id)
                        await self.output_queue.put(
                            LanguageModelToolCallFailure(
                                invocation_id=event.invocation_id,
                                failure=event.failure,
                                cumulative_token_count=event.cumulative_token_count,
                            )
                        )
                    case LlmEndEvent():
                        self._validate_invocation_id(event.invocation_id)
                        await self.output_queue.put(
                            LanguageModelCompleted(
                                invocation_id=event.invocation_id,
                                cumulative_token_count=event.cumulative_token_count,
                            )
                        )
                        return
                    case LlmCancelledEvent():
                        self._validate_invocation_id(event.invocation_id)
                        await self.output_queue.put(_InvocationMarker.CANCELLED)
                        return
                    case LlmWorkerErrorEvent():
                        self._validate_invocation_id(event.invocation_id)
                        raise RuntimeError(f"Qwen worker failed: {event.message}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.reader_error = error
            await self.output_queue.put(_InvocationFailure(error))

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

    def _validate_invocation_id(self, invocation_id: int) -> None:
        if invocation_id != self._require_invocation_id():
            raise RuntimeError("The Qwen worker returned an event for another invocation.")

    def _require_worker(self) -> QwenWorker:
        if self.worker is None:
            raise AssertionError("The Qwen invocation session does not own a worker.")
        return self.worker

    def _require_invocation_id(self) -> int:
        if self.invocation_id is None:
            raise AssertionError("The Qwen invocation session has no invocation ID.")
        return self.invocation_id


def _worker_message(message: ModelMessage) -> LlmWorkerMessage:
    match message:
        case ModelUserMessage():
            return LlmUserMessage(content=message.content)
        case ModelAssistantMessage():
            return LlmAssistantMessage(
                content=message.content,
                tool_calls=message.tool_calls,
            )
        case ModelToolMessage():
            return LlmToolMessage(
                tool_call_id=message.tool_call_id,
                content=message.outcome,
            )
