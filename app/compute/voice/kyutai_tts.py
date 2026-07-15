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

from app.compute.voice.interfaces import (
    SpeechSynthesisSession,
    SynthesisEvent,
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
)
from app.compute.voice.subprocess_start import read_worker_start_event
from app.compute.voice.tts_worker_protocol import (
    CancelTtsCommand,
    FinishTtsCommand,
    ShutdownTtsCommand,
    StartTtsCommand,
    TtsAudioEvent,
    TtsEndEvent,
    TtsWordBoundaryEvent,
    TtsWordCommand,
    TtsWordProcessedEvent,
    TtsWorkerCommand,
    TtsWorkerErrorEvent,
    TtsWorkerEvent,
    TtsWorkerReadyEvent,
    tts_worker_event_adapter,
)

logger = logging.getLogger(__name__)
KYUTAI_TTS_PYTHON_PATH: Final = Path(sys.executable)
KYUTAI_TTS_GENERATION_PROGRESS_TIMEOUT_SECONDS: Final = 5.0
KYUTAI_TTS_CANCEL_TIMEOUT_SECONDS: Final = 2.0
KYUTAI_TTS_WORKER_STOP_TIMEOUT_SECONDS: Final = 5.0
KYUTAI_TTS_WORKER_START_TIMEOUT_SECONDS: Final = 180.0
KYUTAI_TTS_WATCHDOG_POLL_SECONDS: Final = 0.1


class KyutaiTtsWorker(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def send(self, command: TtsWorkerCommand) -> None: ...

    def read_event(self) -> TtsWorkerEvent: ...

    def terminate(self) -> None: ...


class KyutaiTtsWorkerManager(Protocol):
    @property
    def sample_rate(self) -> int: ...

    async def acquire(self) -> KyutaiTtsWorker: ...

    def release(self) -> None: ...

    async def replace(self, failed_worker: KyutaiTtsWorker) -> None: ...


class KyutaiTtsWorkerProcess:
    def __init__(self, python_path: Path) -> None:
        self.process = subprocess.Popen(
            [python_path.as_posix(), "-m", "app.compute.voice.kyutai_tts_worker"],
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
        ready_event = read_worker_start_event(
            self.read_event,
            self.terminate,
            KYUTAI_TTS_WORKER_START_TIMEOUT_SECONDS,
            "Kyutai TTS",
        )
        if not isinstance(ready_event, TtsWorkerReadyEvent):
            self.terminate()
            raise RuntimeError("The Kyutai TTS worker failed to initialize.")
        self._sample_rate = ready_event.sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def send(self, command: TtsWorkerCommand) -> None:
        self.input_stream.write(command.model_dump_json() + "\n")
        self.input_stream.flush()

    def read_event(self) -> TtsWorkerEvent:
        event_json = self.output_stream.readline()
        if not event_json:
            raise RuntimeError("The Kyutai TTS worker stopped unexpectedly.")
        return tts_worker_event_adapter.validate_json(event_json)

    def close(self) -> None:
        if self.process.poll() is not None:
            self._close_streams()
            return
        try:
            self.send(ShutdownTtsCommand())
            self.process.wait(timeout=KYUTAI_TTS_WORKER_STOP_TIMEOUT_SECONDS)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.terminate()
            return
        self._close_streams()

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=KYUTAI_TTS_WORKER_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=KYUTAI_TTS_WORKER_STOP_TIMEOUT_SECONDS)
        self._close_streams()

    def _close_streams(self) -> None:
        self.input_stream.close()
        self.output_stream.close()


class RestartingKyutaiTtsWorkerManager:
    def __init__(self, python_path: Path) -> None:
        self.python_path = python_path
        self.worker: KyutaiTtsWorkerProcess | None = KyutaiTtsWorkerProcess(python_path)
        self._sample_rate = self.worker.sample_rate
        self.lock = asyncio.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def acquire(self) -> KyutaiTtsWorker:
        await self.lock.acquire()
        try:
            if self.worker is None:
                self.worker = await asyncio.to_thread(KyutaiTtsWorkerProcess, self.python_path)
                if self.worker.sample_rate != self._sample_rate:
                    self.worker.terminate()
                    self.worker = None
                    raise RuntimeError("The replacement Kyutai TTS worker changed sample rate.")
            return self.worker
        except BaseException:
            self.lock.release()
            raise

    def release(self) -> None:
        if not self.lock.locked():
            raise AssertionError("Cannot release an unowned Kyutai TTS worker.")
        self.lock.release()

    async def replace(self, failed_worker: KyutaiTtsWorker) -> None:
        if not self.lock.locked():
            raise AssertionError("Cannot replace an unowned Kyutai TTS worker.")
        if failed_worker is not self.worker:
            raise AssertionError("Cannot replace a Kyutai TTS worker owned by another session.")
        logger.error("replacing unresponsive Kyutai TTS worker process")
        self.worker = None
        await asyncio.to_thread(failed_worker.terminate)

    def close(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.close()


class KyutaiSpeechSynthesizer:
    def __init__(self, python_path: Path = KYUTAI_TTS_PYTHON_PATH) -> None:
        self.worker_manager = RestartingKyutaiTtsWorkerManager(python_path)

    @property
    def sample_rate(self) -> int:
        return self.worker_manager.sample_rate

    def start_session(self) -> SpeechSynthesisSession:
        return KyutaiSpeechSynthesisSession(
            worker_manager=self.worker_manager,
            generation_progress_timeout_seconds=KYUTAI_TTS_GENERATION_PROGRESS_TIMEOUT_SECONDS,
            cancel_timeout_seconds=KYUTAI_TTS_CANCEL_TIMEOUT_SECONDS,
        )

    def close(self) -> None:
        self.worker_manager.close()


class _SessionMarker(Enum):
    END = auto()


@dataclass(frozen=True)
class _SessionFailure:
    error: Exception


_SessionOutput = SynthesisEvent | _SessionFailure | _SessionMarker


class KyutaiSpeechSynthesisSession:
    def __init__(
        self,
        worker_manager: KyutaiTtsWorkerManager,
        generation_progress_timeout_seconds: float,
        cancel_timeout_seconds: float,
    ) -> None:
        self.worker_manager = worker_manager
        self.generation_progress_timeout_seconds = generation_progress_timeout_seconds
        self.cancel_timeout_seconds = cancel_timeout_seconds
        self.worker: KyutaiTtsWorker | None = None
        self.output_queue: asyncio.Queue[_SessionOutput] = asyncio.Queue()
        self.start_lock = asyncio.Lock()
        self.finalization_lock = asyncio.Lock()
        self.output_task: asyncio.Task[None] | None = None
        self.watchdog_task: asyncio.Task[None] | None = None
        self.reader_error: Exception | None = None
        self.pending_word_sequences: set[int] = set()
        self.next_word_sequence = 0
        self.last_progress_time = 0.0
        self.finish_pending = False
        self.input_finished = False
        self.worker_finalized = False

    async def add_word(self, word: SynthesisWord) -> None:
        if self.input_finished:
            raise ValueError("Cannot add a word after synthesis input has finished.")
        if not word.text or any(character.isspace() for character in word.text):
            raise ValueError("Synthesis words must be non-empty and contain no whitespace.")
        await self._ensure_started()
        sequence_number = self.next_word_sequence
        self.next_word_sequence += 1
        self.pending_word_sequences.add(sequence_number)
        self._record_progress()
        try:
            self._require_worker().send(
                TtsWordCommand(
                    sequence_number=sequence_number,
                    text=word.text,
                    text_start=word.text_start,
                    text_end=word.text_end,
                )
            )
        except Exception as error:
            await self._fail_worker(error)
            raise RuntimeError("Failed to send text to the Kyutai TTS worker.") from error

    async def finish_input(self) -> None:
        if self.input_finished:
            raise ValueError("Synthesis input may only be finished once.")
        self.input_finished = True
        await self._ensure_started()
        self.finish_pending = True
        self._record_progress()
        try:
            self._require_worker().send(FinishTtsCommand())
        except Exception as error:
            await self._fail_worker(error)
            raise RuntimeError("Failed to finish Kyutai TTS input.") from error

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        await self._ensure_started()
        while True:
            item = await self.output_queue.get()
            match item:
                case _SessionMarker.END:
                    await self._finalize_worker(replace=self.reader_error is not None)
                    await self._stop_background_tasks()
                    return
                case _SessionFailure(error=error):
                    await self._fail_worker(error)
                    await self._stop_background_tasks()
                    raise error
                case SynthesizedAudioChunk() | SynthesizedWordBoundary():
                    yield item
                case _:
                    raise AssertionError(f"Unexpected synthesis output item: {item!r}")

    async def cancel(self) -> None:
        if self.worker is None or self.worker_finalized:
            return
        self.finish_pending = True
        self._record_progress()
        try:
            self._require_worker().send(CancelTtsCommand())
        except Exception as error:
            await self._fail_worker(error)
            await self._stop_background_tasks()
            raise RuntimeError("Failed to cancel Kyutai TTS generation.") from error
        assert self.output_task is not None
        try:
            await asyncio.wait_for(
                asyncio.shield(self.output_task),
                timeout=self.cancel_timeout_seconds,
            )
        except TimeoutError as timeout_error:
            error = RuntimeError("Kyutai TTS cancellation timed out.")
            await self._fail_worker(error)
            await self._stop_background_tasks()
            raise error from timeout_error
        else:
            await self._finalize_worker(replace=self.reader_error is not None)
        await self._stop_background_tasks()

    async def _ensure_started(self) -> None:
        async with self.start_lock:
            if self.worker is not None:
                return
            if self.worker_finalized:
                raise RuntimeError("The Kyutai TTS session has already ended.")
            self.worker = await self.worker_manager.acquire()
            self._record_progress()
            try:
                self.worker.send(StartTtsCommand())
            except Exception as error:
                await self._fail_worker(error)
                raise RuntimeError("Failed to start the Kyutai TTS worker session.") from error
            self.output_task = asyncio.create_task(self._read_output())
            self.watchdog_task = asyncio.create_task(self._watch_progress())

    async def _read_output(self) -> None:
        try:
            while True:
                event = await asyncio.to_thread(self._require_worker().read_event)
                self._record_progress()
                match event:
                    case TtsWorkerReadyEvent():
                        raise RuntimeError("The Kyutai TTS worker sent an unexpected ready event.")
                    case TtsAudioEvent():
                        await self.output_queue.put(
                            SynthesizedAudioChunk(
                                pcm_bytes=event.pcm_bytes(),
                                start_sample=event.start_sample,
                            )
                        )
                    case TtsWordBoundaryEvent():
                        await self.output_queue.put(
                            SynthesizedWordBoundary(
                                text_offset=event.text_offset,
                                start_sample=event.start_sample,
                            )
                        )
                    case TtsWordProcessedEvent():
                        if event.sequence_number not in self.pending_word_sequences:
                            raise RuntimeError(
                                "The Kyutai TTS worker acknowledged an unknown word."
                            )
                        self.pending_word_sequences.remove(event.sequence_number)
                    case TtsEndEvent():
                        self.finish_pending = False
                        await self.output_queue.put(_SessionMarker.END)
                        return
                    case TtsWorkerErrorEvent():
                        raise RuntimeError(f"Kyutai TTS worker failed: {event.message}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.reader_error = error
            await self.output_queue.put(_SessionFailure(error))
            await self.output_queue.put(_SessionMarker.END)

    async def _watch_progress(self) -> None:
        while True:
            await asyncio.sleep(KYUTAI_TTS_WATCHDOG_POLL_SECONDS)
            if not self.pending_word_sequences and not self.finish_pending:
                continue
            elapsed_seconds = asyncio.get_running_loop().time() - self.last_progress_time
            if elapsed_seconds < self.generation_progress_timeout_seconds:
                continue
            error = RuntimeError("Kyutai TTS generation stopped making progress.")
            self.reader_error = error
            await self.output_queue.put(_SessionFailure(error))
            await self.output_queue.put(_SessionMarker.END)
            await self._finalize_worker(replace=True)
            return

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

    async def _stop_background_tasks(self) -> None:
        current_task = asyncio.current_task()
        tasks = tuple(
            task
            for task in (self.output_task, self.watchdog_task)
            if task is not None and task is not current_task and not task.done()
        )
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _record_progress(self) -> None:
        self.last_progress_time = asyncio.get_running_loop().time()

    def _require_worker(self) -> KyutaiTtsWorker:
        if self.worker is None:
            raise AssertionError("The Kyutai TTS session does not own a worker.")
        return self.worker
