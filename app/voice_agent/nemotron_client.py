from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from pathlib import Path
from typing import Final, Protocol, TextIO

from app.voice_agent.asr_worker_protocol import (
    AsrAudioCommand,
    AsrWorkerCommand,
    AsrWorkerErrorEvent,
    AsrWorkerEventType,
    AsrWorkerReadyEvent,
    FinalAsrEvent,
    FinishAsrCommand,
    PartialAsrEvent,
    ShutdownAsrCommand,
    StartAsrCommand,
    asr_worker_event_adapter,
)
from app.voice_agent.interfaces import TranscriptionSession

NEMOTRON_PYTHON_PATH: Final = Path(sys.executable)
STREAMING_CHUNK_DURATION_MS: Final = 80
INPUT_SAMPLE_RATE: Final = 16_000
PCM_BYTES_PER_SAMPLE: Final = 2
STREAMING_CHUNK_BYTE_COUNT: Final = (
    INPUT_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE * STREAMING_CHUNK_DURATION_MS // 1_000
)


class NemotronWorker(Protocol):
    def send(self, command: AsrWorkerCommand) -> None: ...

    def read_event(
        self,
    ) -> AsrWorkerReadyEvent | PartialAsrEvent | FinalAsrEvent | AsrWorkerErrorEvent: ...


class NemotronWorkerProcess:
    def __init__(self, python_path: Path) -> None:
        self.process = subprocess.Popen(
            [python_path.as_posix(), "-m", "app.voice_agent.nemotron_worker"],
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
        ready_event = self.read_event()
        if not isinstance(ready_event, AsrWorkerReadyEvent):
            raise RuntimeError("The Nemotron worker failed to initialize.")

    def send(self, command: AsrWorkerCommand) -> None:
        self.input_stream.write(command.model_dump_json() + "\n")
        self.input_stream.flush()

    def read_event(
        self,
    ) -> AsrWorkerReadyEvent | PartialAsrEvent | FinalAsrEvent | AsrWorkerErrorEvent:
        event_json = self.output_stream.readline()
        if not event_json:
            raise RuntimeError("The Nemotron worker stopped unexpectedly.")
        return asr_worker_event_adapter.validate_json(event_json)

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.send(ShutdownAsrCommand())
        self.process.wait(timeout=10)


class NemotronStreamingTranscriber:
    def __init__(self, python_path: Path = NEMOTRON_PYTHON_PATH) -> None:
        self.worker = NemotronWorkerProcess(python_path)
        self.worker_lock = asyncio.Lock()

    def start_session(self) -> TranscriptionSession:
        return NemotronStreamingSession(worker=self.worker, worker_lock=self.worker_lock)

    def close(self) -> None:
        self.worker.close()


class NemotronStreamingSession:
    def __init__(
        self,
        worker: NemotronWorker,
        worker_lock: asyncio.Lock,
    ) -> None:
        self.worker = worker
        self.worker_lock = worker_lock
        self.pending_audio = bytearray()
        self.partial_text_queue: asyncio.Queue[str] = asyncio.Queue()
        self.output_task: asyncio.Task[str] | None = None
        self.owns_worker = False
        self.finished = False

    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        if self.finished:
            raise RuntimeError("Cannot add audio to a finished Nemotron session.")
        await self._ensure_started()
        self.pending_audio.extend(pcm_bytes)
        while len(self.pending_audio) >= STREAMING_CHUNK_BYTE_COUNT:
            chunk = bytes(self.pending_audio[:STREAMING_CHUNK_BYTE_COUNT])
            del self.pending_audio[:STREAMING_CHUNK_BYTE_COUNT]
            self.worker.send(AsrAudioCommand.from_pcm_bytes(chunk))
        latest_partial_text: str | None = None
        while not self.partial_text_queue.empty():
            latest_partial_text = self.partial_text_queue.get_nowait()
        return latest_partial_text

    async def finish(self) -> str:
        if self.finished:
            raise RuntimeError("Cannot finish a Nemotron session twice.")
        self.finished = True
        if self.output_task is None:
            return ""
        if self.pending_audio:
            self.worker.send(AsrAudioCommand.from_pcm_bytes(bytes(self.pending_audio)))
            self.pending_audio.clear()
        self.worker.send(FinishAsrCommand())
        try:
            return await self.output_task
        finally:
            self._release_worker()

    async def close(self) -> None:
        self.pending_audio.clear()
        if self.output_task is None or self.finished:
            return
        self.finished = True
        self.worker.send(FinishAsrCommand())
        with contextlib.suppress(RuntimeError):
            await self.output_task
        self._release_worker()

    async def _ensure_started(self) -> None:
        if self.output_task is not None:
            return
        await self.worker_lock.acquire()
        self.owns_worker = True
        self.worker.send(StartAsrCommand())
        self.output_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> str:
        while True:
            event = await asyncio.to_thread(self.worker.read_event)
            match event.type:
                case AsrWorkerEventType.READY:
                    raise RuntimeError("The Nemotron worker sent an unexpected ready event.")
                case AsrWorkerEventType.PARTIAL:
                    assert isinstance(event, PartialAsrEvent)
                    await self.partial_text_queue.put(event.text)
                case AsrWorkerEventType.FINAL:
                    assert isinstance(event, FinalAsrEvent)
                    return event.text
                case AsrWorkerEventType.ERROR:
                    assert isinstance(event, AsrWorkerErrorEvent)
                    raise RuntimeError(f"Nemotron worker failed: {event.message}")

    def _release_worker(self) -> None:
        if not self.owns_worker:
            return
        self.owns_worker = False
        self.worker_lock.release()
