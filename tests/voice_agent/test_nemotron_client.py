from __future__ import annotations

import asyncio
import queue

import pytest

from app.compute.voice.asr_worker_protocol import (
    AsrAudioCommand,
    AsrWorkerCommand,
    AsrWorkerCommandType,
    AsrWorkerErrorEvent,
    AsrWorkerReadyEvent,
    FinalAsrEvent,
    PartialAsrEvent,
)
from app.compute.voice.nemotron_client import (
    STREAMING_CHUNK_BYTE_COUNT,
    NemotronStreamingSession,
    NemotronWorker,
)


class FakeNemotronWorker:
    def __init__(self, finalizes_on_finish: bool = True) -> None:
        self.finalizes_on_finish = finalizes_on_finish
        self.commands: list[AsrWorkerCommand] = []
        self.events: queue.Queue[
            AsrWorkerReadyEvent | PartialAsrEvent | FinalAsrEvent | AsrWorkerErrorEvent
        ] = queue.Queue()

    def send(self, command: AsrWorkerCommand) -> None:
        self.commands.append(command)
        match command.type:
            case AsrWorkerCommandType.AUDIO:
                self.events.put(PartialAsrEvent(text="hello"))
            case AsrWorkerCommandType.FINISH:
                if self.finalizes_on_finish:
                    self.events.put(FinalAsrEvent(text="hello world"))

    def read_event(
        self,
    ) -> AsrWorkerReadyEvent | PartialAsrEvent | FinalAsrEvent | AsrWorkerErrorEvent:
        return self.events.get(timeout=1)

    def terminate(self) -> None:
        self.events.put(FinalAsrEvent(text=""))


class FakeNemotronWorkerManager:
    def __init__(self, worker: FakeNemotronWorker) -> None:
        self.worker = worker
        self.lock = asyncio.Lock()
        self.replacement_count = 0

    async def acquire(self) -> FakeNemotronWorker:
        await self.lock.acquire()
        return self.worker

    def release(self) -> None:
        self.lock.release()

    async def replace(self, failed_worker: NemotronWorker) -> None:
        assert failed_worker is self.worker
        self.replacement_count += 1
        self.worker.terminate()


def test_nemotron_session_streams_partial_and_final_text() -> None:
    async def run_session() -> None:
        worker = FakeNemotronWorker()
        worker_manager = FakeNemotronWorkerManager(worker)
        session = NemotronStreamingSession(
            worker_manager=worker_manager,
            finish_timeout_seconds=1.0,
        )

        assert await session.add_audio(b"\x01\x00" * (STREAMING_CHUNK_BYTE_COUNT // 2)) is None
        partial_text: str | None = None
        async with asyncio.timeout(1.0):
            while partial_text is None:
                await asyncio.sleep(0.01)
                partial_text = await session.add_audio(b"")
        assert partial_text == "hello"
        assert await session.finish() == "hello world"

        assert [command.type for command in worker.commands] == [
            AsrWorkerCommandType.START,
            AsrWorkerCommandType.AUDIO,
            AsrWorkerCommandType.FINISH,
        ]
        audio_command = worker.commands[1]
        assert isinstance(audio_command, AsrAudioCommand)
        assert len(audio_command.pcm_bytes()) == STREAMING_CHUNK_BYTE_COUNT

    asyncio.run(run_session())


def test_nemotron_session_replaces_worker_after_finalization_timeout() -> None:
    async def run_session() -> None:
        worker = FakeNemotronWorker(finalizes_on_finish=False)
        worker_manager = FakeNemotronWorkerManager(worker)
        session = NemotronStreamingSession(
            worker_manager=worker_manager,
            finish_timeout_seconds=0.01,
        )

        await session.add_audio(b"\x01\x00" * (STREAMING_CHUNK_BYTE_COUNT // 2))
        with pytest.raises(RuntimeError, match="finalization timed out"):
            await session.finish()

        assert worker_manager.replacement_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_session())
