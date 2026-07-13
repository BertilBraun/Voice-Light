from __future__ import annotations

import asyncio
import queue

from app.voice_agent.asr_worker_protocol import (
    AsrAudioCommand,
    AsrWorkerCommand,
    AsrWorkerCommandType,
    AsrWorkerErrorEvent,
    AsrWorkerReadyEvent,
    FinalAsrEvent,
    PartialAsrEvent,
)
from app.voice_agent.nemotron_client import (
    STREAMING_CHUNK_BYTE_COUNT,
    NemotronStreamingSession,
)


class FakeNemotronWorker:
    def __init__(self) -> None:
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
                self.events.put(FinalAsrEvent(text="hello world"))

    def read_event(
        self,
    ) -> AsrWorkerReadyEvent | PartialAsrEvent | FinalAsrEvent | AsrWorkerErrorEvent:
        return self.events.get(timeout=1)


def test_nemotron_session_streams_partial_and_final_text() -> None:
    async def run_session() -> None:
        worker = FakeNemotronWorker()
        session = NemotronStreamingSession(worker=worker, worker_lock=asyncio.Lock())

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
