from __future__ import annotations

import asyncio
import io
import queue
from pathlib import Path

import pytest

import app.compute.voice.nemotron_client as nemotron_client
from app.compute.voice.asr_worker_protocol import (
    AsrAudioCommand,
    AsrWorkerCommand,
    AsrWorkerCommandType,
    AsrWorkerErrorEvent,
    AsrWorkerReadyEvent,
    FinalAsrEvent,
    PartialAsrEvent,
    asr_worker_command_adapter,
)
from app.compute.voice.nemotron_client import (
    STREAMING_CHUNK_BYTE_COUNT,
    NemotronStreamingSession,
    NemotronWorker,
    RestartingNemotronWorkerManager,
)
from app.compute.voice.schemas import (
    CapturedAudioChunk,
    PlaybackCondition,
    PlaybackConditionAuthority,
    PlaybackState,
    SileroEvidence,
)


class FakeNemotronWorker:
    def __init__(
        self,
        finalizes_on_finish: bool = True,
        failing_command_type: AsrWorkerCommandType | None = None,
        turn_adapter_available: bool = False,
    ) -> None:
        self.finalizes_on_finish = finalizes_on_finish
        self.failing_command_type = failing_command_type
        self._turn_adapter_available = turn_adapter_available
        self.commands: list[AsrWorkerCommand] = []
        self.events: queue.Queue[
            AsrWorkerReadyEvent | PartialAsrEvent | FinalAsrEvent | AsrWorkerErrorEvent
        ] = queue.Queue()

    @property
    def first_prediction_audio_samples(self) -> int:
        return 2_560

    @property
    def subsequent_prediction_audio_samples(self) -> int:
        return 1_280

    @property
    def turn_adapter_available(self) -> bool:
        return self._turn_adapter_available

    def send(self, command: AsrWorkerCommand) -> None:
        if command.type is self.failing_command_type:
            raise BrokenPipeError("synthetic send failure")
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

    @property
    def turn_adapter_available(self) -> bool:
        return self.worker.turn_adapter_available

    async def acquire(self) -> FakeNemotronWorker:
        await self.lock.acquire()
        return self.worker

    def release(self) -> None:
        self.lock.release()

    async def replace(self, failed_worker: NemotronWorker) -> None:
        assert failed_worker is self.worker
        self.replacement_count += 1
        self.worker.terminate()


class FakeNemotronProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(PartialAsrEvent(text="unexpected").model_dump_json() + "\n")
        self.terminated = False

    def poll(self) -> int | None:
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> int:
        del timeout
        self.terminated = True
        return 0

    def kill(self) -> None:
        self.terminated = True


def test_nemotron_session_streams_partial_and_final_text() -> None:
    async def run_session() -> None:
        worker = FakeNemotronWorker()
        worker_manager = FakeNemotronWorkerManager(worker)
        session = NemotronStreamingSession(
            worker_manager=worker_manager,
            finish_timeout_seconds=1.0,
        )

        assert await session.add_audio(_chunk(STREAMING_CHUNK_BYTE_COUNT)) is None
        partial_text: str | None = None
        async with asyncio.timeout(1.0):
            while partial_text is None:
                await asyncio.sleep(0.01)
                partial_text = await session.add_audio(_chunk(0, sequence_number=1))
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
        serialized = audio_command.model_dump_json()
        assert asr_worker_command_adapter.validate_json(serialized) == audio_command

    asyncio.run(run_session())


def test_nemotron_prediction_waiters_follow_encoder_chunk_boundaries() -> None:
    async def run_session() -> None:
        worker = FakeNemotronWorker(turn_adapter_available=True)
        session = NemotronStreamingSession(
            worker_manager=FakeNemotronWorkerManager(worker),
            finish_timeout_seconds=1.0,
        )
        first_chunk = _chunk(STREAMING_CHUNK_BYTE_COUNT)
        second_chunk = _chunk(STREAMING_CHUNK_BYTE_COUNT, sequence_number=1)

        await session.add_audio(first_chunk)
        assert session.expected_prediction_observations == set()
        assert await session.take_prediction(first_chunk) is None

        await session.add_audio(second_chunk)
        assert session.expected_prediction_observations == {"audio:1:1"}
        await session.close()

    asyncio.run(run_session())


def test_asr_ready_event_serializes_prediction_cadence() -> None:
    event = AsrWorkerReadyEvent(
        first_prediction_audio_samples=16_000,
        subsequent_prediction_audio_samples=1_280,
        turn_adapter_available=True,
        turn_adapter_checkpoint_sha256="abc",
    )

    assert event.model_dump(mode="json") == {
        "type": "ready",
        "first_prediction_audio_samples": 16_000,
        "subsequent_prediction_audio_samples": 1_280,
        "turn_adapter_available": True,
        "turn_adapter_checkpoint_sha256": "abc",
        "turn_adapter_error": None,
    }


def test_nemotron_session_replaces_worker_after_finalization_timeout() -> None:
    async def run_session() -> None:
        worker = FakeNemotronWorker(finalizes_on_finish=False)
        worker_manager = FakeNemotronWorkerManager(worker)
        session = NemotronStreamingSession(
            worker_manager=worker_manager,
            finish_timeout_seconds=0.01,
        )

        await session.add_audio(_chunk(STREAMING_CHUNK_BYTE_COUNT))
        with pytest.raises(RuntimeError, match="finalization timed out"):
            await session.finish()

        assert worker_manager.replacement_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_session())


@pytest.mark.parametrize(
    ("failing_command_type", "expected_message"),
    [
        (AsrWorkerCommandType.START, "Failed to start"),
        (AsrWorkerCommandType.AUDIO, "Failed to send audio"),
        (AsrWorkerCommandType.FINISH, "Failed to finish"),
    ],
)
def test_nemotron_send_failure_replaces_worker_and_releases_lock(
    failing_command_type: AsrWorkerCommandType,
    expected_message: str,
) -> None:
    async def run_session() -> None:
        worker = FakeNemotronWorker(failing_command_type=failing_command_type)
        worker_manager = FakeNemotronWorkerManager(worker)
        session = NemotronStreamingSession(
            worker_manager=worker_manager,
            finish_timeout_seconds=1.0,
        )

        with pytest.raises(RuntimeError, match=expected_message):
            if failing_command_type is AsrWorkerCommandType.FINISH:
                await session.add_audio(_chunk(STREAMING_CHUNK_BYTE_COUNT))
                await session.finish()
            else:
                await session.add_audio(_chunk(STREAMING_CHUNK_BYTE_COUNT))

        await session.close()
        assert worker_manager.replacement_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_session())


def test_nemotron_spawn_failure_releases_manager_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_count = 0

    def create_worker(
        python_path: Path,
        cuda_device: nemotron_client.CudaWorkerDevice | None,
    ) -> FakeNemotronWorker:
        nonlocal construction_count
        del python_path, cuda_device
        construction_count += 1
        if construction_count > 1:
            raise RuntimeError("synthetic spawn failure")
        return FakeNemotronWorker()

    monkeypatch.setattr(nemotron_client, "NemotronWorkerProcess", create_worker)
    worker_manager = RestartingNemotronWorkerManager(Path("python"))
    worker_manager.worker = None

    async def acquire_worker() -> None:
        with pytest.raises(RuntimeError, match="synthetic spawn failure"):
            await worker_manager.acquire()
        assert not worker_manager.lock.locked()

    asyncio.run(acquire_worker())


def test_nemotron_wrong_start_event_terminates_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeNemotronProcess()

    def create_process(*arguments: object, **options: object) -> FakeNemotronProcess:
        del arguments, options
        return process

    monkeypatch.setattr(nemotron_client.subprocess, "Popen", create_process)

    with pytest.raises(RuntimeError, match="failed to initialize"):
        nemotron_client.NemotronWorkerProcess(Path("python"))

    assert process.terminated


def _chunk(byte_count: int, sequence_number: int = 0) -> CapturedAudioChunk:
    sample_count = byte_count // 2
    return CapturedAudioChunk(
        pcm16=b"\x01\x00" * sample_count,
        sequence_number=sequence_number,
        start_input_sample=0,
        end_input_sample=sample_count,
        monotonic_observation_time_ns=1,
        stream_epoch=1,
        turn_epoch=1,
        silero_evidence=SileroEvidence(is_speech=True, monotonic_time_ns=1),
        playback_condition=PlaybackCondition(
            event_id=f"playback-{sequence_number}",
            generation_id=None,
            state=PlaybackState.IDLE,
            assistant_audible=False,
            latest_output_sample_position=0,
            latest_source_sample_position=0,
            output_sample_rate=None,
            monotonic_time_ns=1,
            authority=PlaybackConditionAuthority.SERVER_ESTIMATED,
        ),
    )
