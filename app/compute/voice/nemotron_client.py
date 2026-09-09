from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Final, Protocol, TextIO

from app.compute.voice.asr_worker_protocol import (
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
    TurnAdapterDegradedEvent,
    TurnAdapterPredictionEvent,
    asr_worker_event_adapter,
)
from app.compute.voice.interfaces import (
    TranscriptionSession,
    TurnPredictionObservation,
    TurnPredictionSource,
)
from app.compute.voice.schemas import CapturedAudioChunk, InteractionPrediction
from app.compute.voice.subprocess_start import read_worker_start_event

logger = logging.getLogger(__name__)
NEMOTRON_PYTHON_PATH: Final = Path(sys.executable)
NEMOTRON_SESSION_FINISH_TIMEOUT_SECONDS: Final = 15.0
NEMOTRON_WORKER_STOP_TIMEOUT_SECONDS: Final = 5.0
NEMOTRON_WORKER_START_TIMEOUT_SECONDS: Final = 180.0
NEMOTRON_TURN_PREDICTION_TIMEOUT_SECONDS: Final = 5.0
STREAMING_CHUNK_DURATION_MS: Final = 80
INPUT_SAMPLE_RATE: Final = 16_000
PCM_BYTES_PER_SAMPLE: Final = 2
STREAMING_CHUNK_BYTE_COUNT: Final = (
    INPUT_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE * STREAMING_CHUNK_DURATION_MS // 1_000
)


class NemotronWorker(Protocol):
    @property
    def first_prediction_audio_samples(self) -> int: ...

    @property
    def subsequent_prediction_audio_samples(self) -> int: ...

    @property
    def turn_adapter_available(self) -> bool: ...

    def send(self, command: AsrWorkerCommand) -> None: ...

    def read_event(
        self,
    ) -> (
        AsrWorkerReadyEvent
        | PartialAsrEvent
        | FinalAsrEvent
        | AsrWorkerErrorEvent
        | TurnAdapterPredictionEvent
        | TurnAdapterDegradedEvent
    ): ...

    def terminate(self) -> None: ...


class NemotronWorkerManager(Protocol):
    @property
    def turn_adapter_available(self) -> bool: ...

    async def acquire(self) -> NemotronWorker: ...

    def release(self) -> None: ...

    async def replace(self, failed_worker: NemotronWorker) -> None: ...


class NemotronWorkerProcess:
    def __init__(self, python_path: Path) -> None:
        self.process = subprocess.Popen(
            [python_path.as_posix(), "-m", "app.compute.voice.nemotron_worker"],
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
                NEMOTRON_WORKER_START_TIMEOUT_SECONDS,
                "Nemotron",
            )
            if not isinstance(ready_event, AsrWorkerReadyEvent):
                raise RuntimeError("The Nemotron worker failed to initialize.")
            self.ready_event = ready_event
            if ready_event.turn_adapter_error is not None:
                logger.warning(
                    "Nemotron turn adapter is unavailable: %s",
                    ready_event.turn_adapter_error,
                )
            elif ready_event.turn_adapter_available:
                logger.info(
                    "Nemotron turn adapter loaded: checkpoint_sha256=%s",
                    ready_event.turn_adapter_checkpoint_sha256,
                )
        except BaseException:
            self.terminate()
            raise

    def send(self, command: AsrWorkerCommand) -> None:
        self.input_stream.write(command.model_dump_json() + "\n")
        self.input_stream.flush()

    @property
    def turn_adapter_available(self) -> bool:
        return self.ready_event.turn_adapter_available

    @property
    def first_prediction_audio_samples(self) -> int:
        return self.ready_event.first_prediction_audio_samples

    @property
    def subsequent_prediction_audio_samples(self) -> int:
        return self.ready_event.subsequent_prediction_audio_samples

    def read_event(
        self,
    ) -> (
        AsrWorkerReadyEvent
        | PartialAsrEvent
        | FinalAsrEvent
        | AsrWorkerErrorEvent
        | TurnAdapterPredictionEvent
        | TurnAdapterDegradedEvent
    ):
        event_json = self.output_stream.readline()
        if not event_json:
            raise RuntimeError("The Nemotron worker stopped unexpectedly.")
        return asr_worker_event_adapter.validate_json(event_json)

    def close(self) -> None:
        if self.process.poll() is not None:
            self._close_streams()
            return
        try:
            self.send(ShutdownAsrCommand())
            self.process.wait(timeout=NEMOTRON_WORKER_STOP_TIMEOUT_SECONDS)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.terminate()
            return
        self._close_streams()

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=NEMOTRON_WORKER_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=NEMOTRON_WORKER_STOP_TIMEOUT_SECONDS)
        self._close_streams()

    def _close_streams(self) -> None:
        self.input_stream.close()
        self.output_stream.close()


class RestartingNemotronWorkerManager:
    def __init__(self, python_path: Path) -> None:
        self.python_path = python_path
        self.worker: NemotronWorkerProcess | None = NemotronWorkerProcess(python_path)
        self.lock = asyncio.Lock()

    async def acquire(self) -> NemotronWorker:
        await self.lock.acquire()
        try:
            if self.worker is None:
                self.worker = await asyncio.to_thread(NemotronWorkerProcess, self.python_path)
            return self.worker
        except BaseException:
            self.lock.release()
            raise

    @property
    def turn_adapter_available(self) -> bool:
        return self.worker is not None and self.worker.turn_adapter_available

    def release(self) -> None:
        if not self.lock.locked():
            raise AssertionError("Cannot release an unowned Nemotron worker.")
        self.lock.release()

    async def replace(self, failed_worker: NemotronWorker) -> None:
        if not self.lock.locked():
            raise AssertionError("Cannot replace an unowned Nemotron worker.")
        if failed_worker is not self.worker:
            raise AssertionError("Cannot replace a Nemotron worker owned by another session.")
        logger.error("replacing unresponsive Nemotron worker process")
        self.worker = None
        await asyncio.to_thread(failed_worker.terminate)

    def close(self) -> None:
        if self.worker is None:
            return
        self.worker.close()
        self.worker = None


class NemotronStreamingTranscriber:
    def __init__(self, python_path: Path = NEMOTRON_PYTHON_PATH) -> None:
        self.worker_manager = RestartingNemotronWorkerManager(python_path)
        self.active_session: NemotronStreamingSession | None = None

    def start_session(self) -> TranscriptionSession:
        session = NemotronStreamingSession(
            worker_manager=self.worker_manager,
            finish_timeout_seconds=NEMOTRON_SESSION_FINISH_TIMEOUT_SECONDS,
        )
        self.active_session = session
        return session

    @property
    def turn_adapter_available(self) -> bool:
        return self.worker_manager.turn_adapter_available

    def require_active_session(self) -> NemotronStreamingSession:
        if self.active_session is None:
            raise RuntimeError("No Nemotron transcription session is active.")
        return self.active_session

    def close(self) -> None:
        self.worker_manager.close()


class NemotronTurnPredictionProvider:
    def __init__(self, transcriber: NemotronStreamingTranscriber) -> None:
        self.transcriber = transcriber

    def create_session(self) -> TurnPredictionSource:
        return NemotronTurnPredictionSource(self.transcriber)

    def close(self) -> None:
        return


class NemotronTurnPredictionSource:
    def __init__(self, transcriber: NemotronStreamingTranscriber) -> None:
        self.transcriber = transcriber

    def prediction_expected(self, observation: TurnPredictionObservation) -> bool:
        return self.transcriber.require_active_session().prediction_expected(
            observation.audio_chunk
        )

    async def predict(
        self,
        observation: TurnPredictionObservation,
    ) -> InteractionPrediction | None:
        return await self.transcriber.require_active_session().take_prediction(
            observation.audio_chunk
        )

    async def close(self) -> None:
        return


class NemotronStreamingSession:
    def __init__(
        self,
        worker_manager: NemotronWorkerManager,
        finish_timeout_seconds: float,
    ) -> None:
        self.worker_manager = worker_manager
        self.finish_timeout_seconds = finish_timeout_seconds
        self.worker: NemotronWorker | None = None
        self.pending_audio = bytearray()
        self.latest_chunk: CapturedAudioChunk | None = None
        self.partial_text_queue: asyncio.Queue[str] = asyncio.Queue()
        self.output_task: asyncio.Task[str] | None = None
        self.owns_worker = False
        self.finished = False
        self.expected_prediction_observations: set[str] = set()
        self.predictions: dict[str, InteractionPrediction] = {}
        self.prediction_waiters: dict[str, asyncio.Event] = {}
        self.turn_adapter_error: str | None = None
        self.prediction_audio_samples_remaining: int | None = None

    async def add_audio(self, chunk: CapturedAudioChunk) -> str | None:
        if self.finished:
            raise RuntimeError("Cannot add audio to a finished Nemotron session.")
        await self._ensure_started()
        self.pending_audio.extend(chunk.pcm16)
        self.latest_chunk = chunk
        while len(self.pending_audio) >= STREAMING_CHUNK_BYTE_COUNT:
            audio_frame = bytes(self.pending_audio[:STREAMING_CHUNK_BYTE_COUNT])
            del self.pending_audio[:STREAMING_CHUNK_BYTE_COUNT]
            try:
                self._send_audio_chunk(audio_frame, observation=chunk)
            except Exception as error:
                await self._fail_worker(error)
                raise RuntimeError("Failed to send audio to the Nemotron worker.") from error
        latest_partial_text: str | None = None
        while not self.partial_text_queue.empty():
            latest_partial_text = self.partial_text_queue.get_nowait()
        return latest_partial_text

    async def take_prediction(
        self,
        chunk: CapturedAudioChunk,
        timeout_seconds: float = NEMOTRON_TURN_PREDICTION_TIMEOUT_SECONDS,
    ) -> InteractionPrediction | None:
        observation_id = _observation_id(chunk)
        if observation_id not in self.expected_prediction_observations:
            return None
        if self.turn_adapter_error is not None:
            raise RuntimeError(self.turn_adapter_error)
        waiter = self.prediction_waiters[observation_id]
        try:
            await asyncio.wait_for(waiter.wait(), timeout=timeout_seconds)
        except TimeoutError:
            self.expected_prediction_observations.discard(observation_id)
            self.prediction_waiters.pop(observation_id, None)
            return None
        self.expected_prediction_observations.discard(observation_id)
        self.prediction_waiters.pop(observation_id, None)
        if self.turn_adapter_error is not None:
            raise RuntimeError(self.turn_adapter_error)
        return self.predictions.pop(observation_id)

    def prediction_expected(self, chunk: CapturedAudioChunk) -> bool:
        return _observation_id(chunk) in self.expected_prediction_observations

    async def finish(self) -> str:
        if self.finished:
            raise RuntimeError("Cannot finish a Nemotron session twice.")
        self.finished = True
        if self.output_task is None:
            return ""
        if self.pending_audio:
            try:
                assert self.latest_chunk is not None
                self._send_audio_chunk(
                    bytes(self.pending_audio),
                    observation=self.latest_chunk,
                    final_audio=True,
                )
            except Exception as error:
                await self._fail_worker(error)
                raise RuntimeError("Failed to send final audio to the Nemotron worker.") from error
            self.pending_audio.clear()
        try:
            self._require_worker().send(FinishAsrCommand())
        except Exception as error:
            await self._fail_worker(error)
            raise RuntimeError("Failed to finish the Nemotron worker session.") from error
        return await self._await_final_output()

    async def close(self) -> None:
        self.pending_audio.clear()
        if self.output_task is None or self.finished:
            return
        self.finished = True
        try:
            self._require_worker().send(FinishAsrCommand())
        except Exception as error:
            await self._fail_worker(error)
            raise RuntimeError("Failed to close the Nemotron worker session.") from error
        await self._await_final_output()

    async def _ensure_started(self) -> None:
        if self.output_task is not None:
            return
        self.worker = await self.worker_manager.acquire()
        self.owns_worker = True
        self.prediction_audio_samples_remaining = self.worker.first_prediction_audio_samples
        try:
            self.worker.send(StartAsrCommand())
        except Exception as error:
            await self._fail_worker(error)
            raise RuntimeError("Failed to start the Nemotron worker session.") from error
        self.output_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> str:
        while True:
            event = await asyncio.to_thread(self._require_worker().read_event)
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
                case AsrWorkerEventType.TURN_PREDICTION:
                    assert isinstance(event, TurnAdapterPredictionEvent)
                    observation_id = event.prediction.stamp.observation_id
                    waiter = self.prediction_waiters.get(observation_id)
                    if waiter is not None:
                        self.predictions[observation_id] = event.prediction
                        waiter.set()
                case AsrWorkerEventType.TURN_ADAPTER_DEGRADED:
                    assert isinstance(event, TurnAdapterDegradedEvent)
                    self.turn_adapter_error = event.reason
                    for waiter in self.prediction_waiters.values():
                        waiter.set()

    def _send_audio_chunk(
        self,
        pcm_bytes: bytes,
        observation: CapturedAudioChunk,
        final_audio: bool = False,
    ) -> None:
        observation_id = _observation_id(observation)
        if self._prediction_expected(len(pcm_bytes) // PCM_BYTES_PER_SAMPLE, final_audio):
            self.expected_prediction_observations.add(observation_id)
            self.prediction_waiters[observation_id] = asyncio.Event()
        self._require_worker().send(
            AsrAudioCommand.from_observation(
                pcm_bytes=pcm_bytes,
                sequence_number=observation.sequence_number,
                start_input_sample=observation.start_input_sample,
                end_input_sample=observation.end_input_sample,
                observation_monotonic_time_ns=observation.monotonic_observation_time_ns,
                stream_epoch=observation.stream_epoch,
                turn_epoch=observation.turn_epoch,
                silero_is_speech=observation.silero_evidence.is_speech,
                playback_condition=observation.playback_condition,
            )
        )

    def _prediction_expected(self, sample_count: int, final_audio: bool) -> bool:
        worker = self._require_worker()
        if not worker.turn_adapter_available:
            return False
        samples_remaining = self.prediction_audio_samples_remaining
        if samples_remaining is None:
            raise AssertionError("Nemotron prediction cadence is not initialized.")
        if sample_count < samples_remaining and not final_audio:
            self.prediction_audio_samples_remaining = samples_remaining - sample_count
            return False
        overflow_samples = max(sample_count - samples_remaining, 0)
        prediction_interval = worker.subsequent_prediction_audio_samples
        while overflow_samples >= prediction_interval:
            overflow_samples -= prediction_interval
        self.prediction_audio_samples_remaining = prediction_interval - overflow_samples
        return True

    async def _await_final_output(self) -> str:
        assert self.output_task is not None
        worker = self._require_worker()
        try:
            return await asyncio.wait_for(
                self.output_task,
                timeout=self.finish_timeout_seconds,
            )
        except asyncio.CancelledError:
            await self.worker_manager.replace(worker)
            raise
        except TimeoutError as error:
            await self.worker_manager.replace(worker)
            raise RuntimeError("Nemotron transcription finalization timed out.") from error
        except Exception:
            await self.worker_manager.replace(worker)
            raise
        finally:
            self._release_worker()

    async def _fail_worker(self, error: Exception) -> None:
        worker = self._require_worker()
        self.finished = True
        self.pending_audio.clear()
        logger.error("Nemotron worker operation failed: %s", error)
        try:
            await self.worker_manager.replace(worker)
        finally:
            self._release_worker()
            output_task = self.output_task
            if (
                output_task is not None
                and output_task is not asyncio.current_task()
                and not output_task.done()
            ):
                output_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await output_task

    def _require_worker(self) -> NemotronWorker:
        if self.worker is None:
            raise AssertionError("The Nemotron session does not own a worker.")
        return self.worker

    def _release_worker(self) -> None:
        if not self.owns_worker:
            return
        self.owns_worker = False
        self.worker = None
        self.worker_manager.release()


def _observation_id(chunk: CapturedAudioChunk) -> str:
    return f"audio:{chunk.stream_epoch}:{chunk.sequence_number}"
