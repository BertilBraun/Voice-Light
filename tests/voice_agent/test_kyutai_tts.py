from __future__ import annotations

import asyncio
import queue
from collections.abc import Callable

import pytest

from app.compute.voice.interfaces import (
    KyutaiSynthesisFirstAudioMetrics,
    SynthesisEvent,
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
    VoxtreamSynthesisFirstAudioMetrics,
)
from app.compute.voice.subprocess_tts import (
    SubprocessSpeechSynthesisSession,
    TtsWorker,
)
from app.compute.voice.tts_worker_protocol import (
    TtsAudioEvent,
    TtsEndEvent,
    TtsFirstAudioMetricsEvent,
    TtsWordBoundaryEvent,
    TtsWordCommand,
    TtsWordProcessedEvent,
    TtsWorkerCommand,
    TtsWorkerCommandType,
    TtsWorkerEvent,
    VoxtreamTtsFirstAudioMetricsEvent,
)


class FakeKyutaiTtsWorker:
    def __init__(
        self,
        processes_words: bool = True,
        ends_on_cancel: bool = True,
        first_audio_metrics_event: TtsWorkerEvent | None = None,
    ) -> None:
        self.processes_words = processes_words
        self.ends_on_cancel = ends_on_cancel
        self.first_audio_metrics_event = (
            first_audio_metrics_event
            if first_audio_metrics_event is not None
            else TtsFirstAudioMetricsEvent(
                first_word_to_audio_seconds=0.25,
                tokenization_seconds=0.01,
                language_model_step_seconds=0.2,
                mimi_decode_seconds=0.04,
                model_step_count=6,
                first_audio_model_step=6,
            )
        )
        self.commands: list[TtsWorkerCommand] = []
        self.events: queue.Queue[TtsWorkerEvent] = queue.Queue()
        self.termination_count = 0

    @property
    def sample_rate(self) -> int:
        return 24_000

    def send(self, command: TtsWorkerCommand) -> None:
        self.commands.append(command)
        match command.type:
            case TtsWorkerCommandType.WORD:
                assert isinstance(command, TtsWordCommand)
                if self.processes_words:
                    if command.sequence_number == 0:
                        self.events.put(self.first_audio_metrics_event)
                    self.events.put(
                        TtsWordBoundaryEvent(
                            text_offset=command.text_end,
                            start_sample=0,
                        )
                    )
                    self.events.put(
                        TtsAudioEvent.from_pcm_bytes(
                            pcm_bytes=b"\x01\x00\x02\x00",
                            start_sample=0,
                        )
                    )
                    self.events.put(TtsWordProcessedEvent(sequence_number=command.sequence_number))
            case TtsWorkerCommandType.FINISH:
                self.events.put(TtsEndEvent(cancelled=False))
            case TtsWorkerCommandType.CANCEL:
                if self.ends_on_cancel:
                    self.events.put(TtsEndEvent(cancelled=True))

    def read_event(self) -> TtsWorkerEvent:
        return self.events.get(timeout=1.0)

    def terminate(self) -> None:
        self.termination_count += 1
        self.events.put(TtsEndEvent(cancelled=True))


class FakeKyutaiTtsWorkerManager:
    def __init__(
        self,
        worker: FakeKyutaiTtsWorker,
        replacement_worker_factory: Callable[[], FakeKyutaiTtsWorker] = FakeKyutaiTtsWorker,
    ) -> None:
        self.worker = worker
        self.replacement_worker_factory = replacement_worker_factory
        self.workers = [worker]
        self.lock = asyncio.Lock()
        self.replacement_count = 0

    @property
    def sample_rate(self) -> int:
        return self.worker.sample_rate

    async def acquire(self) -> FakeKyutaiTtsWorker:
        await self.lock.acquire()
        return self.worker

    def release(self) -> None:
        self.lock.release()

    async def replace(self, failed_worker: TtsWorker) -> None:
        assert failed_worker is self.worker
        self.replacement_count += 1
        self.worker.terminate()
        self.worker = self.replacement_worker_factory()
        self.workers.append(self.worker)


def _make_session(
    worker_manager: FakeKyutaiTtsWorkerManager,
    progress_timeout_seconds: float = 1.0,
    cancel_timeout_seconds: float = 1.0,
) -> SubprocessSpeechSynthesisSession:
    return SubprocessSpeechSynthesisSession(
        worker_manager=worker_manager,
        provider_name="Test TTS",
        generation_progress_timeout_seconds=progress_timeout_seconds,
        cancel_timeout_seconds=cancel_timeout_seconds,
        watchdog_poll_seconds=0.001,
    )


async def _collect_events(session: SubprocessSpeechSynthesisSession) -> list[SynthesisEvent]:
    return [event async for event in session.stream_events()]


def test_kyutai_session_streams_audio_and_word_boundaries() -> None:
    async def run_session() -> None:
        worker = FakeKyutaiTtsWorker()
        worker_manager = FakeKyutaiTtsWorkerManager(worker)
        session = _make_session(worker_manager)

        await session.add_word(SynthesisWord(text="Hello", text_start=0, text_end=5))
        await session.finish_input()
        events = await _collect_events(session)

        assert events == [
            KyutaiSynthesisFirstAudioMetrics(
                first_word_to_audio_seconds=0.25,
                tokenization_seconds=0.01,
                language_model_step_seconds=0.2,
                mimi_decode_seconds=0.04,
                model_step_count=6,
                first_audio_model_step=6,
            ),
            SynthesizedWordBoundary(text_offset=5, start_sample=0),
            SynthesizedAudioChunk(pcm_bytes=b"\x01\x00\x02\x00", start_sample=0),
        ]
        assert [command.type for command in worker.commands] == [
            TtsWorkerCommandType.START,
            TtsWorkerCommandType.WORD,
            TtsWorkerCommandType.FINISH,
        ]
        assert not worker_manager.lock.locked()
        assert worker_manager.replacement_count == 0

    asyncio.run(run_session())


def test_kyutai_session_cancels_cooperatively() -> None:
    async def run_session() -> None:
        worker = FakeKyutaiTtsWorker()
        worker_manager = FakeKyutaiTtsWorkerManager(worker)
        session = _make_session(worker_manager)

        await session.add_word(SynthesisWord(text="Stop", text_start=0, text_end=4))
        await session.cancel()

        assert worker.commands[-1].type is TtsWorkerCommandType.CANCEL
        assert worker_manager.replacement_count == 0
        assert not worker_manager.lock.locked()

    asyncio.run(run_session())


def test_subprocess_session_maps_voxtream_metrics() -> None:
    async def run_session() -> None:
        worker = FakeKyutaiTtsWorker(
            first_audio_metrics_event=VoxtreamTtsFirstAudioMetricsEvent(
                first_word_to_audio_seconds=0.12,
                prompt_preparation_seconds=0.05,
                first_frame_generation_seconds=0.06,
            )
        )
        worker_manager = FakeKyutaiTtsWorkerManager(worker)
        session = _make_session(worker_manager)

        await session.add_word(SynthesisWord(text="Hello", text_start=0, text_end=5))
        await session.finish_input()
        events = await _collect_events(session)

        assert events[0] == VoxtreamSynthesisFirstAudioMetrics(
            first_word_to_audio_seconds=0.12,
            prompt_preparation_seconds=0.05,
            first_frame_generation_seconds=0.06,
        )

    asyncio.run(run_session())


def test_kyutai_session_replaces_worker_after_progress_timeout() -> None:
    async def run_session() -> None:
        worker = FakeKyutaiTtsWorker(processes_words=False)
        worker_manager = FakeKyutaiTtsWorkerManager(worker)
        session = _make_session(worker_manager, progress_timeout_seconds=0.02)

        await session.add_word(SynthesisWord(text="Hang", text_start=0, text_end=4))
        with pytest.raises(RuntimeError, match="stopped making progress"):
            async with asyncio.timeout(1.0):
                await _collect_events(session)

        assert worker_manager.replacement_count == 1
        assert worker.termination_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_session())


def test_kyutai_session_replaces_worker_after_cancel_timeout() -> None:
    async def run_session() -> None:
        worker = FakeKyutaiTtsWorker(ends_on_cancel=False)
        worker_manager = FakeKyutaiTtsWorkerManager(worker)
        session = _make_session(worker_manager, cancel_timeout_seconds=0.01)

        await session.add_word(SynthesisWord(text="Stop", text_start=0, text_end=4))
        with pytest.raises(RuntimeError, match="cancellation timed out"):
            await session.cancel()

        assert worker_manager.replacement_count == 1
        assert worker.termination_count == 1
        assert not worker_manager.lock.locked()

    asyncio.run(run_session())


def test_kyutai_worker_is_reused_across_completed_turns() -> None:
    async def run_sessions() -> None:
        worker = FakeKyutaiTtsWorker()
        worker_manager = FakeKyutaiTtsWorkerManager(worker)

        for _ in range(2):
            session = _make_session(worker_manager)
            await session.add_word(SynthesisWord(text="Again", text_start=0, text_end=5))
            await session.finish_input()
            await _collect_events(session)

        assert [
            command.type
            for command in worker.commands
            if command.type is TtsWorkerCommandType.START
        ] == [TtsWorkerCommandType.START, TtsWorkerCommandType.START]
        assert worker_manager.replacement_count == 0
        assert not worker_manager.lock.locked()

    asyncio.run(run_sessions())


def test_kyutai_session_recovers_on_fresh_worker_after_progress_timeout() -> None:
    async def run_sessions() -> None:
        failed_worker = FakeKyutaiTtsWorker(processes_words=False)
        replacement_worker = FakeKyutaiTtsWorker()
        worker_manager = FakeKyutaiTtsWorkerManager(
            failed_worker,
            replacement_worker_factory=lambda: replacement_worker,
        )
        failed_session = _make_session(worker_manager, progress_timeout_seconds=0.02)

        await failed_session.add_word(SynthesisWord(text="Hang", text_start=0, text_end=4))
        with pytest.raises(RuntimeError, match="stopped making progress"):
            async with asyncio.timeout(1.0):
                await _collect_events(failed_session)

        recovered_session = _make_session(worker_manager)
        await recovered_session.add_word(SynthesisWord(text="Recovered", text_start=0, text_end=9))
        await recovered_session.finish_input()
        events = await _collect_events(recovered_session)

        assert events == [
            KyutaiSynthesisFirstAudioMetrics(
                first_word_to_audio_seconds=0.25,
                tokenization_seconds=0.01,
                language_model_step_seconds=0.2,
                mimi_decode_seconds=0.04,
                model_step_count=6,
                first_audio_model_step=6,
            ),
            SynthesizedWordBoundary(text_offset=9, start_sample=0),
            SynthesizedAudioChunk(pcm_bytes=b"\x01\x00\x02\x00", start_sample=0),
        ]
        assert worker_manager.workers == [failed_worker, replacement_worker]
        assert worker_manager.worker is replacement_worker
        assert failed_worker.termination_count == 1
        assert [command.type for command in replacement_worker.commands] == [
            TtsWorkerCommandType.START,
            TtsWorkerCommandType.WORD,
            TtsWorkerCommandType.FINISH,
        ]
        assert not worker_manager.lock.locked()

    asyncio.run(run_sessions())
