from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("silero_vad", reason="Compute runtime tests require compute dependencies.")

from app.compute.runtime import ComputeRuntime
from app.shared.compute_api import ModelStageStatus, ModelWarmupStatus


class RecordingComputeRuntime(ComputeRuntime):
    def __init__(self, cache_directory: Path) -> None:
        super().__init__(
            voice_stack_settings=None,
            dataset_audio_cache_directory=cache_directory,
        )
        self.events: list[str] = []
        self.parallel_loads_started = asyncio.Event()
        self.parallel_load_count = 0
        self.early_warmup_started = asyncio.Event()

    async def _load_speech_detector(self) -> None:
        self.events.append("speech_detection")

    async def _load_streaming_asr(self) -> None:
        await self._record_parallel_load("streaming_asr")

    async def _load_language_model(self) -> None:
        await self._record_parallel_load("language_model")

    async def _load_search_text_generator(self) -> None:
        self.events.append("search_summarizer")

    async def _load_speech_synthesizer(self) -> None:
        await self._record_parallel_load("speech_synthesis")

    async def _warm_streaming_asr(self) -> None:
        await self._record_warmup("streaming_asr")

    async def _warm_language_model(self) -> None:
        await self._record_warmup("language_model")

    async def _warm_speech_synthesizer(self) -> None:
        await self._record_warmup("speech_synthesis")

    async def _record_parallel_load(self, stage_name: str) -> None:
        self.events.append(f"{stage_name}_started")
        self.parallel_load_count += 1
        if self.parallel_load_count == 3:
            self.parallel_loads_started.set()
        await asyncio.wait_for(self.parallel_loads_started.wait(), timeout=1.0)
        if stage_name == "speech_synthesis":
            await asyncio.wait_for(self.early_warmup_started.wait(), timeout=1.0)
        self.events.append(f"{stage_name}_completed")

    async def _record_warmup(self, stage_name: str) -> None:
        self.events.append(f"{stage_name}_warmup_started")
        if stage_name == "streaming_asr":
            self.early_warmup_started.set()
        await asyncio.sleep(0)
        self.events.append(f"{stage_name}_warmup_completed")


def test_runtime_loads_and_warms_voice_models_concurrently(tmp_path: Path) -> None:
    async def exercise() -> list[str]:
        runtime = RecordingComputeRuntime(tmp_path / "cache")
        await runtime._load_models()
        return runtime.events

    events = asyncio.run(exercise())

    assert events[0] == "speech_detection"
    assert set(events[1:4]) == {
        "streaming_asr_started",
        "language_model_started",
        "speech_synthesis_started",
    }
    assert events.index("streaming_asr_warmup_started") < events.index("speech_synthesis_completed")
    for stage_name in ("streaming_asr", "language_model", "speech_synthesis"):
        assert events.index(f"{stage_name}_completed") < events.index(
            f"{stage_name}_warmup_started"
        )
        assert events.index(f"{stage_name}_warmup_started") < events.index(
            f"{stage_name}_warmup_completed"
        )
    assert events[-1] == "search_summarizer"


def test_warmup_telemetry_is_ready_only_after_probe(tmp_path: Path) -> None:
    async def exercise() -> tuple[ModelWarmupStatus, float | None]:
        runtime = RecordingComputeRuntime(tmp_path / "cache")
        stage = runtime.language_model_stage
        stage.status = ModelStageStatus.READY

        async def probe() -> None:
            assert stage.warmup_status is ModelWarmupStatus.RUNNING

        await runtime._timed_warmup(stage, probe)
        return stage.warmup_status, stage.warmup_time_seconds

    status, elapsed = asyncio.run(exercise())
    assert status is ModelWarmupStatus.READY
    assert elapsed is not None


def test_failed_warmup_prevents_readiness(tmp_path: Path) -> None:
    async def exercise() -> tuple[ModelStageStatus, ModelWarmupStatus, str | None]:
        runtime = RecordingComputeRuntime(tmp_path / "cache")
        stage = runtime.language_model_stage
        stage.status = ModelStageStatus.READY

        async def probe() -> None:
            raise RuntimeError("probe failed")

        await runtime._timed_warmup(stage, probe)
        return stage.status, stage.warmup_status, stage.warmup_error

    status, warmup_status, error = asyncio.run(exercise())
    assert status is ModelStageStatus.FAILED
    assert warmup_status is ModelWarmupStatus.FAILED
    assert error == "probe failed"
