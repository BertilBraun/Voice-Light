from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("silero_vad", reason="Compute runtime tests require compute dependencies.")

from app.compute.runtime import ComputeRuntime


class RecordingComputeRuntime(ComputeRuntime):
    def __init__(self, cache_directory: Path) -> None:
        super().__init__(
            voice_stack_settings=None,
            dataset_audio_cache_directory=cache_directory,
        )
        self.events: list[str] = []
        self.parallel_loads_started = asyncio.Event()
        self.parallel_load_count = 0

    async def _load_speech_detector(self) -> None:
        self.events.append("speech_detection")

    async def _load_streaming_asr(self) -> None:
        await self._record_parallel_load("streaming_asr")

    async def _load_language_model(self) -> None:
        await self._record_parallel_load("language_model")

    async def _load_search_text_generator(self) -> None:
        self.events.append("search_summarizer")

    async def _load_speech_synthesizer(self) -> None:
        self.events.append("speech_synthesis")

    async def _record_parallel_load(self, stage_name: str) -> None:
        self.events.append(f"{stage_name}_started")
        self.parallel_load_count += 1
        if self.parallel_load_count == 2:
            self.parallel_loads_started.set()
        await asyncio.wait_for(self.parallel_loads_started.wait(), timeout=1.0)
        self.events.append(f"{stage_name}_completed")


def test_runtime_stages_nemotron_and_qwen_startup_concurrently(tmp_path: Path) -> None:
    async def exercise() -> list[str]:
        runtime = RecordingComputeRuntime(tmp_path / "cache")
        await runtime._load_models()
        return runtime.events

    events = asyncio.run(exercise())

    assert events[0] == "speech_detection"
    assert set(events[1:5]) == {
        "streaming_asr_started",
        "streaming_asr_completed",
        "language_model_started",
        "language_model_completed",
    }
    assert events[5:] == ["search_summarizer", "speech_synthesis"]
