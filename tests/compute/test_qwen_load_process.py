from __future__ import annotations

import asyncio
from pathlib import Path

from app.compute.voice.interfaces import TextGenerationRequest
from deployment.compute.qwen_load_process import (
    ActiveQwenWorkers,
    QwenLoadMode,
    QwenLoadReadySignal,
    QwenLoadRuntimeProtocol,
    run_load_process,
)


class FakeTextGenerator:
    def __init__(self, stop_event: asyncio.Event | None = None) -> None:
        self.stop_event = stop_event
        self.requests: list[TextGenerationRequest] = []

    async def generate_text(self, request: TextGenerationRequest) -> str:
        self.requests.append(request)
        if self.stop_event is not None:
            self.stop_event.set()
        return "Generated load response."


class FakeQwenLoadRuntime:
    def __init__(self, stop_event: asyncio.Event | None = None) -> None:
        self.conversation = FakeTextGenerator(stop_event)
        self.search = FakeTextGenerator()
        self.closed = False

    def close(self) -> None:
        self.closed = True


async def wait_for_file(path: Path) -> None:
    for _ in range(100):
        if path.is_file():
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"Timed out waiting for {path}.")


def test_resident_idle_load_signals_readiness_and_closes_both_workers(
    tmp_path: Path,
) -> None:
    async def run_test() -> None:
        stop_event = asyncio.Event()
        runtime = FakeQwenLoadRuntime()
        ready_file = tmp_path / "resident-ready.json"

        def create_runtime(python_path: Path) -> QwenLoadRuntimeProtocol:
            assert python_path == Path("vllm-python")
            return runtime

        task = asyncio.create_task(
            run_load_process(
                mode=QwenLoadMode.RESIDENT_IDLE,
                active_workers=ActiveQwenWorkers.CONVERSATION,
                python_path=Path("vllm-python"),
                ready_file=ready_file,
                runtime_factory=create_runtime,
                stop_event=stop_event,
            )
        )
        await wait_for_file(ready_file)
        ready_signal = QwenLoadReadySignal.model_validate_json(
            ready_file.read_text(encoding="utf-8")
        )
        assert ready_signal.mode is QwenLoadMode.RESIDENT_IDLE
        assert ready_signal.active_workers is ActiveQwenWorkers.CONVERSATION
        assert not runtime.conversation.requests
        assert not runtime.search.requests

        stop_event.set()
        await task
        assert runtime.closed
        assert not ready_file.exists()

    asyncio.run(run_test())


def test_continuous_load_targets_only_selected_worker(tmp_path: Path) -> None:
    async def run_test() -> None:
        stop_event = asyncio.Event()
        runtime = FakeQwenLoadRuntime(stop_event)

        await run_load_process(
            mode=QwenLoadMode.CONTINUOUS_ACTIVE,
            active_workers=ActiveQwenWorkers.CONVERSATION,
            python_path=Path("vllm-python"),
            ready_file=tmp_path / "active-ready.json",
            runtime_factory=lambda python_path: runtime,
            stop_event=stop_event,
        )

        assert len(runtime.conversation.requests) == 1
        assert not runtime.search.requests
        assert runtime.closed

    asyncio.run(run_test())
