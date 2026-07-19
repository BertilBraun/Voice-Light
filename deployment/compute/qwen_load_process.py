from __future__ import annotations

import argparse
import asyncio
import os
import signal
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from app.compute.voice.interfaces import TextGenerationRequest, TextGenerator
from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_SYSTEM_PROMPT,
    SEARCH_SUMMARIZER_MODEL_NAME,
    SEARCH_SUMMARIZER_MODEL_REVISION,
)
from app.compute.voice.models import QWEN_PYTHON_PATH, VllmLanguageModel, VllmTextGenerator
from app.compute.voice.qwen_config import language_model_configuration_from_environment
from app.compute.voice.search import SEARCH_SUMMARIZER_SYSTEM_PROMPT
from app.shared.base_model import FrozenBaseModel

CONVERSATION_LOAD_REQUEST = TextGenerationRequest(
    system_prompt=LANGUAGE_MODEL_SYSTEM_PROMPT,
    user_prompt=(
        "Explain, in about one hundred spoken words, how a voice assistant can stay responsive "
        "while it processes audio, generates text, and synthesizes speech."
    ),
    max_new_tokens=160,
)
SEARCH_LOAD_REQUEST = TextGenerationRequest(
    system_prompt=SEARCH_SUMMARIZER_SYSTEM_PROMPT,
    user_prompt=(
        "Search query: What changed in version 4.2?\n\n"
        "Result 1 title: Release notes\n"
        "Result 1 URL: https://example.test/releases\n"
        "Result 1 snippet: Version 4.2 adds faster synchronization and fixes two "
        "recovery issues.\n\n"
        "Result 2 title: Migration guide\n"
        "Result 2 URL: https://example.test/migration\n"
        "Result 2 snippet: Existing installations can upgrade directly from version 4.1."
    ),
    max_new_tokens=160,
)


class QwenLoadMode(StrEnum):
    RESIDENT_IDLE = "resident-idle"
    CONTINUOUS_ACTIVE = "continuous-active"


class ActiveQwenWorkers(StrEnum):
    CONVERSATION = "conversation"
    SEARCH = "search"
    BOTH = "both"


class QwenLoadReadySignal(FrozenBaseModel):
    process_id: int
    mode: QwenLoadMode
    active_workers: ActiveQwenWorkers
    conversation_model_name: str
    conversation_model_revision: str
    search_model_name: str
    search_model_revision: str


class QwenLoadRuntimeProtocol(Protocol):
    conversation: TextGenerator
    search: TextGenerator

    def close(self) -> None: ...


class QwenLoadRuntime:
    def __init__(self, python_path: Path) -> None:
        self.conversation = VllmLanguageModel(python_path)
        try:
            self.search = VllmTextGenerator(python_path)
        except BaseException:
            self.conversation.close()
            raise

    def close(self) -> None:
        self.conversation.close()
        self.search.close()


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Keep both production Qwen vLLM workers resident or continuously active."
    )
    parser.add_argument(
        "--mode",
        type=QwenLoadMode,
        choices=tuple(QwenLoadMode),
        required=True,
    )
    parser.add_argument(
        "--active-workers",
        type=ActiveQwenWorkers,
        choices=tuple(ActiveQwenWorkers),
        default=ActiveQwenWorkers.CONVERSATION,
    )
    parser.add_argument("--python", type=Path, default=QWEN_PYTHON_PATH)
    parser.add_argument("--ready-file", type=Path, required=True)
    options = parser.parse_args(arguments)
    asyncio.run(
        run_load_process(
            mode=options.mode,
            active_workers=options.active_workers,
            python_path=options.python,
            ready_file=options.ready_file,
        )
    )


async def run_load_process(
    mode: QwenLoadMode,
    active_workers: ActiveQwenWorkers,
    python_path: Path,
    ready_file: Path,
    runtime_factory: Callable[[Path], QwenLoadRuntimeProtocol] = QwenLoadRuntime,
    stop_event: asyncio.Event | None = None,
) -> None:
    if ready_file.exists():
        raise ValueError(f"Qwen load readiness file already exists: {ready_file}")
    active_stop_event = stop_event
    if active_stop_event is None:
        active_stop_event = asyncio.Event()
        install_signal_handlers(active_stop_event)
    runtime = await asyncio.to_thread(runtime_factory, python_path)
    conversation_configuration = language_model_configuration_from_environment(os.environ)
    ready_signal = QwenLoadReadySignal(
        process_id=os.getpid(),
        mode=mode,
        active_workers=active_workers,
        conversation_model_name=conversation_configuration.model_name,
        conversation_model_revision=conversation_configuration.model_revision,
        search_model_name=SEARCH_SUMMARIZER_MODEL_NAME,
        search_model_revision=SEARCH_SUMMARIZER_MODEL_REVISION,
    )
    try:
        write_ready_signal(ready_file, ready_signal)
        match mode:
            case QwenLoadMode.RESIDENT_IDLE:
                await active_stop_event.wait()
            case QwenLoadMode.CONTINUOUS_ACTIVE:
                await run_continuous_load(
                    runtime=runtime,
                    active_workers=active_workers,
                    stop_event=active_stop_event,
                )
    finally:
        ready_file.unlink(missing_ok=True)
        await asyncio.to_thread(runtime.close)


async def run_continuous_load(
    runtime: QwenLoadRuntimeProtocol,
    active_workers: ActiveQwenWorkers,
    stop_event: asyncio.Event,
) -> None:
    match active_workers:
        case ActiveQwenWorkers.CONVERSATION:
            await generate_until_stopped(
                generator=runtime.conversation,
                request=CONVERSATION_LOAD_REQUEST,
                stop_event=stop_event,
            )
        case ActiveQwenWorkers.SEARCH:
            await generate_until_stopped(
                generator=runtime.search,
                request=SEARCH_LOAD_REQUEST,
                stop_event=stop_event,
            )
        case ActiveQwenWorkers.BOTH:
            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(
                    generate_until_stopped(
                        generator=runtime.conversation,
                        request=CONVERSATION_LOAD_REQUEST,
                        stop_event=stop_event,
                    )
                )
                task_group.create_task(
                    generate_until_stopped(
                        generator=runtime.search,
                        request=SEARCH_LOAD_REQUEST,
                        stop_event=stop_event,
                    )
                )


async def generate_until_stopped(
    generator: TextGenerator,
    request: TextGenerationRequest,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        await generator.generate_text(request)


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    event_loop = asyncio.get_running_loop()
    event_loop.add_signal_handler(signal.SIGINT, stop_event.set)
    event_loop.add_signal_handler(signal.SIGTERM, stop_event.set)


def write_ready_signal(path: Path, signal_value: QwenLoadReadySignal) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.writing")
    temporary_path.write_text(signal_value.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


if __name__ == "__main__":
    main()
