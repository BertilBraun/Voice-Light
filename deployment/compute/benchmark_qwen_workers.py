from __future__ import annotations

import argparse
import math
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from app.compute.voice.llm_worker_protocol import (
    GenerateTextLlmCommand,
    LlmAssistantMessage,
    LlmCancelledEvent,
    LlmEndEvent,
    LlmSpokenTextDeltaEvent,
    LlmToolCallEvent,
    LlmToolCallFailureEvent,
    LlmToolCallStartedEvent,
    LlmUserMessage,
    LlmWorkerCommand,
    LlmWorkerErrorEvent,
    StartLlmCommand,
)
from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_ADAPTER_NAME,
    LANGUAGE_MODEL_ADAPTER_REVISION,
    LANGUAGE_MODEL_GPU_MEMORY_UTILIZATION,
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
    QWEN_MAXIMUM_MODEL_LENGTH,
    SEARCH_SUMMARIZER_GPU_MEMORY_UTILIZATION,
    SEARCH_SUMMARIZER_MODEL_NAME,
    SEARCH_SUMMARIZER_MODEL_REVISION,
)
from app.compute.voice.models import (
    QWEN_PYTHON_PATH,
    QwenWorkerConfiguration,
    QwenWorkerProcess,
)
from app.compute.voice.qwen_config import QwenAdapterConfiguration, QwenModelConfiguration
from app.compute.voice.search import (
    MAXIMUM_SEARCH_SUMMARY_TOKENS,
    SEARCH_SUMMARIZER_SYSTEM_PROMPT,
    SearchResult,
    render_search_summary_prompt,
)
from app.compute.voice.tools import runtime_tool_specifications
from app.shared.base_model import FrozenBaseModel


class WorkerBenchmarkTrial(FrozenBaseModel):
    name: str
    prompt_index: int = Field(gt=0)
    first_text_ms: float = Field(ge=0.0)
    total_ms: float = Field(gt=0.0)
    output_tokens: int = Field(gt=0)
    tokens_per_second: float = Field(gt=0.0)
    output_text: str


class WorkerBenchmarkResult(FrozenBaseModel):
    component_name: str
    model_name: str
    model_revision: str
    adapter_name: str | None
    adapter_revision: str | None
    startup_seconds: float = Field(gt=0.0)
    gpu_memory_mb: int = Field(ge=0)
    first_text_p50_ms: float = Field(ge=0.0)
    total_p50_ms: float = Field(gt=0.0)
    tokens_per_second_p50: float = Field(gt=0.0)
    trials: tuple[WorkerBenchmarkTrial, ...]


class QwenWorkerBenchmarkReport(FrozenBaseModel):
    runtime_label: str
    conversation: WorkerBenchmarkResult
    search_summarizer: WorkerBenchmarkResult


def run_benchmark(runtime_label: str) -> QwenWorkerBenchmarkReport:
    conversation = benchmark_conversation_worker()
    search_summarizer = benchmark_search_worker()
    return QwenWorkerBenchmarkReport(
        runtime_label=runtime_label,
        conversation=conversation,
        search_summarizer=search_summarizer,
    )


def benchmark_conversation_worker() -> WorkerBenchmarkResult:
    configuration = QwenWorkerConfiguration(
        model=QwenModelConfiguration(
            model_name=LANGUAGE_MODEL_NAME,
            model_revision=LANGUAGE_MODEL_REVISION,
            adapter=QwenAdapterConfiguration(
                repository_id=LANGUAGE_MODEL_ADAPTER_NAME,
                revision=LANGUAGE_MODEL_ADAPTER_REVISION,
            ),
            gpu_memory_utilization=LANGUAGE_MODEL_GPU_MEMORY_UTILIZATION,
            maximum_model_length=QWEN_MAXIMUM_MODEL_LENGTH,
        ),
        component_name="Qwen language model",
    )
    worker, startup_seconds = start_worker(configuration)
    try:
        gpu_memory_mb = active_gpu_memory_mb()
        messages: list[LlmUserMessage | LlmAssistantMessage] = []
        trials: list[WorkerBenchmarkTrial] = []
        user_prompts = (
            "Rewrite this to sound warmer: The deployment is delayed.",
            "Make it shorter and keep the meaning.",
            "Now make it sound natural when spoken aloud.",
        )
        for prompt_index, user_prompt in enumerate(user_prompts, start=1):
            messages.append(LlmUserMessage(content=user_prompt))
            result = run_trial(
                worker=worker,
                command=conversation_command(prompt_index, tuple(messages)),
                name=f"conversation_turn_{prompt_index}",
                prompt_index=prompt_index,
            )
            messages.append(LlmAssistantMessage(content=result.output_text))
            trials.append(result)
        return build_worker_result(
            configuration=configuration,
            startup_seconds=startup_seconds,
            gpu_memory_mb=gpu_memory_mb,
            trials=tuple(trials),
        )
    finally:
        worker.close()


def conversation_command(
    invocation_id: int,
    messages: tuple[LlmUserMessage | LlmAssistantMessage, ...],
) -> LlmWorkerCommand:
    return StartLlmCommand(
        invocation_id=invocation_id,
        assistant_generation_id=1,
        messages=messages,
        tools=runtime_tool_specifications(),
    )


def benchmark_search_worker() -> WorkerBenchmarkResult:
    configuration = QwenWorkerConfiguration(
        model=QwenModelConfiguration(
            model_name=SEARCH_SUMMARIZER_MODEL_NAME,
            model_revision=SEARCH_SUMMARIZER_MODEL_REVISION,
            adapter=None,
            gpu_memory_utilization=SEARCH_SUMMARIZER_GPU_MEMORY_UTILIZATION,
            maximum_model_length=QWEN_MAXIMUM_MODEL_LENGTH,
        ),
        component_name="Qwen search summarizer",
    )
    worker, startup_seconds = start_worker(configuration)
    try:
        gpu_memory_mb = active_gpu_memory_mb()
        trials = tuple(
            run_trial(
                worker=worker,
                command=GenerateTextLlmCommand(
                    invocation_id=prompt_index,
                    system_prompt=SEARCH_SUMMARIZER_SYSTEM_PROMPT,
                    user_prompt=render_search_summary_prompt(query, results),
                    max_new_tokens=MAXIMUM_SEARCH_SUMMARY_TOKENS,
                ),
                name=f"search_summary_{prompt_index}",
                prompt_index=prompt_index,
            )
            for prompt_index, (query, results) in enumerate(search_cases(), start=1)
        )
        return build_worker_result(
            configuration=configuration,
            startup_seconds=startup_seconds,
            gpu_memory_mb=gpu_memory_mb,
            trials=trials,
        )
    finally:
        worker.close()


def start_worker(
    configuration: QwenWorkerConfiguration,
) -> tuple[QwenWorkerProcess, float]:
    started_at = time.perf_counter()
    worker = QwenWorkerProcess(QWEN_PYTHON_PATH, configuration)
    return worker, time.perf_counter() - started_at


def run_trial(
    worker: QwenWorkerProcess,
    command: LlmWorkerCommand,
    name: str,
    prompt_index: int,
) -> WorkerBenchmarkTrial:
    started_at = time.perf_counter()
    first_text_at: float | None = None
    output_parts: list[str] = []
    worker.send(command)
    while True:
        event = worker.read_event()
        match event:
            case LlmSpokenTextDeltaEvent(text=text):
                if text and first_text_at is None:
                    first_text_at = time.perf_counter()
                output_parts.append(text)
            case LlmEndEvent(cumulative_token_count=output_tokens):
                completed_at = time.perf_counter()
                break
            case LlmToolCallStartedEvent() | LlmToolCallEvent() | LlmToolCallFailureEvent():
                raise RuntimeError(f"{name} unexpectedly emitted a tool event.")
            case LlmCancelledEvent():
                raise RuntimeError(f"{name} was unexpectedly cancelled.")
            case LlmWorkerErrorEvent(message=message):
                raise RuntimeError(f"{name} failed: {message}")
            case _:
                raise RuntimeError(f"{name} returned an unexpected worker event.")
    if first_text_at is None:
        raise RuntimeError(f"{name} produced no text.")
    if output_tokens <= 0:
        raise RuntimeError(f"{name} produced no output tokens.")
    total_seconds = completed_at - started_at
    return WorkerBenchmarkTrial(
        name=name,
        prompt_index=prompt_index,
        first_text_ms=(first_text_at - started_at) * 1_000,
        total_ms=total_seconds * 1_000,
        output_tokens=output_tokens,
        tokens_per_second=output_tokens / total_seconds,
        output_text="".join(output_parts).strip(),
    )


def build_worker_result(
    configuration: QwenWorkerConfiguration,
    startup_seconds: float,
    gpu_memory_mb: int,
    trials: tuple[WorkerBenchmarkTrial, ...],
) -> WorkerBenchmarkResult:
    adapter = configuration.model.adapter
    return WorkerBenchmarkResult(
        component_name=configuration.component_name,
        model_name=configuration.model.model_name,
        model_revision=configuration.model.model_revision,
        adapter_name=None if adapter is None else adapter.repository_id,
        adapter_revision=None if adapter is None else adapter.revision,
        startup_seconds=startup_seconds,
        gpu_memory_mb=gpu_memory_mb,
        first_text_p50_ms=percentile(
            tuple(trial.first_text_ms for trial in trials),
            0.50,
        ),
        total_p50_ms=percentile(
            tuple(trial.total_ms for trial in trials),
            0.50,
        ),
        tokens_per_second_p50=percentile(
            tuple(trial.tokens_per_second for trial in trials),
            0.50,
        ),
        trials=trials,
    )


def percentile(values: tuple[float, ...], quantile: float) -> float:
    if not values:
        raise ValueError("A percentile requires at least one value.")
    ordered_values = sorted(values)
    rank = max(math.ceil(quantile * len(ordered_values)), 1)
    return ordered_values[rank - 1]


def active_gpu_memory_mb() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory_values = tuple(
        int(line.split(",", maxsplit=1)[1].strip())
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    if not memory_values:
        raise RuntimeError("nvidia-smi did not report an active Qwen inference process.")
    return sum(memory_values)


def search_cases() -> tuple[tuple[str, tuple[SearchResult, ...]], ...]:
    results = (
        SearchResult(
            title="Release notes",
            url="https://example.test/releases",
            snippet=(
                "Version 4.2 was released on Tuesday. It adds faster synchronization and fixes "
                "two connection-recovery issues."
            ),
        ),
        SearchResult(
            title="Service status",
            url="https://status.example.test/",
            snippet="All services are operational. No incidents have been reported today.",
        ),
        SearchResult(
            title="Migration guide",
            url="https://example.test/migration",
            snippet="Existing installations can upgrade directly from version 4.1 to version 4.2.",
        ),
    )
    return (
        ("What changed in version 4.2?", results),
        ("Are the services currently operational?", results),
        ("Can version 4.1 upgrade directly to 4.2?", results),
    )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark production Qwen worker processes.")
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    options = parse_arguments(arguments)
    report = run_benchmark(options.runtime_label)
    serialized_report = report.model_dump_json(indent=2)
    print(serialized_report)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(serialized_report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
