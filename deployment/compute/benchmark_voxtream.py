from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import subprocess
import time
import wave
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from app.compute.voice.interfaces import (
    SpeechSynthesizer,
    SynthesisWord,
    SynthesizedAudioChunk,
    VoxtreamSynthesisFirstAudioMetrics,
)
from app.compute.voice.voxtream_speaking_rate import FinalPhraseSlowdown
from app.compute.voice.voxtream_tts import VoxtreamSpeechSynthesizer
from app.shared.base_model import FrozenBaseModel

DEFAULT_CASES = (
    (
        "short",
        "Hello, I'm ready to help.",
    ),
    (
        "conversational",
        "That makes sense. I'll check the details and explain what I find.",
    ),
    (
        "numbers",
        "The update finished in 2.7 seconds, and all 14 checks passed.",
    ),
)
PCM_SAMPLE_WIDTH_BYTES = 2


class BenchmarkCondition(StrEnum):
    IDLE = "idle"
    CONCURRENT_LLM = "concurrent-llm"


class BenchmarkPhase(StrEnum):
    FIRST_MEASURED = "first-measured"
    WARM = "warm"


class VoxtreamBenchmarkCase(FrozenBaseModel):
    name: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MetricSummary(FrozenBaseModel):
    mean_ms: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p90_ms: float = Field(ge=0.0)
    minimum_ms: float = Field(ge=0.0)
    maximum_ms: float = Field(ge=0.0)


class TrialMetricSummary(FrozenBaseModel):
    first_word_to_pcm: MetricSummary
    worker_first_word_to_audio: MetricSummary
    prompt_preparation: MetricSummary
    first_frame_generation: MetricSummary
    total: MetricSummary
    real_time_factor_mean: float = Field(gt=0.0)


class VoxtreamBenchmarkTrial(FrozenBaseModel):
    case_name: str
    run: int = Field(gt=0)
    phase: BenchmarkPhase
    first_word_to_pcm_ms: float = Field(ge=0.0)
    worker_first_word_to_audio_ms: float = Field(ge=0.0)
    prompt_preparation_ms: float = Field(ge=0.0)
    first_frame_generation_ms: float = Field(ge=0.0)
    total_ms: float = Field(gt=0.0)
    audio_duration_ms: float = Field(gt=0.0)
    real_time_factor: float = Field(gt=0.0)
    sample_count: int = Field(gt=0)
    wav_path: str
    wav_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VoxtreamCaseResult(FrozenBaseModel):
    case: VoxtreamBenchmarkCase
    first_measured: VoxtreamBenchmarkTrial
    warm_summary: TrialMetricSummary | None
    trials: tuple[VoxtreamBenchmarkTrial, ...]


class VoxtreamBenchmarkConfiguration(FrozenBaseModel):
    variant: str
    condition: BenchmarkCondition
    runs_per_case: int = Field(gt=0)
    compile_model: bool
    cache_prompt_in_memory: bool
    python_path: str
    config_path: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaking_rate_config_path: str | None
    speaking_rate_config_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_audio_path: str
    prompt_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_slowdown_syllables_per_second: float | None = Field(gt=0.0)
    final_slowdown_word_count: int = Field(gt=0)
    load_command: tuple[str, ...] | None
    load_startup_seconds: float = Field(ge=0.0)
    load_ready_file: str | None
    load_ready_timeout_seconds: float = Field(gt=0.0)
    worker_performs_startup_warmup: bool = True


class VoxtreamBenchmarkReport(FrozenBaseModel):
    configuration: VoxtreamBenchmarkConfiguration
    model_startup_seconds: float = Field(gt=0.0)
    sample_rate: int = Field(gt=0)
    output_directory: str
    first_measured_summary: TrialMetricSummary
    warm_summary: TrialMetricSummary | None
    results: tuple[VoxtreamCaseResult, ...]


class ManagedLoadProcess:
    def __init__(
        self,
        command: tuple[str, ...],
        startup_seconds: float,
        ready_file: Path | None,
        ready_timeout_seconds: float,
    ) -> None:
        if not command:
            raise ValueError("The concurrent load command cannot be empty.")
        self.command = command
        self.startup_seconds = startup_seconds
        self.ready_file = ready_file
        self.ready_timeout_seconds = ready_timeout_seconds
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.process is not None:
            raise AssertionError("The concurrent load process may only be started once.")
        if self.ready_file is not None and self.ready_file.exists():
            raise ValueError(f"The load readiness file already exists: {self.ready_file}")
        self.process = subprocess.Popen(self.command)
        try:
            if self.ready_file is not None:
                self._wait_until_ready()
                return
            self._wait_for_startup_delay()
        except BaseException:
            self.close()
            raise

    def _wait_until_ready(self) -> None:
        assert self.process is not None
        assert self.ready_file is not None
        deadline = time.monotonic() + self.ready_timeout_seconds
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Concurrent load command exited before readiness with code {return_code}."
                )
            if self.ready_file.is_file():
                return
            time.sleep(min(0.1, deadline - time.monotonic()))
        raise TimeoutError(
            f"Concurrent load command did not create {self.ready_file} within "
            f"{self.ready_timeout_seconds:.1f} seconds."
        )

    def _wait_for_startup_delay(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.startup_seconds
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Concurrent load command exited during startup with code {return_code}."
                )
            time.sleep(min(0.1, deadline - time.monotonic()))
        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Concurrent load command exited before the benchmark with code {return_code}."
            )

    def close(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10.0)


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the production VoXtream worker and save auditable WAV samples."
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--speaking-rate-config", type=Path)
    parser.add_argument("--prompt-audio", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--condition",
        type=BenchmarkCondition,
        choices=tuple(BenchmarkCondition),
        default=BenchmarkCondition.IDLE,
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="compile_model",
    )
    parser.add_argument(
        "--cache-prompt-in-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--final-slowdown-syllables-per-second", type=float)
    parser.add_argument("--final-slowdown-word-count", type=int, default=4)
    parser.add_argument(
        "--load-command-json",
        help=(
            "Optional JSON string array for a managed load process. It is started before VoXtream "
            "so LLM memory reservations match production load order."
        ),
    )
    parser.add_argument("--load-startup-seconds", type=float, default=0.0)
    parser.add_argument(
        "--load-ready-file",
        type=Path,
        help="Wait for this file instead of relying on --load-startup-seconds.",
    )
    parser.add_argument("--load-ready-timeout-seconds", type=float, default=420.0)
    options = parser.parse_args(arguments)
    if options.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if options.load_startup_seconds < 0:
        raise ValueError("--load-startup-seconds cannot be negative.")
    if options.load_ready_timeout_seconds <= 0:
        raise ValueError("--load-ready-timeout-seconds must be positive.")
    if (options.speaking_rate_config is None) != (
        options.final_slowdown_syllables_per_second is None
    ):
        raise ValueError(
            "--speaking-rate-config and --final-slowdown-syllables-per-second "
            "must be provided together."
        )
    final_phrase_slowdown = (
        None
        if options.final_slowdown_syllables_per_second is None
        else FinalPhraseSlowdown(
            syllables_per_second=options.final_slowdown_syllables_per_second,
            word_count=options.final_slowdown_word_count,
        )
    )

    variant = normalized_variant(options.variant)
    output_directory = options.output_root / variant
    load_command = parse_load_command(options.load_command_json)
    if options.load_ready_file is not None and load_command is None:
        raise ValueError("--load-ready-file requires --load-command-json.")
    configuration = VoxtreamBenchmarkConfiguration(
        variant=variant,
        condition=options.condition,
        runs_per_case=options.runs,
        compile_model=options.compile_model,
        cache_prompt_in_memory=options.cache_prompt_in_memory,
        python_path=str(absolute_path_preserving_symlinks(options.python)),
        config_path=str(options.config.resolve()),
        config_sha256=sha256_file(options.config),
        speaking_rate_config_path=(
            None
            if options.speaking_rate_config is None
            else str(options.speaking_rate_config.resolve())
        ),
        speaking_rate_config_sha256=(
            None
            if options.speaking_rate_config is None
            else sha256_file(options.speaking_rate_config)
        ),
        prompt_audio_path=str(options.prompt_audio.resolve()),
        prompt_audio_sha256=sha256_file(options.prompt_audio),
        final_slowdown_syllables_per_second=(
            None if final_phrase_slowdown is None else final_phrase_slowdown.syllables_per_second
        ),
        final_slowdown_word_count=options.final_slowdown_word_count,
        load_command=load_command,
        load_startup_seconds=options.load_startup_seconds,
        load_ready_file=(
            None if options.load_ready_file is None else str(options.load_ready_file.resolve())
        ),
        load_ready_timeout_seconds=options.load_ready_timeout_seconds,
    )
    cases = tuple(VoxtreamBenchmarkCase(name=name, text=text) for name, text in DEFAULT_CASES)
    report = asyncio.run(
        run_benchmark(
            configuration=configuration,
            cases=cases,
            output_directory=output_directory,
        )
    )
    report_path = output_directory / "report.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))


async def run_benchmark(
    configuration: VoxtreamBenchmarkConfiguration,
    cases: tuple[VoxtreamBenchmarkCase, ...],
    output_directory: Path,
) -> VoxtreamBenchmarkReport:
    if not cases:
        raise ValueError("At least one VoXtream benchmark case is required.")
    output_directory.mkdir(parents=True, exist_ok=True)
    load_process = (
        None
        if configuration.load_command is None
        else ManagedLoadProcess(
            command=configuration.load_command,
            startup_seconds=configuration.load_startup_seconds,
            ready_file=(
                None
                if configuration.load_ready_file is None
                else Path(configuration.load_ready_file)
            ),
            ready_timeout_seconds=configuration.load_ready_timeout_seconds,
        )
    )
    if load_process is not None:
        await asyncio.to_thread(load_process.start)

    synthesizer: SpeechSynthesizer | None = None
    try:
        startup_started_at = time.perf_counter()
        synthesizer = await asyncio.to_thread(create_synthesizer, configuration)
        model_startup_seconds = time.perf_counter() - startup_started_at
        results = tuple(
            [
                await benchmark_case(
                    synthesizer=synthesizer,
                    benchmark_case=case,
                    runs=configuration.runs_per_case,
                    output_directory=output_directory,
                )
                for case in cases
            ]
        )
        first_measured_trials = tuple(result.first_measured for result in results)
        warm_trials = tuple(
            trial
            for result in results
            for trial in result.trials
            if trial.phase is BenchmarkPhase.WARM
        )
        return VoxtreamBenchmarkReport(
            configuration=configuration,
            model_startup_seconds=model_startup_seconds,
            sample_rate=synthesizer.sample_rate,
            output_directory=str(output_directory.resolve()),
            first_measured_summary=summarize_trials(first_measured_trials),
            warm_summary=summarize_trials(warm_trials) if warm_trials else None,
            results=results,
        )
    finally:
        if synthesizer is not None:
            await asyncio.to_thread(synthesizer.close)
        if load_process is not None:
            await asyncio.to_thread(load_process.close)


def create_synthesizer(
    configuration: VoxtreamBenchmarkConfiguration,
) -> VoxtreamSpeechSynthesizer:
    return VoxtreamSpeechSynthesizer(
        python_path=Path(configuration.python_path),
        config_path=Path(configuration.config_path),
        prompt_audio_path=Path(configuration.prompt_audio_path),
        compile_model=configuration.compile_model,
        cache_prompt_in_memory=configuration.cache_prompt_in_memory,
        speaking_rate_config_path=(
            None
            if configuration.speaking_rate_config_path is None
            else Path(configuration.speaking_rate_config_path)
        ),
        final_phrase_slowdown=(
            None
            if configuration.final_slowdown_syllables_per_second is None
            else FinalPhraseSlowdown(
                syllables_per_second=(configuration.final_slowdown_syllables_per_second),
                word_count=configuration.final_slowdown_word_count,
            )
        ),
    )


async def benchmark_case(
    synthesizer: SpeechSynthesizer,
    benchmark_case: VoxtreamBenchmarkCase,
    runs: int,
    output_directory: Path,
) -> VoxtreamCaseResult:
    words = synthesis_words(benchmark_case.text)
    trials = tuple(
        [
            await benchmark_trial(
                synthesizer=synthesizer,
                benchmark_case=benchmark_case,
                words=words,
                run=run,
                output_directory=output_directory,
            )
            for run in range(1, runs + 1)
        ]
    )
    warm_trials = trials[1:]
    return VoxtreamCaseResult(
        case=benchmark_case,
        first_measured=trials[0],
        warm_summary=summarize_trials(warm_trials) if warm_trials else None,
        trials=trials,
    )


async def benchmark_trial(
    synthesizer: SpeechSynthesizer,
    benchmark_case: VoxtreamBenchmarkCase,
    words: tuple[SynthesisWord, ...],
    run: int,
    output_directory: Path,
) -> VoxtreamBenchmarkTrial:
    session = synthesizer.start_session()
    first_word_sent_at = time.perf_counter()
    first_pcm_at: float | None = None
    first_audio_metrics: VoxtreamSynthesisFirstAudioMetrics | None = None
    pcm_parts: list[bytes] = []
    sample_count = 0
    try:
        for word in words:
            await session.add_word(word)
        await session.finish_input()
        async for event in session.stream_events():
            match event:
                case VoxtreamSynthesisFirstAudioMetrics():
                    first_audio_metrics = event
                case SynthesizedAudioChunk():
                    if event.start_sample != sample_count:
                        raise RuntimeError(
                            "VoXtream benchmark received non-contiguous audio: "
                            f"expected {sample_count}, received {event.start_sample}."
                        )
                    if first_pcm_at is None:
                        first_pcm_at = time.perf_counter()
                    pcm_parts.append(event.pcm_bytes)
                    sample_count += len(event.pcm_bytes) // PCM_SAMPLE_WIDTH_BYTES
        completed_at = time.perf_counter()
    finally:
        await session.cancel()

    if first_pcm_at is None or sample_count == 0:
        raise RuntimeError("VoXtream benchmark produced no audio.")
    if first_audio_metrics is None:
        raise RuntimeError("VoXtream benchmark produced no first-audio metrics.")
    pcm_bytes = b"".join(pcm_parts)
    wav_path = output_directory / f"{benchmark_case.name}-run-{run:02d}.wav"
    write_pcm16_wav(
        path=wav_path,
        pcm_bytes=pcm_bytes,
        sample_rate=synthesizer.sample_rate,
    )
    total_seconds = completed_at - first_word_sent_at
    audio_duration_seconds = sample_count / synthesizer.sample_rate
    return VoxtreamBenchmarkTrial(
        case_name=benchmark_case.name,
        run=run,
        phase=BenchmarkPhase.FIRST_MEASURED if run == 1 else BenchmarkPhase.WARM,
        first_word_to_pcm_ms=(first_pcm_at - first_word_sent_at) * 1_000,
        worker_first_word_to_audio_ms=(first_audio_metrics.first_word_to_audio_seconds * 1_000),
        prompt_preparation_ms=first_audio_metrics.prompt_preparation_seconds * 1_000,
        first_frame_generation_ms=first_audio_metrics.first_frame_generation_seconds * 1_000,
        total_ms=total_seconds * 1_000,
        audio_duration_ms=audio_duration_seconds * 1_000,
        real_time_factor=total_seconds / audio_duration_seconds,
        sample_count=sample_count,
        wav_path=wav_path.name,
        wav_sha256=sha256_file(wav_path),
    )


def synthesis_words(text: str) -> tuple[SynthesisWord, ...]:
    words = tuple(
        SynthesisWord(
            text=match.group(),
            text_start=match.start(),
            text_end=match.end(),
        )
        for match in re.finditer(r"\S+", text)
    )
    if not words:
        raise ValueError("VoXtream benchmark text must contain a non-whitespace word.")
    return words


def summarize_trials(
    trials: tuple[VoxtreamBenchmarkTrial, ...],
) -> TrialMetricSummary:
    if not trials:
        raise ValueError("Cannot summarize an empty sequence of VoXtream trials.")
    return TrialMetricSummary(
        first_word_to_pcm=summarize_metric(tuple(trial.first_word_to_pcm_ms for trial in trials)),
        worker_first_word_to_audio=summarize_metric(
            tuple(trial.worker_first_word_to_audio_ms for trial in trials)
        ),
        prompt_preparation=summarize_metric(tuple(trial.prompt_preparation_ms for trial in trials)),
        first_frame_generation=summarize_metric(
            tuple(trial.first_frame_generation_ms for trial in trials)
        ),
        total=summarize_metric(tuple(trial.total_ms for trial in trials)),
        real_time_factor_mean=statistics.fmean(trial.real_time_factor for trial in trials),
    )


def summarize_metric(values: tuple[float, ...]) -> MetricSummary:
    if not values:
        raise ValueError("Cannot summarize an empty metric.")
    sorted_values = tuple(sorted(values))
    return MetricSummary(
        mean_ms=statistics.fmean(sorted_values),
        p50_ms=percentile(sorted_values, 0.50),
        p90_ms=percentile(sorted_values, 0.90),
        minimum_ms=sorted_values[0],
        maximum_ms=sorted_values[-1],
    )


def percentile(sorted_values: tuple[float, ...], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile probability must be between zero and one.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    upper_weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - upper_weight)
        + sorted_values[upper_index] * upper_weight
    )


def write_pcm16_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    if len(pcm_bytes) % PCM_SAMPLE_WIDTH_BYTES != 0:
        raise ValueError("PCM16 audio must contain complete samples.")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_stream:
        while chunk := input_stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_load_command(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    decoded = json.loads(value)
    if (
        not isinstance(decoded, list)
        or not decoded
        or any(not isinstance(part, str) or not part for part in decoded)
    ):
        raise ValueError("--load-command-json must be a non-empty JSON string array.")
    return tuple(decoded)


def normalized_variant(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("--variant must contain at least one letter or number.")
    return normalized


def absolute_path_preserving_symlinks(path: Path) -> Path:
    return path.expanduser().absolute()


if __name__ == "__main__":
    main()
