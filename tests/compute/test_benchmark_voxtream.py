from __future__ import annotations

import asyncio
import hashlib
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.compute.voice.interfaces import (
    SpeechSynthesisSession,
    SynthesisEvent,
    SynthesisWord,
    SynthesizedAudioChunk,
    VoxtreamSynthesisFirstAudioMetrics,
)
from app.compute.voice.voxtream_speaking_rate import FinalPhraseSlowdown
from deployment.compute.benchmark_voxtream import (
    BenchmarkPhase,
    ManagedLoadProcess,
    VoxtreamBenchmarkCase,
    VoxtreamBenchmarkConfiguration,
    absolute_path_preserving_symlinks,
    benchmark_case,
    create_synthesizer,
    normalized_variant,
    parse_load_command,
    percentile,
    run_benchmark,
    synthesis_words,
)


class FakeSpeechSynthesisSession:
    def __init__(self, pcm_bytes: bytes) -> None:
        self.pcm_bytes = pcm_bytes
        self.words: list[SynthesisWord] = []
        self.input_finished = False
        self.cancelled = False

    async def add_word(self, word: SynthesisWord) -> None:
        self.words.append(word)

    async def finish_input(self) -> None:
        self.input_finished = True

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        if not self.input_finished:
            raise AssertionError("The fake input must be finished before streaming.")
        yield VoxtreamSynthesisFirstAudioMetrics(
            first_word_to_audio_seconds=0.125,
            prompt_preparation_seconds=0.025,
            first_frame_generation_seconds=0.100,
        )
        midpoint = len(self.pcm_bytes) // 2
        yield SynthesizedAudioChunk(
            pcm_bytes=self.pcm_bytes[:midpoint],
            start_sample=0,
        )
        yield SynthesizedAudioChunk(
            pcm_bytes=self.pcm_bytes[midpoint:],
            start_sample=midpoint // 2,
        )

    async def cancel(self) -> None:
        self.cancelled = True


class FakeSpeechSynthesizer:
    def __init__(self, pcm_bytes: bytes, sample_rate: int) -> None:
        self.pcm_bytes = pcm_bytes
        self._sample_rate = sample_rate
        self.sessions: list[FakeSpeechSynthesisSession] = []

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start_session(self) -> SpeechSynthesisSession:
        session = FakeSpeechSynthesisSession(self.pcm_bytes)
        self.sessions.append(session)
        return session

    def close(self) -> None:
        pass


def test_benchmark_case_writes_auditable_wav_files_and_warm_summary(
    tmp_path: Path,
) -> None:
    pcm_bytes = b"\x00\x00\x01\x00\x02\x00\x03\x00"
    synthesizer = FakeSpeechSynthesizer(pcm_bytes=pcm_bytes, sample_rate=16_000)
    result = asyncio.run(
        benchmark_case(
            synthesizer=synthesizer,
            benchmark_case=VoxtreamBenchmarkCase(
                name="short",
                text="Hello there.",
            ),
            runs=3,
            output_directory=tmp_path,
        )
    )

    assert result.first_measured.phase is BenchmarkPhase.FIRST_MEASURED
    assert result.warm_summary is not None
    assert len(result.trials) == 3
    assert all(trial.sample_count == 4 for trial in result.trials)
    assert all(trial.worker_first_word_to_audio_ms == 125 for trial in result.trials)
    assert all(trial.prompt_preparation_ms == 25 for trial in result.trials)
    assert all(trial.first_frame_generation_ms == 100 for trial in result.trials)
    assert all(session.cancelled for session in synthesizer.sessions)

    first_wav = tmp_path / result.trials[0].wav_path
    assert result.trials[0].wav_sha256 == hashlib.sha256(first_wav.read_bytes()).hexdigest()
    with wave.open(str(first_wav), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.readframes(4) == pcm_bytes


def test_synthesis_words_preserve_text_offsets() -> None:
    assert synthesis_words("Hello,  world!") == (
        SynthesisWord(text="Hello,", text_start=0, text_end=6),
        SynthesisWord(text="world!", text_start=8, text_end=14),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Compiled + cached", "compiled-cached"),
        ("baseline", "baseline"),
        ("A/B.test_1", "a-b.test_1"),
    ],
)
def test_normalized_variant_produces_stable_directory_names(
    value: str,
    expected: str,
) -> None:
    assert normalized_variant(value) == expected


def test_load_command_is_a_shell_free_json_array() -> None:
    assert parse_load_command('["python", "-m", "load_generator"]') == (
        "python",
        "-m",
        "load_generator",
    )
    with pytest.raises(ValueError, match="JSON string array"):
        parse_load_command('{"command": "python"}')


def test_managed_load_rejects_stale_readiness_file(tmp_path: Path) -> None:
    ready_file = tmp_path / "ready.json"
    ready_file.write_text("stale", encoding="utf-8")
    load_process = ManagedLoadProcess(
        command=("python", "-m", "load_generator"),
        startup_seconds=0.0,
        ready_file=ready_file,
        ready_timeout_seconds=1.0,
    )

    with pytest.raises(ValueError, match="already exists"):
        load_process.start()


def test_percentile_interpolates_small_benchmark_samples() -> None:
    assert percentile((10.0, 20.0, 30.0), 0.5) == 20.0
    assert percentile((10.0, 20.0, 30.0), 0.9) == pytest.approx(28.0)


def test_create_synthesizer_preserves_virtual_environment_python_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python_path = tmp_path / "venv" / "bin" / "python"
    configuration = VoxtreamBenchmarkConfiguration(
        variant="test",
        condition="idle",
        runs_per_case=1,
        compile_model=False,
        cache_prompt_in_memory=False,
        python_path=str(python_path),
        config_path=str(tmp_path / "generator.json"),
        config_sha256="0" * 64,
        speaking_rate_config_path=None,
        speaking_rate_config_sha256=None,
        prompt_audio_path=str(tmp_path / "prompt.wav"),
        prompt_audio_sha256="1" * 64,
        final_slowdown_syllables_per_second=None,
        final_slowdown_word_count=4,
        load_command=None,
        load_startup_seconds=0.0,
        load_ready_file=None,
        load_ready_timeout_seconds=1.0,
    )
    received_python_paths: list[Path] = []

    def create_voxtream(
        *,
        python_path: Path,
        config_path: Path,
        prompt_audio_path: Path,
        compile_model: bool,
        cache_prompt_in_memory: bool,
        speaking_rate_config_path: Path | None,
        final_phrase_slowdown: FinalPhraseSlowdown | None,
    ) -> FakeSpeechSynthesizer:
        del (
            config_path,
            prompt_audio_path,
            compile_model,
            cache_prompt_in_memory,
            speaking_rate_config_path,
            final_phrase_slowdown,
        )
        received_python_paths.append(python_path)
        return FakeSpeechSynthesizer(pcm_bytes=b"\x00\x00", sample_rate=16_000)

    monkeypatch.setattr(
        "deployment.compute.benchmark_voxtream.VoxtreamSpeechSynthesizer",
        create_voxtream,
    )

    create_synthesizer(configuration)

    assert received_python_paths == [python_path]


def test_absolute_python_path_does_not_dereference_virtual_environment_symlink(
    tmp_path: Path,
) -> None:
    interpreter = tmp_path / "python-base"
    interpreter.touch()
    virtual_environment_python = tmp_path / "venv" / "bin" / "python"
    virtual_environment_python.parent.mkdir(parents=True)
    try:
        virtual_environment_python.symlink_to(interpreter)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this platform.")

    assert absolute_path_preserving_symlinks(virtual_environment_python) == (
        virtual_environment_python.absolute()
    )
    assert absolute_path_preserving_symlinks(virtual_environment_python) != (
        virtual_environment_python.resolve()
    )


def test_run_benchmark_executes_cases_without_shadowing_case_function(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    synthesizer = FakeSpeechSynthesizer(
        pcm_bytes=b"\x00\x00\x01\x00\x02\x00\x03\x00",
        sample_rate=16_000,
    )
    configuration = VoxtreamBenchmarkConfiguration(
        variant="test",
        condition="idle",
        runs_per_case=1,
        compile_model=False,
        cache_prompt_in_memory=False,
        python_path=str(tmp_path / "python"),
        config_path=str(tmp_path / "generator.json"),
        config_sha256="0" * 64,
        speaking_rate_config_path=None,
        speaking_rate_config_sha256=None,
        prompt_audio_path=str(tmp_path / "prompt.wav"),
        prompt_audio_sha256="1" * 64,
        final_slowdown_syllables_per_second=None,
        final_slowdown_word_count=4,
        load_command=None,
        load_startup_seconds=0.0,
        load_ready_file=None,
        load_ready_timeout_seconds=1.0,
    )
    monkeypatch.setattr(
        "deployment.compute.benchmark_voxtream.create_synthesizer",
        lambda benchmark_configuration: synthesizer,
    )

    report = asyncio.run(
        run_benchmark(
            configuration=configuration,
            cases=(VoxtreamBenchmarkCase(name="short", text="Hello."),),
            output_directory=tmp_path,
        )
    )

    assert len(report.results) == 1
    assert report.results[0].case.name == "short"
