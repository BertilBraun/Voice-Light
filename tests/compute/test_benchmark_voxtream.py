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
from deployment.compute.benchmark_voxtream import (
    BenchmarkPhase,
    VoxtreamBenchmarkCase,
    benchmark_case,
    normalized_variant,
    parse_load_command,
    percentile,
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


def test_percentile_interpolates_small_benchmark_samples() -> None:
    assert percentile((10.0, 20.0, 30.0), 0.5) == 20.0
    assert percentile((10.0, 20.0, 30.0), 0.9) == pytest.approx(28.0)
