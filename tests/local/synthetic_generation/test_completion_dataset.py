from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.local.synthetic_generation.completion_dataset import (
    CompletionBoundaryKind,
    PromptLanguage,
    SyntheticSpeechPrompt,
    TtsGenerationProvenance,
    TtsProvider,
    analyze_generated_samples,
    completion_window_plans,
)


def test_analysis_labels_long_internal_silence_as_hold_and_trims_tail() -> None:
    sample_rate_hz = 16_000
    samples = np.concatenate(
        (
            _tone(1.0, sample_rate_hz),
            np.zeros(round(0.7 * sample_rate_hz), dtype=np.float32),
            _tone(1.2, sample_rate_hz),
            np.zeros(round(1.1 * sample_rate_hz), dtype=np.float32),
        )
    )

    trimmed, annotation = analyze_generated_samples(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        prompt=_prompt(),
        audio_path=Path("audio/example.wav"),
        provenance=_provenance(),
    )

    assert [boundary.kind for boundary in annotation.boundaries] == [
        CompletionBoundaryKind.HOLD,
        CompletionBoundaryKind.END_OF_TURN,
    ]
    assert annotation.boundaries[0].time_seconds == 1.0
    assert annotation.boundaries[0].silence_duration_seconds == pytest.approx(0.7)
    assert annotation.trimmed_duration_seconds == 2.9
    assert annotation.trimmed_trailing_seconds == 1.1
    assert trimmed.size == round(2.9 * sample_rate_hz)


def test_window_plans_move_each_boundary_and_include_all_visible_labels() -> None:
    sample_rate_hz = 16_000
    samples = np.concatenate(
        (
            _tone(8.0, sample_rate_hz),
            np.zeros(round(0.6 * sample_rate_hz), dtype=np.float32),
            _tone(14.0, sample_rate_hz),
        )
    )
    _, annotation = analyze_generated_samples(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        prompt=_prompt(),
        audio_path=Path("audio/example.wav"),
        provenance=_provenance(),
    )

    windows = completion_window_plans(annotation, seed=9)

    assert len(windows) == 2
    assert windows[0].anchor_kind is CompletionBoundaryKind.HOLD
    assert windows[1].anchor_kind is CompletionBoundaryKind.END_OF_TURN
    assert all(window.duration_seconds == 20.0 for window in windows)
    assert all(4.0 <= window.boundaries[0].time_seconds < 20.0 for window in windows)


def test_analysis_keeps_quiet_speech_inside_the_turn() -> None:
    sample_rate_hz = 16_000
    samples = np.concatenate(
        (
            _tone(1.0, sample_rate_hz),
            _tone(0.7, sample_rate_hz) * 0.02,
            _tone(1.0, sample_rate_hz),
            np.zeros(round(0.8 * sample_rate_hz), dtype=np.float32),
        )
    )

    _, annotation = analyze_generated_samples(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        prompt=_prompt(),
        audio_path=Path("audio/example.wav"),
        provenance=_provenance(),
    )

    assert [boundary.kind for boundary in annotation.boundaries] == [
        CompletionBoundaryKind.END_OF_TURN
    ]
    assert annotation.trimmed_duration_seconds == 2.7


def _tone(duration_seconds: float, sample_rate_hz: int) -> np.ndarray:
    times = np.arange(round(duration_seconds * sample_rate_hz)) / sample_rate_hz
    return (0.2 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)


def _prompt() -> SyntheticSpeechPrompt:
    return SyntheticSpeechPrompt(
        prompt_id="example_prompt",
        language=PromptLanguage.ENGLISH,
        text=(
            "I reviewed the schedule carefully before calling the team, and after comparing "
            "the available trains with the meeting times, I realized we should leave much "
            "earlier than planned. There is still enough time for breakfast near the station, "
            "but we should confirm the platform before everyone arrives tomorrow morning."
        ),
        voice_instruction="Warm adult voice, natural conversational pace, two thoughtful pauses.",
        topic="travel planning",
        seed=3,
    )


def _provenance() -> TtsGenerationProvenance:
    return TtsGenerationProvenance(
        provider=TtsProvider.QWEN3_VOICE_DESIGN,
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        model_revision="test-revision",
        runtime_version="0.1.1",
        model_license="Apache-2.0",
        generation_seconds=1.0,
        real_time_factor=0.5,
        batch_size=2,
    )
