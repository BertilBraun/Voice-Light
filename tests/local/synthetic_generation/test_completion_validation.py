from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from app.local.synthetic_generation.completion_dataset import (
    QwenVoiceDesignProvenance,
    SyntheticEnglishSpeechPrompt,
    analyze_generated_samples,
)
from app.local.synthetic_generation.validate_completion_corpus import (
    CompletionValidationIssueCode,
    validate_completion_corpus,
)


def test_validation_reports_short_audio_without_a_hold(tmp_path: Path) -> None:
    sample_rate_hz = 16_000
    times = np.arange(sample_rate_hz * 3) / sample_rate_hz
    samples = (0.2 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    audio_path = tmp_path / "audio" / "example.wav"
    audio_path.parent.mkdir()
    _, annotation = analyze_generated_samples(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        utterance_id="example",
        prompt=_prompt(),
        audio_path=Path("audio/example.wav"),
        provenance=QwenVoiceDesignProvenance(
            model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            model_revision="revision",
            runtime_version="0.1.1",
            model_license="Apache-2.0",
            generation_seconds=1.0,
            real_time_factor=1.0 / 3.0,
            batch_size=1,
        ),
    )
    sf.write(audio_path, samples, sample_rate_hz, subtype="PCM_16")
    (tmp_path / "utterances.jsonl").write_text(
        annotation.model_dump_json() + "\n",
        encoding="utf-8",
    )

    report = validate_completion_corpus(tmp_path)

    assert report.utterance_count == 1
    assert report.valid_utterance_count == 0
    assert {issue.code for issue in report.issues} == {
        CompletionValidationIssueCode.TOO_SHORT,
        CompletionValidationIssueCode.ELEVATED_NOISE_FLOOR,
        CompletionValidationIssueCode.NO_HOLD,
    }


def _prompt() -> SyntheticEnglishSpeechPrompt:
    return SyntheticEnglishSpeechPrompt(
        prompt_id="example_prompt",
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
