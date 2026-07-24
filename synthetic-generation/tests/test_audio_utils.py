from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio_utils import audio_duration_seconds, trim_to_speech


def test_trim_to_speech_removes_leading_and_trailing_silence(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    sample_rate = 16_000
    silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    times = np.arange(int(sample_rate * 0.40), dtype=np.float32) / float(sample_rate)
    tone = 0.2 * np.sin(2.0 * np.pi * 440.0 * times)
    samples = np.concatenate([silence, tone, silence])
    sf.write(audio_path, samples, sample_rate, format="WAV", subtype="PCM_16")

    speech_bounds = trim_to_speech(audio_path, padding_seconds=0.03)

    assert 0.20 <= speech_bounds.onset_seconds <= 0.25
    assert 0.65 <= speech_bounds.offset_seconds <= 0.70
    assert audio_duration_seconds(audio_path) < speech_bounds.original_duration_seconds
    assert 0.48 <= speech_bounds.duration_seconds <= 0.50
