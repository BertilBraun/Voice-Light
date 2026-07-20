from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.shared.audio.wav import ANALYSIS_AUDIO_MAX_DURATION_SECONDS
from app.shared.audio.waveform import (
    capped_waveform_envelope,
    full_waveform_envelope,
    windowed_waveform_envelope,
)


def test_full_waveform_envelope_covers_complete_file(tmp_path: Path) -> None:
    wave_path = tmp_path / "complete.wav"
    samples = np.array([-32768, -16000, 0, 8000, 16000, 32767, -4000, 4000], dtype="<i2")
    with wave.open(str(wave_path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(4)
        wave_writer.writeframes(samples.tobytes())

    envelope = full_waveform_envelope(wave_path=wave_path, point_count=4)

    assert envelope.duration_seconds == pytest.approx(2.0)
    assert envelope.sample_rate == 4
    assert len(envelope.points) == 4
    assert envelope.points[0].minimum_amplitude == pytest.approx(-1.0)
    assert envelope.points[2].maximum_amplitude == pytest.approx(32767 / 32768)


def test_full_waveform_envelope_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Audio file does not exist"):
        full_waveform_envelope(wave_path=tmp_path / "missing.wav", point_count=100)


def test_capped_waveform_envelope_covers_analysis_window(tmp_path: Path) -> None:
    wave_path = tmp_path / "long.wav"
    sample_rate = 2
    duration_seconds = int(ANALYSIS_AUDIO_MAX_DURATION_SECONDS) + 20
    samples = np.arange(sample_rate * duration_seconds, dtype="<i2")
    with wave.open(str(wave_path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(samples.tobytes())

    envelope = capped_waveform_envelope(wave_path=wave_path, point_count=100)

    assert envelope.duration_seconds == ANALYSIS_AUDIO_MAX_DURATION_SECONDS
    assert len(envelope.points) == 100


def test_windowed_waveform_envelope_seeks_to_requested_context(tmp_path: Path) -> None:
    wave_path = tmp_path / "recording.wav"
    samples = np.concatenate(
        (
            np.full(100, -1000, dtype="<i2"),
            np.full(100, 2000, dtype="<i2"),
            np.full(100, -3000, dtype="<i2"),
        )
    )
    with wave.open(str(wave_path), "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(100)
        wave_writer.writeframes(samples.tobytes())

    envelope = windowed_waveform_envelope(
        wave_path=wave_path,
        point_count=2,
        start_seconds=1.0,
        duration_seconds=1.0,
    )

    assert envelope.duration_seconds == pytest.approx(1.0)
    assert len(envelope.points) == 2
    assert all(point.minimum_amplitude > 0.0 for point in envelope.points)
