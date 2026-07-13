from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio.waveform import full_waveform_envelope


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
