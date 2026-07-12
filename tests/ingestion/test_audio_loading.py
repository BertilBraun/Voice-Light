from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio import load_audio
from app.storage.local import LocalStorageBackend


def test_load_audio_decodes_metadata_and_samples(tmp_path: Path) -> None:
    audio_path = tmp_path / "track.wav"
    with wave.open(str(audio_path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8_000)
        stereo_frames = np.tile(np.array([1_000, -1_000], dtype="<i2"), (4_000, 1))
        writer.writeframes(stereo_frames.tobytes())

    audio = load_audio(LocalStorageBackend(), audio_path.as_posix())

    assert audio.metadata.duration_seconds == 0.5
    assert audio.metadata.sample_rate == 8_000
    assert audio.metadata.channels == 2
    assert audio.metadata.sample_count == 4_000
    assert audio.samples.shape == (4_000, 2)
    assert audio.samples[0] == pytest.approx(np.array([1_000 / 32_768, -1_000 / 32_768]))
