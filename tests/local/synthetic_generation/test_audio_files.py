from pathlib import Path

import numpy as np
import pytest
import soundfile

from app.local.synthetic_generation.audio_files import (
    read_mono_pcm16_audio,
    write_mono_pcm16_audio,
)


@pytest.mark.parametrize("suffix, expected_format", [(".flac", "FLAC"), (".wav", "WAV")])
def test_lossless_pcm16_audio_round_trip(
    tmp_path: Path,
    suffix: str,
    expected_format: str,
) -> None:
    path = tmp_path / f"clip{suffix}"
    samples = np.linspace(-0.5, 0.5, 1_600, dtype=np.float32)

    write_mono_pcm16_audio(path, samples, 16_000)
    restored, sample_rate_hz = read_mono_pcm16_audio(path)

    information = soundfile.info(path)
    assert information.format == expected_format
    assert information.subtype == "PCM_16"
    assert sample_rate_hz == 16_000
    assert np.allclose(restored, samples, atol=1 / 32768)


def test_audio_file_rejects_lossy_or_ambiguous_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supported lossless suffix"):
        write_mono_pcm16_audio(tmp_path / "clip.mp3", np.ones(10, dtype=np.float32), 16_000)
