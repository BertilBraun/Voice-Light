from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile

SUPPORTED_LOSSLESS_AUDIO_SUFFIXES = frozenset({".flac", ".wav"})


def read_mono_pcm16_audio(path: Path) -> tuple[np.ndarray, int]:
    _validate_suffix(path)
    information = soundfile.info(path)
    if information.subtype != "PCM_16":
        raise ValueError(f"Rendered user clip must use 16-bit PCM: {path}")
    samples, sample_rate_hz = soundfile.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    return np.mean(samples, axis=1, dtype=np.float32), int(sample_rate_hz)


def write_mono_pcm16_audio(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    _validate_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(
        path,
        np.clip(samples, -1.0, 1.0),
        sample_rate_hz,
        subtype="PCM_16",
    )


def _validate_suffix(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_LOSSLESS_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_LOSSLESS_AUDIO_SUFFIXES))
        raise ValueError(f"Audio path must use a supported lossless suffix ({supported}): {path}")
