from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.audio.wav import mono_samples
from app.quality.models import AudioMetadata


@dataclass(frozen=True)
class AudioTrack:
    samples: NDArray[np.float32]
    metadata: AudioMetadata


def load_audio(path: Path) -> AudioTrack:
    with wave.open(str(path), "rb") as wave_reader:
        sample_rate = wave_reader.getframerate()
        sample_width = wave_reader.getsampwidth()
        channel_count = wave_reader.getnchannels()
        frame_count = wave_reader.getnframes()
        fragment = wave_reader.readframes(frame_count)
    integer_samples = mono_samples(
        fragment=fragment,
        sample_width=sample_width,
        channel_count=channel_count,
    )
    maximum_amplitude = float((1 << (sample_width * 8 - 1)) - 1)
    normalized_samples = np.clip(integer_samples / maximum_amplitude, -1.0, 1.0).astype(np.float32)
    return AudioTrack(
        samples=normalized_samples,
        metadata=AudioMetadata(
            duration_seconds=frame_count / sample_rate,
            sample_rate=sample_rate,
            channels=channel_count,
            sample_count=frame_count,
        ),
    )
