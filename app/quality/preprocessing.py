from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.audio import AudioTrack
from app.quality.models import AudioMetadata

QUALITY_SAMPLE_RATE = 16_000


def prepare_audio_track(
    audio_track: AudioTrack,
    target_sample_rate: int = QUALITY_SAMPLE_RATE,
) -> AudioTrack:
    samples = resample_linear(
        audio_track.samples,
        audio_track.metadata.sample_rate,
        target_sample_rate,
    )
    return AudioTrack(
        samples=np.clip(samples.astype(np.float32), -1.0, 1.0),
        metadata=AudioMetadata(
            duration_seconds=len(samples) / target_sample_rate,
            sample_rate=target_sample_rate,
            channels=audio_track.metadata.channels,
            sample_count=len(samples),
        ),
    )


def resample_linear(
    samples: NDArray[np.float32],
    source_sample_rate: int,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    if len(samples) == 0 or source_sample_rate == target_sample_rate:
        return samples
    source_duration_seconds = len(samples) / source_sample_rate
    target_sample_count = max(1, round(source_duration_seconds * target_sample_rate))
    source_positions = np.linspace(
        0.0,
        source_duration_seconds,
        num=len(samples),
        endpoint=False,
    )
    target_positions = np.linspace(
        0.0,
        source_duration_seconds,
        num=target_sample_count,
        endpoint=False,
    )
    return np.interp(target_positions, source_positions, samples).astype(np.float32)
