from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.shared.audio import AudioTrack
from app.shared.quality import QUALITY_SAMPLE_RATE, AudioMetadata


@dataclass(frozen=True)
class PreparedAudioTrack:
    samples: NDArray[np.float32]
    metadata: AudioMetadata


def prepare_audio_track(
    audio_track: AudioTrack,
    target_sample_rate: int = QUALITY_SAMPLE_RATE,
) -> PreparedAudioTrack:
    if audio_track.samples.ndim != 2:
        raise ValueError("decoded audio samples must have frame and channel dimensions")
    if audio_track.samples.shape[1] != audio_track.metadata.channels:
        raise ValueError("decoded audio channel count does not match metadata")
    mono_samples = audio_track.samples.mean(axis=1)
    samples = resample_linear(
        mono_samples,
        audio_track.metadata.sample_rate,
        target_sample_rate,
    )
    return PreparedAudioTrack(
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
