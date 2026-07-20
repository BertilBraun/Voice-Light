from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.shared.audio.loading import AudioTrack
from app.shared.quality import AudioMetadata


@dataclass(frozen=True)
class SharedTimelineAudioPair:
    speaker1: AudioTrack
    speaker2: AudioTrack

    @property
    def duration_seconds(self) -> float:
        return self.speaker1.metadata.duration_seconds


def pad_audio_tracks_to_shared_timeline(
    speaker1: AudioTrack,
    speaker2: AudioTrack,
) -> SharedTimelineAudioPair:
    if speaker1.metadata.sample_rate != speaker2.metadata.sample_rate:
        raise ValueError("Shared audio timeline requires matching sample rates.")
    shared_sample_count = max(len(speaker1.samples), len(speaker2.samples))
    return SharedTimelineAudioPair(
        speaker1=_audio_track_with_sample_count(
            audio=speaker1,
            samples=speaker1.samples,
            sample_count=shared_sample_count,
        ),
        speaker2=_audio_track_with_sample_count(
            audio=speaker2,
            samples=speaker2.samples,
            sample_count=shared_sample_count,
        ),
    )


def _audio_track_with_sample_count(
    audio: AudioTrack,
    samples: NDArray[np.float32],
    sample_count: int,
) -> AudioTrack:
    missing_sample_count = sample_count - len(samples)
    assert missing_sample_count >= 0
    padded_samples = (
        np.pad(samples, ((0, missing_sample_count), (0, 0))) if missing_sample_count else samples
    )
    return AudioTrack(
        samples=padded_samples.astype(np.float32, copy=False),
        metadata=AudioMetadata(
            duration_seconds=sample_count / audio.metadata.sample_rate,
            sample_rate=audio.metadata.sample_rate,
            channels=padded_samples.shape[1],
            sample_count=sample_count,
        ),
    )
