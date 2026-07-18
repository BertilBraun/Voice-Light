from __future__ import annotations

import numpy as np
import pytest

from app.local.ingestion.alignment import pad_audio_tracks_to_shared_timeline
from app.shared.audio import AudioTrack
from app.shared.quality import AudioMetadata

SAMPLE_RATE = 16_000


def test_tail_padding_preserves_both_track_starts_and_shared_timeline() -> None:
    speaker1 = audio_track(sample_count=32_000, value=1.0)
    speaker2 = audio_track(sample_count=40_000, value=2.0)

    padded = pad_audio_tracks_to_shared_timeline(
        speaker1=speaker1,
        speaker2=speaker2,
    )

    assert len(padded.speaker1.samples) == len(padded.speaker2.samples) == 40_000
    assert padded.duration_seconds == pytest.approx(2.5)
    assert np.all(padded.speaker1.samples[:32_000] == 1.0)
    assert np.all(padded.speaker1.samples[32_000:] == 0.0)
    assert np.all(padded.speaker2.samples == 2.0)


def test_tail_length_difference_is_end_padded_instead_of_invalidated() -> None:
    padded = pad_audio_tracks_to_shared_timeline(
        speaker1=audio_track(sample_count=16_000, value=1.0),
        speaker2=audio_track(sample_count=48_000, value=2.0),
    )

    assert padded.duration_seconds == pytest.approx(3.0)
    assert len(padded.speaker1.samples) == len(padded.speaker2.samples) == 48_000


def audio_track(sample_count: int, value: float) -> AudioTrack:
    samples = np.full((sample_count, 1), value, dtype=np.float32)
    return AudioTrack(
        samples=samples,
        metadata=AudioMetadata(
            duration_seconds=sample_count / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            channels=1,
            sample_count=sample_count,
        ),
    )
