from __future__ import annotations

import numpy as np
import pytest

from app.compute.quality.preprocessing import (
    QUALITY_SAMPLE_RATE,
    prepare_audio_track,
    resample_linear,
)
from app.compute.quality.service import score_two_track_sample
from app.shared.audio import AudioTrack
from app.shared.quality import AudioMetadata, ProcessingStatus


def audio_track(sample_rate: int, duration_seconds: float = 1.0) -> AudioTrack:
    sample_count = round(sample_rate * duration_seconds)
    sample_positions = np.arange(sample_count, dtype=np.float32) / sample_rate
    mono_samples = (0.1 * np.sin(2.0 * np.pi * 220.0 * sample_positions)).astype(np.float32)
    samples = mono_samples[:, np.newaxis]
    return AudioTrack(
        samples=samples,
        metadata=AudioMetadata(
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=1,
            sample_count=sample_count,
        ),
    )


def test_prepare_audio_track_resamples_to_quality_sample_rate() -> None:
    prepared_track = prepare_audio_track(audio_track(sample_rate=8_000))

    assert prepared_track.metadata.sample_rate == QUALITY_SAMPLE_RATE
    assert prepared_track.metadata.sample_count == QUALITY_SAMPLE_RATE
    assert prepared_track.metadata.duration_seconds == 1.0


def test_prepare_audio_track_averages_channels() -> None:
    source_track = audio_track(sample_rate=QUALITY_SAMPLE_RATE)
    stereo_track = AudioTrack(
        samples=np.column_stack((source_track.samples[:, 0], -source_track.samples[:, 0])),
        metadata=source_track.metadata.model_copy(update={"channels": 2}),
    )

    prepared_track = prepare_audio_track(stereo_track)

    assert prepared_track.samples == pytest.approx(np.zeros(QUALITY_SAMPLE_RATE))


def test_quality_assessment_accepts_tracks_with_different_source_sample_rates() -> None:
    result = score_two_track_sample(
        sample_id="sample",
        speaker1_audio=audio_track(sample_rate=8_000),
        speaker2_audio=audio_track(sample_rate=16_000),
        speaker1_uri="speaker1.wav",
        speaker2_uri="speaker2.wav",
    )

    assert result.status == ProcessingStatus.COMPLETED
    assert result.error is None


@pytest.mark.parametrize("source_sample_rate,target_sample_rate", [(0, 16_000), (16_000, 0)])
def test_resample_linear_rejects_nonpositive_sample_rates(
    source_sample_rate: int,
    target_sample_rate: int,
) -> None:
    with pytest.raises(ValueError, match="sample rates must be positive"):
        resample_linear(
            np.ones(4, dtype=np.float32),
            source_sample_rate,
            target_sample_rate,
        )
