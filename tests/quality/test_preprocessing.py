from __future__ import annotations

import numpy as np
import pytest

from app.shared.audio import AudioTrack
from app.shared.quality import (
    AudioMetadata,
    ConversationAnnotation,
    ProcessingStatus,
    SpeakerConversationAnnotation,
    SpeakerSide,
)
from app.shared.quality_analysis.preprocessing import (
    QUALITY_SAMPLE_RATE,
    prepare_audio_track,
    resample_linear,
)
from app.shared.quality_analysis.service import conversation_count_estimate, score_two_track_sample


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
    assert result.total_quality_score == result.raw_quality_score
    assert "calibrated_quality_score" not in result.model_dump()


def test_conversation_counts_scale_to_full_sample_duration() -> None:
    annotation = ConversationAnnotation(
        annotation_version="test",
        analyzed_duration_seconds=180.0,
        speaker1=empty_speaker_annotation(SpeakerSide.SPEAKER1),
        speaker2=empty_speaker_annotation(SpeakerSide.SPEAKER2),
        speech_segment_count=30,
        turn_count=12,
        turn_taking_count=10,
        interaction_count=15,
        pause_count=4,
        backchannel_count=3,
        interruption_count=2,
        usable_event_count=21,
        events_per_hour=420.0,
        speaker_balance_score=0.9,
        quality_score=0.8,
    )

    estimate = conversation_count_estimate(annotation, represented_duration_seconds=900.0)

    assert estimate is not None
    assert estimate.scale_factor == 5.0
    assert estimate.observed.interaction_count == 15
    assert estimate.estimated.speech_segment_count == 150
    assert estimate.estimated.interaction_count == 75
    assert estimate.estimated.turn_count == 60
    assert estimate.estimated.usable_event_count == 105


def empty_speaker_annotation(side: SpeakerSide) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=(),
        pauses=(),
        backchannels=(),
        turns=(),
        interruptions=(),
        segment_targets=(),
        connection_targets=(),
        speech_duration_seconds=1.0,
        pause_duration_seconds=0.0,
        backchannel_duration_seconds=0.0,
    )


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
