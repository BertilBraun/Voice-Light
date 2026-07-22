from __future__ import annotations

import pytest

from app.local.asr.transcript import SpeakerTrack, Word
from app.local.timeline_repair.transform import (
    GlobalTimelineRepair,
    PiecewiseTimelineRepair,
    canonical_available_intervals,
    interval_intersects_exclusion,
    map_canonical_audio_interval,
    transform_speech_segments,
    transform_words,
)
from app.shared.quality import SpeechSegment

PIECEWISE_REPAIR = PiecewiseTimelineRepair(
    first_shift_seconds=0.2,
    second_shift_seconds=4.0,
    canonical_transition_seconds=100.0,
    exclusion_start_seconds=90.0,
    exclusion_end_seconds=110.0,
)


def test_speaker1_timeline_is_unchanged() -> None:
    words = (Word(text="hello", start_seconds=1.0, end_seconds=1.4),)
    segments = (SpeechSegment(start_seconds=1.0, end_seconds=1.4),)

    assert transform_words(words, SpeakerTrack.SPEAKER1, PIECEWISE_REPAIR) is words
    assert transform_speech_segments(segments, SpeakerTrack.SPEAKER1, PIECEWISE_REPAIR) is segments


def test_global_repair_shifts_all_speaker2_words() -> None:
    words = (
        Word(text="first", start_seconds=1.0, end_seconds=1.2, confidence=0.8),
        Word(text="second", start_seconds=5.0, end_seconds=5.4),
    )

    transformed = transform_words(
        words,
        SpeakerTrack.SPEAKER2,
        GlobalTimelineRepair(shift_seconds=-0.25),
    )

    assert transformed == (
        Word(text="first", start_seconds=0.75, end_seconds=0.95, confidence=0.8),
        Word(text="second", start_seconds=4.75, end_seconds=5.15),
    )


def test_piecewise_words_use_their_side_and_only_straddling_word_is_dropped() -> None:
    words = (
        Word(text="early", start_seconds=95.0, end_seconds=96.0),
        Word(text="early-boundary", start_seconds=99.5, end_seconds=99.8),
        Word(text="straddles", start_seconds=95.5, end_seconds=100.1),
        Word(text="late-boundary", start_seconds=99.8, end_seconds=100.0),
        Word(text="late", start_seconds=105.0, end_seconds=106.0),
    )

    transformed = transform_words(words, SpeakerTrack.SPEAKER2, PIECEWISE_REPAIR)

    assert tuple(word.text for word in transformed) == (
        "early",
        "early-boundary",
        "late-boundary",
        "late",
    )
    assert tuple((word.start_seconds, word.end_seconds) for word in transformed) == (
        (95.2, 96.2),
        (99.7, 100.0),
        (103.8, 104.0),
        (109.0, 110.0),
    )


def test_piecewise_vad_splits_a_segment_at_the_transition() -> None:
    transformed = transform_speech_segments(
        (SpeechSegment(start_seconds=97.0, end_seconds=101.0),),
        SpeakerTrack.SPEAKER2,
        PIECEWISE_REPAIR,
    )

    assert tuple((segment.start_seconds, segment.end_seconds) for segment in transformed) == (
        pytest.approx((97.2, 98.1)),
        pytest.approx((101.9, 105.0)),
    )


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds", "expected"),
    (
        (80.0, 90.0, False),
        (89.9, 90.1, True),
        (90.0, 110.0, True),
        (109.9, 120.0, True),
        (110.0, 120.0, False),
    ),
)
def test_exclusion_intersection_uses_half_open_boundaries(
    start_seconds: float,
    end_seconds: float,
    expected: bool,
) -> None:
    assert interval_intersects_exclusion(start_seconds, end_seconds, PIECEWISE_REPAIR) is expected


def test_audio_mapping_uses_identity_for_speaker1_and_inverse_shift_for_speaker2() -> None:
    speaker1 = map_canonical_audio_interval(
        SpeakerTrack.SPEAKER1,
        canonical_start_seconds=120.0,
        canonical_end_seconds=130.0,
        source_duration_seconds=200.0,
        repair=PIECEWISE_REPAIR,
    )
    speaker2 = map_canonical_audio_interval(
        SpeakerTrack.SPEAKER2,
        canonical_start_seconds=120.0,
        canonical_end_seconds=130.0,
        source_duration_seconds=200.0,
        repair=PIECEWISE_REPAIR,
    )

    assert (speaker1.source_start_seconds, speaker1.source_end_seconds) == (120.0, 130.0)
    assert (speaker2.source_start_seconds, speaker2.source_end_seconds) == (116.0, 126.0)


def test_audio_mapping_rejects_exclusion_and_source_boundary_crossings() -> None:
    with pytest.raises(ValueError, match="exclusion"):
        map_canonical_audio_interval(
            SpeakerTrack.SPEAKER2,
            canonical_start_seconds=89.0,
            canonical_end_seconds=91.0,
            source_duration_seconds=200.0,
            repair=PIECEWISE_REPAIR,
        )
    with pytest.raises(ValueError, match="outside"):
        map_canonical_audio_interval(
            SpeakerTrack.SPEAKER2,
            canonical_start_seconds=1.0,
            canonical_end_seconds=2.0,
            source_duration_seconds=200.0,
            repair=GlobalTimelineRepair(shift_seconds=5.0),
        )


def test_available_intervals_account_for_shifts_source_bounds_and_exclusion() -> None:
    speaker1 = canonical_available_intervals(
        SpeakerTrack.SPEAKER1,
        source_duration_seconds=200.0,
        repair=PIECEWISE_REPAIR,
    )
    speaker2 = canonical_available_intervals(
        SpeakerTrack.SPEAKER2,
        source_duration_seconds=200.0,
        repair=PIECEWISE_REPAIR,
    )

    assert tuple((item.start_seconds, item.end_seconds) for item in speaker1) == (
        (0.0, 90.0),
        (110.0, 200.0),
    )
    assert tuple((item.start_seconds, item.end_seconds) for item in speaker2) == (
        (0.2, 90.0),
        (110.0, 204.0),
    )


def test_global_available_interval_clips_negative_canonical_time() -> None:
    intervals = canonical_available_intervals(
        SpeakerTrack.SPEAKER2,
        source_duration_seconds=10.0,
        repair=GlobalTimelineRepair(shift_seconds=-2.0),
    )

    assert tuple((item.start_seconds, item.end_seconds) for item in intervals) == ((0.0, 8.0),)


def test_invalid_piecewise_repair_is_rejected() -> None:
    with pytest.raises(ValueError, match="inside"):
        PiecewiseTimelineRepair(
            first_shift_seconds=0.0,
            second_shift_seconds=1.0,
            canonical_transition_seconds=100.0,
            exclusion_start_seconds=110.0,
            exclusion_end_seconds=120.0,
        )
