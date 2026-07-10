from pathlib import Path

import pytest

from app.analyses.end_of_turn.detectors.naive_vad import (
    _end_of_turn_events,
    _speech_segments_from_flags,
)
from app.analyses.end_of_turn.service import SpeechSegment
from app.data.sessions import list_sessions


@pytest.mark.parametrize(
    ("speech_flags", "expected_segments"),
    [
        ([False, True, True, False], [SpeechSegment(start_seconds=0.1, end_seconds=0.3)]),
        ([True, False, True], []),
        ([True, True, True], [SpeechSegment(start_seconds=0.0, end_seconds=0.3)]),
    ],
)
def test_speech_segments_from_flags(
    speech_flags: list[bool],
    expected_segments: list[SpeechSegment],
) -> None:
    assert (
        _speech_segments_from_flags(
            speech_flags=speech_flags,
            frame_seconds=0.1,
            min_speech_seconds=0.2,
        )
        == expected_segments
    )


def test_end_of_turn_events_require_minimum_silence() -> None:
    speech_segments = [
        SpeechSegment(start_seconds=0.0, end_seconds=1.0),
        SpeechSegment(start_seconds=1.5, end_seconds=2.0),
        SpeechSegment(start_seconds=3.0, end_seconds=4.0),
    ]

    events = _end_of_turn_events(
        speech_segments=speech_segments,
        min_silence_seconds=0.7,
    )

    assert len(events) == 1
    assert events[0].time_seconds == 2.7
    assert events[0].speech_start_seconds == 1.5
    assert events[0].speech_end_seconds == 2.0
    assert events[0].silence_seconds == 1.0


def test_data_folder_exists() -> None:
    assert Path("data/luel/sessions/manifest.json").exists()


def test_session_listing_uses_session_directories() -> None:
    session_identifiers = {session_entry.identifier for session_entry in list_sessions()}

    assert "pmt_001" in session_identifiers
