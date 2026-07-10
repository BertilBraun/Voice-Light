from __future__ import annotations

import numpy as np

from app.analyses.end_of_turn.detectors.silero_vad import (
    SileroSpeechTimestamp,
    _end_of_turn_events,
    _resample_audio,
    _speech_segments_from_silero_timestamps,
)
from app.analyses.end_of_turn.service import SpeechSegment


def test_speech_segments_from_silero_timestamps_preserve_seconds() -> None:
    timestamps: list[SileroSpeechTimestamp] = [
        {"start": 0.0320004, "end": 0.5120004},
        {"start": 1.28, "end": 1.6},
    ]

    speech_segments = _speech_segments_from_silero_timestamps(timestamps=timestamps)

    assert speech_segments == [
        SpeechSegment(start_seconds=0.032, end_seconds=0.512),
        SpeechSegment(start_seconds=1.28, end_seconds=1.6),
    ]


def test_end_of_turn_events_use_fixed_silence_hysteresis() -> None:
    events = _end_of_turn_events(
        speech_segments=[
            SpeechSegment(start_seconds=0.1, end_seconds=0.4),
            SpeechSegment(start_seconds=0.7, end_seconds=0.9),
            SpeechSegment(start_seconds=1.5, end_seconds=1.8),
        ],
        audio_duration_seconds=2.4,
        min_silence_seconds=0.4,
    )

    assert [event.time_seconds for event in events] == [1.3, 2.2]
    assert [event.silence_seconds for event in events] == [0.6, 0.6]


def test_resample_audio_keeps_duration() -> None:
    audio = np.arange(4, dtype=np.float32)

    resampled_audio = _resample_audio(
        audio=audio,
        source_sample_rate=4,
        target_sample_rate=8,
    )

    assert len(resampled_audio) == 8
