from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from app.analyses.end_of_turn.detectors.pipecat_smart_turn_v3 import (
    CandidatePause,
    SmartTurnPrediction,
    _candidate_pauses,
    _smart_turn_events,
    _speech_segments_from_flags_with_pause,
)
from app.analyses.end_of_turn.service import SpeechSegment


@dataclass
class RecordingSmartTurnAnalyzer:
    probabilities: list[float]
    observed_sample_counts: list[int] = field(default_factory=list)

    def _predict_endpoint(self, audio_array: NDArray[np.float32]) -> SmartTurnPrediction:
        self.observed_sample_counts.append(audio_array.size)
        probability = self.probabilities.pop(0)
        return SmartTurnPrediction(
            prediction=1 if probability >= 0.5 else 0,
            probability=probability,
        )


def test_speech_segments_from_flags_with_pause_bridges_short_silence() -> None:
    speech_flags = [True, True, False, True, True, False, False, False]

    speech_segments = _speech_segments_from_flags_with_pause(
        speech_flags=speech_flags,
        frame_seconds=0.1,
        min_speech_seconds=0.2,
        candidate_silence_seconds=0.2,
    )

    assert speech_segments == [SpeechSegment(start_seconds=0.0, end_seconds=0.5)]


def test_candidate_pauses_include_intermediate_and_trailing_silence() -> None:
    speech_segments = [
        SpeechSegment(start_seconds=0.0, end_seconds=1.0),
        SpeechSegment(start_seconds=1.1, end_seconds=2.0),
        SpeechSegment(start_seconds=3.0, end_seconds=3.5),
    ]

    candidate_pauses = _candidate_pauses(
        speech_segments=speech_segments,
        audio_duration_seconds=4.0,
        candidate_silence_seconds=0.2,
    )

    assert candidate_pauses == [
        CandidatePause(
            speech_start_seconds=1.1,
            speech_end_seconds=2.0,
            evaluate_seconds=2.2,
            following_silence_seconds=1.0,
        ),
        CandidatePause(
            speech_start_seconds=3.0,
            speech_end_seconds=3.5,
            evaluate_seconds=3.7,
            following_silence_seconds=0.5,
        ),
    ]


def test_smart_turn_events_accumulate_until_completion() -> None:
    analyzer = RecordingSmartTurnAnalyzer(probabilities=[0.2, 0.8, 0.9])
    samples = np.zeros(40, dtype=np.float32)
    candidate_pauses = [
        CandidatePause(
            speech_start_seconds=0.0,
            speech_end_seconds=1.0,
            evaluate_seconds=1.2,
            following_silence_seconds=0.5,
        ),
        CandidatePause(
            speech_start_seconds=1.5,
            speech_end_seconds=2.0,
            evaluate_seconds=2.2,
            following_silence_seconds=1.0,
        ),
        CandidatePause(
            speech_start_seconds=3.0,
            speech_end_seconds=3.2,
            evaluate_seconds=3.4,
            following_silence_seconds=0.6,
        ),
    ]

    events = _smart_turn_events(
        analyzer=analyzer,
        samples=samples,
        sample_rate=10,
        candidate_pauses=candidate_pauses,
        completion_threshold=0.5,
    )

    assert analyzer.observed_sample_counts == [12, 22, 4]
    assert [event.time_seconds for event in events] == pytest.approx([2.2, 3.4])
