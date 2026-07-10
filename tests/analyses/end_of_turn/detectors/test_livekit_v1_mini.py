from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from app.analyses.end_of_turn.detectors.livekit_v1_mini import (
    EOT_MAX_SAMPLES,
    VAD_WINDOW_SAMPLES,
    CandidatePause,
    _end_of_turn_events,
    _model_window,
    _speech_segments_from_vad,
)
from app.analyses.end_of_turn.service import SpeechSegment


@dataclass
class FixedVadModel:
    probabilities: list[float]
    call_index: int = 0

    def predict(self, pcm: NDArray[np.int16]) -> float:
        probability = self.probabilities[self.call_index]
        self.call_index += 1
        return probability


@dataclass(frozen=True)
class FixedEndOfTurnModel:
    completion_probability: float

    def predict(self, pcm: NDArray[np.int16]) -> float:
        return self.completion_probability


def test_speech_segments_from_vad_closes_segment_after_candidate_silence() -> None:
    samples = np.ones(VAD_WINDOW_SAMPLES * 9, dtype=np.int16)
    vad_model = FixedVadModel(probabilities=[0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8])

    speech_segments = _speech_segments_from_vad(
        samples=samples,
        sample_rate=16000,
        vad_model=vad_model,
        vad_speech_threshold=0.5,
        min_speech_seconds=0.09,
        candidate_silence_seconds=0.09,
    )

    assert speech_segments == [
        SpeechSegment(start_seconds=0.0, end_seconds=0.096),
        SpeechSegment(start_seconds=0.192, end_seconds=0.288),
    ]


@pytest.mark.parametrize(
    ("completion_probability", "expected_event_count"),
    [
        (0.6, 1),
        (0.2, 0),
    ],
)
def test_end_of_turn_events_gate_candidate_pauses_by_probability(
    completion_probability: float,
    expected_event_count: int,
) -> None:
    events = _end_of_turn_events(
        samples=np.ones(16000, dtype=np.int16),
        sample_rate=16000,
        candidate_pauses=[
            CandidatePause(
                speech_start_seconds=0.0,
                speech_end_seconds=0.2,
                evaluate_seconds=0.5,
                following_silence_seconds=0.8,
            )
        ],
        end_of_turn_model=FixedEndOfTurnModel(completion_probability=completion_probability),
        end_of_turn_threshold=0.36,
    )

    assert len(events) == expected_event_count


def test_model_window_keeps_latest_livekit_receptive_field() -> None:
    samples = np.arange(EOT_MAX_SAMPLES + 10, dtype=np.int16)

    window = _model_window(
        samples=samples,
        sample_rate=16000,
        end_seconds=(EOT_MAX_SAMPLES + 10) / 16000,
    )

    np.testing.assert_array_equal(window, np.arange(10, EOT_MAX_SAMPLES + 10, dtype=np.int16))
