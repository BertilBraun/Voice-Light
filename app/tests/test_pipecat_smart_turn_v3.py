from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray
from transformers.feature_extraction_utils import BatchFeature

from app.analyses.end_of_turn.detectors.pipecat_smart_turn_v3 import (
    CandidatePause,
    SmartTurnV3Inference,
    _candidate_pauses,
    _completion_probability,
    _smart_turn_events,
    _speech_segments_from_flags_with_pause,
)
from app.analyses.end_of_turn.service import SpeechSegment


@dataclass
class RecordingFeatureExtractor:
    observed_sample_counts: list[int] = field(default_factory=list)

    def __call__(
        self,
        raw_speech: NDArray[np.float32],
        sampling_rate: int,
        return_tensors: str,
        padding: str,
        max_length: int,
        truncation: bool,
        do_normalize: bool,
    ) -> BatchFeature:
        self.observed_sample_counts.append(raw_speech.size)
        return BatchFeature(data={"input_features": raw_speech.reshape(1, 1, -1)})


@dataclass
class RecordingSession:
    probabilities: list[float]

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        return [np.array([self.probabilities.pop(0)], dtype=np.float32)]


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
    feature_extractor = RecordingFeatureExtractor()
    inference = SmartTurnV3Inference(
        feature_extractor=feature_extractor,
        session=RecordingSession(probabilities=[0.2, 0.8, 0.9]),
    )
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
        inference=inference,
        samples=samples,
        sample_rate=10,
        candidate_pauses=candidate_pauses,
        completion_threshold=0.5,
    )

    assert feature_extractor.observed_sample_counts == [128000, 128000, 128000]
    assert [event.time_seconds for event in events] == pytest.approx([2.2, 3.4])


def test_completion_probability_uses_session_probability() -> None:
    inference = SmartTurnV3Inference(
        feature_extractor=RecordingFeatureExtractor(),
        session=RecordingSession(probabilities=[0.73]),
    )

    probability = _completion_probability(
        inference=inference,
        audio=np.zeros(16000, dtype=np.float32),
    )

    assert probability == pytest.approx(0.73)
