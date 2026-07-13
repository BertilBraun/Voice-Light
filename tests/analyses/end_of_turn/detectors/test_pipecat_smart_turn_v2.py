from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from app.local.analyses.end_of_turn.detectors.pipecat_smart_turn_v2 import (
    SmartTurnInferenceComponents,
    _candidate_speech_segments,
    _model_window_audio,
    _smart_turn_events,
)
from app.local.analyses.end_of_turn.service import SpeechSegment


@dataclass(frozen=True)
class FixedFeatureExtractor:
    def __call__(
        self,
        raw_speech: np.ndarray,
        sampling_rate: int,
        return_tensors: str,
        padding: str,
        max_length: int,
        truncation: bool,
    ) -> dict[str, torch.Tensor]:
        return {"input_values": torch.from_numpy(raw_speech.reshape(1, -1))}


@dataclass(frozen=True)
class FixedModel:
    completion_logit: float

    def eval(self) -> FixedModel:
        return self

    def __call__(self, input_values: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[self.completion_logit]], dtype=torch.float32)


@pytest.mark.parametrize(
    ("completion_logit", "expected_event_count"),
    [
        (1.0, 1),
        (-1.0, 0),
    ],
)
def test_smart_turn_events_gate_candidates_by_completion_probability(
    completion_logit: float,
    expected_event_count: int,
) -> None:
    inference_components = SmartTurnInferenceComponents(
        feature_extractor=FixedFeatureExtractor(),
        model=FixedModel(completion_logit=completion_logit),
    )

    events = _smart_turn_events(
        audio=np.ones(16000, dtype=np.float32),
        sample_rate=16000,
        speech_segments=[SpeechSegment(start_seconds=0.0, end_seconds=0.2)],
        inference_components=inference_components,
        completion_threshold=0.5,
        min_silence_seconds=0.4,
        max_window_seconds=8.0,
    )

    assert len(events) == expected_event_count


def test_candidate_speech_segments_merge_short_internal_pauses() -> None:
    audio = np.concatenate(
        [
            np.full(1600, 0.2, dtype=np.float32),
            np.zeros(1600, dtype=np.float32),
            np.full(1600, 0.2, dtype=np.float32),
            np.zeros(1600, dtype=np.float32),
        ]
    )

    speech_segments = _candidate_speech_segments(
        audio=audio,
        sample_rate=16000,
        frame_seconds=0.1,
        min_speech_seconds=0.1,
        merge_gap_seconds=0.15,
    )

    assert speech_segments == [SpeechSegment(start_seconds=0.0, end_seconds=0.3)]


def test_model_window_audio_keeps_latest_samples() -> None:
    audio = np.arange(20, dtype=np.float32)

    window_audio = _model_window_audio(
        audio=audio,
        sample_rate=10,
        end_seconds=2.0,
        max_window_seconds=0.8,
    )

    np.testing.assert_array_equal(window_audio, np.arange(12, 20, dtype=np.float32))
