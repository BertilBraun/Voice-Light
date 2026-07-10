from __future__ import annotations

import sys
from dataclasses import dataclass
from types import TracebackType

import numpy as np
import pytest

from app.analyses.end_of_turn.detectors.pipecat_smart_turn_v2 import (
    SmartTurnInferenceComponents,
    _candidate_speech_segments,
    _model_window_audio,
    _smart_turn_events,
)
from app.analyses.end_of_turn.service import SpeechSegment


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
    ) -> dict[str, np.ndarray]:
        return {"input_values": raw_speech.reshape(1, -1)}


@dataclass(frozen=True)
class FixedModel:
    completion_probability: float

    def eval(self) -> FixedModel:
        return self

    def __call__(self, input_values: np.ndarray) -> FixedSequenceClassifierOutput:
        return FixedSequenceClassifierOutput(logits=FixedTensor(value=self.completion_probability))


@dataclass(frozen=True)
class FixedSequenceClassifierOutput:
    logits: FixedTensor


@dataclass(frozen=True)
class FixedTensor:
    value: float

    def squeeze(self) -> FixedTensor:
        return self

    def detach(self) -> FixedTensor:
        return self

    def cpu(self) -> FixedTensor:
        return self

    def item(self) -> float:
        return self.value


@dataclass(frozen=True)
class FixedNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False


@dataclass(frozen=True)
class FixedTorch:
    def no_grad(self) -> FixedNoGrad:
        return FixedNoGrad()

    def sigmoid(self, fixed_tensor: FixedTensor) -> FixedTensor:
        return fixed_tensor


@pytest.mark.parametrize(
    ("completion_probability", "expected_event_count"),
    [
        (0.7, 1),
        (0.3, 0),
    ],
)
def test_smart_turn_events_gate_candidates_by_completion_probability(
    monkeypatch: pytest.MonkeyPatch,
    completion_probability: float,
    expected_event_count: int,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FixedTorch())
    inference_components = SmartTurnInferenceComponents(
        feature_extractor=FixedFeatureExtractor(),
        model=FixedModel(completion_probability=completion_probability),
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
