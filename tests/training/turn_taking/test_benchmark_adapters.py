from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor

from app.training.turn_taking.benchmark_adapters import (
    FloatProbabilityScorer,
    Int16ProbabilityScorer,
    livekit_candidate_predictions,
    silero_candidate_predictions,
    smart_turn_candidate_predictions,
)
from app.training.turn_taking.benchmark_models import CandidateTargetPoint, SilenceCandidate


@dataclass
class RecordingAudioProvider:
    sample_rate_hz: int
    loads: list[tuple[float, float]] = field(default_factory=list)

    def load(
        self,
        candidate: SilenceCandidate,
        start_seconds: float,
        end_seconds: float,
    ) -> NDArray[np.float32]:
        self.loads.append((start_seconds, end_seconds))
        return np.zeros(round((end_seconds - start_seconds) * self.sample_rate_hz), np.float32)


@dataclass
class ConstantFloatScorer(FloatProbabilityScorer):
    probability: float
    observed_sample_counts: list[int] = field(default_factory=list)

    def score(self, audio: NDArray[np.float32]) -> float:
        self.observed_sample_counts.append(audio.size)
        return self.probability


@dataclass
class ConstantInt16Scorer(Int16ProbabilityScorer):
    probability: float
    observed_dtypes: list[np.dtype[np.int16]] = field(default_factory=list)

    def score(self, audio: NDArray[np.int16]) -> float:
        self.observed_dtypes.append(audio.dtype)
        return self.probability


@dataclass
class FakeSileroModel:
    probabilities: list[float]
    reset_count: int = 0

    def __call__(self, audio: Tensor, sample_rate_hz: int) -> Tensor:
        assert audio.numel() == 512
        assert sample_rate_hz == 16_000
        return torch.tensor([self.probabilities.pop(0)])

    def reset_states(self) -> None:
        self.reset_count += 1


def test_smart_turn_scores_once_at_native_200ms_candidate() -> None:
    provider = _provider()
    scorer = ConstantFloatScorer(probability=0.73)

    predictions = smart_turn_candidate_predictions(
        candidates=(_candidate(),),
        audio_provider=provider,
        scorer=scorer,
        candidate_silence_seconds=0.2,
        maximum_window_seconds=8.0,
    )

    assert provider.loads == [(1.0, 2.24)]
    assert scorer.observed_sample_counts == [round(1.24 * 16_000)]
    assert len(predictions) == 1
    assert predictions[0].silence_duration_seconds == pytest.approx(0.24)
    assert predictions[0].yield_probability == pytest.approx(0.73)


def test_livekit_scores_int16_window_once_at_native_300ms_candidate() -> None:
    provider = _provider()
    scorer = ConstantInt16Scorer(probability=0.61)

    predictions = livekit_candidate_predictions(
        candidates=(_candidate(),),
        audio_provider=provider,
        scorer=scorer,
        candidate_silence_seconds=0.3,
        maximum_window_seconds=1.2,
    )

    assert provider.loads[0] == pytest.approx((1.12, 2.32))
    assert scorer.observed_dtypes == [np.dtype(np.int16)]
    assert predictions[0].silence_duration_seconds == pytest.approx(0.32)


def test_silero_emits_only_native_32ms_frames_inside_candidate_silence() -> None:
    provider = _provider()
    model = FakeSileroModel(probabilities=[0.9] * 31 + [0.2] * 19)

    predictions = silero_candidate_predictions(
        candidates=(_candidate(),),
        audio_provider=provider,
        model=model,
    )

    assert model.reset_count == 1
    assert provider.loads == [(1.0, 2.6)]
    assert predictions[0].silence_duration_seconds == pytest.approx(0.024)
    assert predictions[-1].silence_duration_seconds == pytest.approx(0.6)
    assert predictions[-1].yield_probability == pytest.approx(0.8)


def _provider() -> RecordingAudioProvider:
    return RecordingAudioProvider(
        sample_rate_hz=16_000,
    )


def _candidate() -> SilenceCandidate:
    return SilenceCandidate(
        candidate_id="1" * 64,
        dataset_id="dataset-id",
        dataset_name="dataset",
        conversation_id="conversation",
        external_id="external",
        user_side="speaker1",
        user_audio_path="audio.flac",
        preceding_speech_start_seconds=1.0,
        start_seconds=2.0,
        end_seconds=2.6,
        target_points=tuple(
            CandidateTargetPoint(
                absolute_time_seconds=2.0 + duration,
                silence_duration_seconds=duration,
                yield_probability=0.8,
            )
            for duration in (0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56)
        ),
        categories=("turn_shift",),
        source_window_ids=("2" * 64,),
    )
