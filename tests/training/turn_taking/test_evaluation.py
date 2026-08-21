import math

import pytest
import torch

from app.training.turn_taking.config import LossConfig
from app.training.turn_taking.data import FrameTargets
from app.training.turn_taking.evaluation import (
    EvaluationAccumulator,
    EvaluationHead,
    EvaluationModel,
    PredictionProbabilities,
    assistant_floor_probabilities,
    fit_class_priors,
)


def test_evaluation_accumulator_matches_masked_proper_scores() -> None:
    targets = _targets()
    predictions = PredictionProbabilities(
        user_yield=torch.tensor([[0.25, 0.75]]),
        events=torch.tensor([[[0.25] * 5, [0.75] * 5]]),
        future_activity=torch.tensor([[[0.25] * 4, [0.75] * 4]]),
    )
    accumulator = EvaluationAccumulator()
    accumulator.update(predictions, targets, torch.ones((1, 2), dtype=torch.bool))

    result = accumulator.finalize(EvaluationModel.TRAINED_ADAPTER, LossConfig())

    expected_binary_cross_entropy = -math.log(0.75)
    assert result.primary_loss == pytest.approx(expected_binary_cross_entropy)
    assert result.event_loss == pytest.approx(expected_binary_cross_entropy)
    assert result.future_activity_loss == pytest.approx(expected_binary_cross_entropy)
    assert result.total_loss == pytest.approx(expected_binary_cross_entropy * 1.5)
    assert result.heads[0].head is EvaluationHead.USER_YIELD
    assert result.heads[0].brier_score == pytest.approx(0.0625)
    assert result.heads[0].effective_support == 2.0


def test_class_priors_use_only_valid_targets() -> None:
    targets = _targets()
    targets.primary_mask[0, 0] = False
    targets.event_mask[0, 0] = False
    targets.future_activity_mask[0, 0] = False

    priors = fit_class_priors((targets,))

    assert priors == pytest.approx((1.0,) * 10)


def test_assistant_floor_heuristic_aligns_to_output_frames() -> None:
    probabilities = assistant_floor_probabilities(
        assistant_speaking=torch.tensor([[0.0, 1.0]]),
        priors=(0.5,) * 10,
        frame_count=4,
    )

    assert probabilities.user_yield[0].tolist() == pytest.approx([0.01, 0.01, 0.99, 0.99])
    assert probabilities.events.shape == (1, 4, 5)
    assert probabilities.future_activity.shape == (1, 4, 4)


def _targets() -> FrameTargets:
    binary = torch.tensor([[0.0, 1.0]])
    return FrameTargets(
        yield_probability=binary.clone(),
        primary_weight=torch.ones((1, 2)),
        primary_mask=torch.ones((1, 2), dtype=torch.bool),
        event_targets=binary.unsqueeze(-1).expand(1, 2, 5).clone(),
        event_mask=torch.ones((1, 2, 5), dtype=torch.bool),
        future_activity=binary.unsqueeze(-1).expand(1, 2, 4).clone(),
        future_activity_mask=torch.ones((1, 2, 4), dtype=torch.bool),
    )
