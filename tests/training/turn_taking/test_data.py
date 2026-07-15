from pathlib import Path

import pytest
import torch

from app.training.turn_taking.data import build_assistant_speaking_input, build_frame_targets
from app.training.turn_taking.schema import (
    ActivitySpan,
    DecisionTarget,
    EventTargetDistribution,
    TurnTakingSample,
)


def test_frame_targets_are_sparse_soft_and_separately_weighted() -> None:
    sample = TurnTakingSample(
        sample_id="sample",
        conversation_id="conversation",
        target_speaker_id="target",
        target_audio_path=Path("target.wav"),
        annotation_reference_audio_path=None,
        sample_rate_hz=16_000,
        context_start_seconds=0.0,
        decision_start_seconds=0.0,
        decision_end_seconds=3.0,
        assistant_speech_spans=(
            ActivitySpan(start_seconds=0.4, end_seconds=0.8),
            ActivitySpan(start_seconds=1.5, end_seconds=2.0),
        ),
        decisions=(
            _decision(time_seconds=0.5, yield_probability=0.2, reliability=0.9),
            _decision(time_seconds=1.5, yield_probability=0.7, reliability=0.4),
            _decision(time_seconds=2.0, yield_probability=0.3, reliability=None),
        ),
        source_dataset="test",
        source_license="test",
    )

    targets = build_frame_targets(
        sample,
        frame_seconds=0.1,
        burn_in_seconds=1.0,
        unmeasured_reliability_weight=0.75,
    )

    assert not targets.primary_mask[5]
    assert targets.primary_mask[15]
    assert targets.yield_probability[15].item() == pytest.approx(0.7)
    assert targets.primary_weight[15].item() == pytest.approx(0.4)
    assert targets.primary_weight[20].item() == pytest.approx(0.75)
    assert torch.equal(targets.future_activity_mask[15], torch.tensor([True, True, False, True]))
    assert targets.event_distribution.shape == (30, 5)
    assert targets.future_activity.shape == (30, 4)

    assistant_speaking = build_assistant_speaking_input(sample, frame_seconds=0.1)
    assert assistant_speaking[4]
    assert not assistant_speaking[8]
    assert assistant_speaking[15]
    assert not assistant_speaking[20]


def _decision(
    time_seconds: float, yield_probability: float, reliability: float | None
) -> DecisionTarget:
    return DecisionTarget(
        time_seconds=time_seconds,
        yield_probability=yield_probability,
        primary_reliability=reliability,
        event_distribution=EventTargetDistribution(
            turn_completion=yield_probability,
            continuation_pause=1.0 - yield_probability,
            backchannel=0.0,
            interruption=0.0,
            other=0.0,
        ),
        event_reliability=reliability,
        future_user_activity=(False, True, None, True),
    )
