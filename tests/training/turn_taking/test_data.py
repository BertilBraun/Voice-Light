from pathlib import Path

import torch

from app.training.turn_taking.data import POLICY_INDEX, build_frame_targets
from app.training.turn_taking.schema import (
    AnnotationSource,
    EventType,
    PolicyClass,
    TurnEvent,
    TurnTakingSample,
)


def test_frame_targets_include_policy_and_nested_horizons() -> None:
    sample = TurnTakingSample(
        sample_id="sample",
        conversation_id="conversation",
        target_speaker_id="target",
        target_audio_path=Path("target.wav"),
        reference_audio_path=None,
        sample_rate_hz=16_000,
        context_start_seconds=0.0,
        decision_start_seconds=1.0,
        decision_end_seconds=3.0,
        events=(
            _event("speech", EventType.TARGET_SPEECH, 0.0, 1.8),
            _event("hold", EventType.HOLD, 1.0, 1.2),
            _event("shift", EventType.TURN_SHIFT, 2.0, 2.2),
            _event("partner", EventType.PARTNER_SPEECH, 2.1, 3.0),
        ),
        source_dataset="test",
        source_license="test",
    )

    targets = build_frame_targets(sample, frame_seconds=0.1, burn_in_seconds=0.5)

    assert targets.policy[20].item() == POLICY_INDEX[PolicyClass.TAKE_TURN]
    assert torch.equal(targets.end_horizons[15], torch.tensor([1.0, 1.0, 1.0]))
    assert not targets.loss_mask[5]
    assert targets.loss_mask[10]
    assert targets.future_activity.shape == (30, 6)


def _event(identifier: str, event_type: EventType, start: float, end: float) -> TurnEvent:
    return TurnEvent(
        event_id=identifier,
        event_type=event_type,
        start_seconds=start,
        end_seconds=end,
        confidence=1.0,
        source=AnnotationSource.HUMAN,
    )
