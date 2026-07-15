from pathlib import Path

import pytest
from pydantic import ValidationError

from app.training.turn_taking.schema import (
    DecisionTarget,
    EventTargetDistribution,
    TurnTakingSample,
    read_manifest,
    write_manifest,
)


def test_manifest_round_trip(tmp_path: Path) -> None:
    sample = _sample()
    path = tmp_path / "manifest.jsonl"

    write_manifest(path, [sample])

    assert read_manifest(path) == (sample,)


def test_sample_rejects_invalid_decision_interval() -> None:
    with pytest.raises(ValidationError, match="Sample times"):
        _sample(decision_start_seconds=4.0, decision_end_seconds=3.0)


def test_event_target_rejects_non_normalized_distribution() -> None:
    with pytest.raises(ValidationError, match="sum to one"):
        EventTargetDistribution(
            turn_completion=0.2,
            continuation_pause=0.2,
            backchannel=0.2,
            interruption=0.2,
            other=0.3,
        )


def _sample(
    decision_start_seconds: float = 1.0, decision_end_seconds: float = 3.0
) -> TurnTakingSample:
    return TurnTakingSample(
        sample_id="sample-1",
        conversation_id="conversation-1",
        target_speaker_id="speaker-a",
        target_audio_path=Path("speaker-a.wav"),
        annotation_reference_audio_path=Path("speaker-b.wav"),
        sample_rate_hz=16_000,
        context_start_seconds=0.0,
        decision_start_seconds=decision_start_seconds,
        decision_end_seconds=decision_end_seconds,
        decisions=(
            DecisionTarget(
                time_seconds=2.0,
                yield_probability=0.7,
                primary_reliability=None,
                event_distribution=EventTargetDistribution(
                    turn_completion=0.7,
                    continuation_pause=0.2,
                    backchannel=0.05,
                    interruption=0.03,
                    other=0.02,
                ),
                event_reliability=0.9,
                future_user_activity=(False, False, True, True),
            ),
        ),
        source_dataset="test",
        source_license="test-only",
    )
