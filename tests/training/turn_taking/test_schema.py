from pathlib import Path

import pytest
from pydantic import ValidationError

from app.training.turn_taking.schema import (
    AnnotationSource,
    EventType,
    TurnEvent,
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


def _sample(
    decision_start_seconds: float = 1.0, decision_end_seconds: float = 3.0
) -> TurnTakingSample:
    return TurnTakingSample(
        sample_id="sample-1",
        conversation_id="conversation-1",
        target_speaker_id="speaker-a",
        target_audio_path=Path("speaker-a.wav"),
        reference_audio_path=Path("speaker-b.wav"),
        sample_rate_hz=16_000,
        context_start_seconds=0.0,
        decision_start_seconds=decision_start_seconds,
        decision_end_seconds=decision_end_seconds,
        events=(
            TurnEvent(
                event_id="shift-1",
                event_type=EventType.TURN_SHIFT,
                start_seconds=2.0,
                end_seconds=2.2,
                confidence=1.0,
                source=AnnotationSource.HUMAN,
            ),
        ),
        source_dataset="test",
        source_license="test-only",
    )
