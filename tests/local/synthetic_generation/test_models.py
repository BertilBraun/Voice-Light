from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.labels import materialize_training_sample
from app.local.synthetic_generation.models import (
    BackchannelEvent,
    PauseEvent,
    SpeakerRole,
    SpeakerSpecification,
    SpeechEvent,
    SyntheticConversationPlan,
    TurnBoundary,
)
from app.local.training_samples.service import FRAME_SECONDS, INPUT_DURATION_SECONDS


def test_plan_requires_one_speaker_per_role() -> None:
    with pytest.raises(ValidationError, match="one user and one assistant"):
        _plan(
            speakers=(
                _speaker("speaker_1", SpeakerRole.USER),
                _speaker("speaker_2", SpeakerRole.USER),
            )
        )


def test_plan_rejects_unknown_pause_continuation() -> None:
    with pytest.raises(ValidationError, match="unknown continuation"):
        _plan(
            events=(
                PauseEvent(
                    event_id="pause_one",
                    speaker_id="speaker_1",
                    start_seconds=1.0,
                    end_seconds=1.5,
                    continuation_event_id="missing_event",
                ),
            )
        )


def test_labels_derive_completion_and_continuation_from_plan() -> None:
    sample = materialize_training_sample(
        _plan(), Path("audio/user.wav"), Path("audio/assistant.wav")
    )

    continuation_frame = round(2.0 / FRAME_SECONDS)
    completion_frame = round(4.0 / FRAME_SECONDS)
    assert sample.turn_completion[continuation_frame] == 0.0
    assert sample.continuation_pause[continuation_frame] == 1.0
    assert sample.turn_completion[completion_frame] == 1.0
    assert sample.continuation_pause[completion_frame] == 0.0
    assert sample.p_assistant_backchannel[round(1.0 / FRAME_SECONDS)] == 1.0
    assert sample.split.value == "train"


def test_labels_preserve_overlap_on_independent_floor_channels() -> None:
    sample = materialize_training_sample(
        _plan(), Path("audio/user.wav"), Path("audio/assistant.wav")
    )

    overlap_frame = round(3.7 / FRAME_SECONDS)
    assert sample.p_user_has_floor[overlap_frame] == 1.0
    assert sample.assistant_has_floor[overlap_frame] == 1.0


def _speaker(speaker_id: str, role: SpeakerRole) -> SpeakerSpecification:
    return SpeakerSpecification(
        speaker_id=speaker_id,
        role=role,
        voice_id=f"pilot_{role.value}",
        language="en-US",
    )


def _plan(
    speakers: tuple[SpeakerSpecification, SpeakerSpecification] | None = None,
    events: tuple[SpeechEvent | PauseEvent | BackchannelEvent, ...] | None = None,
) -> SyntheticConversationPlan:
    default_events: tuple[SpeechEvent | PauseEvent | BackchannelEvent, ...] = (
        SpeechEvent(
            event_id="user_clause",
            speaker_id="speaker_1",
            text="I was thinking",
            start_seconds=0.4,
            end_seconds=2.0,
            boundary_after=TurnBoundary.CONTINUATION,
        ),
        BackchannelEvent(
            event_id="assistant_ack",
            speaker_id="speaker_2",
            text="Mm-hm",
            start_seconds=1.0,
            end_seconds=1.4,
        ),
        PauseEvent(
            event_id="user_pause",
            speaker_id="speaker_1",
            start_seconds=2.0,
            end_seconds=2.6,
            continuation_event_id="user_completion",
        ),
        SpeechEvent(
            event_id="user_completion",
            speaker_id="speaker_1",
            text="we could leave before lunch.",
            start_seconds=2.6,
            end_seconds=4.0,
            boundary_after=TurnBoundary.COMPLETION,
        ),
        SpeechEvent(
            event_id="assistant_response",
            speaker_id="speaker_2",
            text="That works for me.",
            start_seconds=3.6,
            end_seconds=5.0,
            boundary_after=TurnBoundary.COMPLETION,
        ),
    )
    return SyntheticConversationPlan(
        plan_id="pilot_overlap",
        description="Continuation, backchannel, completion, and overlapping response.",
        seed=7,
        duration_seconds=INPUT_DURATION_SECONDS,
        speakers=speakers
        or (
            _speaker("speaker_1", SpeakerRole.USER),
            _speaker("speaker_2", SpeakerRole.ASSISTANT),
        ),
        events=events or default_events,
    )
