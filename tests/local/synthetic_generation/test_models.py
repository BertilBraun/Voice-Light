from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.models import (
    CompletionPlacement,
    SpeakerSpecification,
    SyntheticConversationCase,
    SyntheticConversationSet,
)


def test_conversation_set_accepts_typed_case_with_placement() -> None:
    conversation_set = SyntheticConversationSet(
        set_id="turn_taking_examples",
        description="Synthetic conversation specifications for future evaluation.",
        cases=(_conversation_case(),),
    )

    assert conversation_set.cases[0].placement == CompletionPlacement(
        minimum_delay_milliseconds=100,
        maximum_delay_milliseconds=400,
    )


def test_completion_placement_requires_complete_delay_range() -> None:
    with pytest.raises(ValidationError, match="both delay bounds"):
        CompletionPlacement(minimum_delay_milliseconds=100)


def test_conversation_case_requires_distinct_speakers() -> None:
    with pytest.raises(ValidationError, match="distinct speaker_id"):
        SyntheticConversationCase(
            case_id="shared_speaker",
            title="Shared speaker",
            description="Invalid two speaker specification.",
            first_speaker=SpeakerSpecification(speaker_id="speaker_1", text="Hello."),
            second_speaker=SpeakerSpecification(speaker_id="speaker_1", text="Hi."),
        )


def test_conversation_set_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValidationError, match="case IDs must be unique"):
        SyntheticConversationSet(
            set_id="duplicates",
            description="Invalid duplicate case collection.",
            cases=(_conversation_case(), _conversation_case()),
        )


def _conversation_case() -> SyntheticConversationCase:
    return SyntheticConversationCase(
        case_id="completion_after_pause",
        title="Completion after pause",
        description="The second speaker begins after the first speaker finishes.",
        first_speaker=SpeakerSpecification(speaker_id="speaker_1", text="I think we should go."),
        second_speaker=SpeakerSpecification(speaker_id="speaker_2", text="That sounds good."),
        placement=CompletionPlacement(
            minimum_delay_milliseconds=100,
            maximum_delay_milliseconds=400,
        ),
    )
