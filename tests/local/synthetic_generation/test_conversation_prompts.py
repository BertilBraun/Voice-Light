from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.conversation_prompts import (
    Affect,
    AssistantTurnPrompt,
    BaseUserVoice,
    CompletionUserPrompt,
    ConversationPromptGeneratorProvenance,
    Energy,
    EnglishAccent,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    HoldUserPrompt,
    NonFloorFeedbackKind,
    NonFloorFeedbackUserPrompt,
    PerceivedAge,
    SegmentDelivery,
    SpeakingPace,
    SpeechAct,
    TopicDomain,
    VocalPitch,
    VocalWeight,
    conversation_generation_instruction,
    qwen_voice_instruction,
    representative_conversation_briefs,
    validate_conversation_prompt_set_id,
)


def test_representative_briefs_are_deterministic_and_cover_dimensions() -> None:
    briefs = representative_conversation_briefs(count=20, seed=41)

    assert briefs == representative_conversation_briefs(count=20, seed=41)
    assert len({brief.domain for brief in briefs}) == len(TopicDomain)
    assert {brief.pace for brief in briefs} == set(SpeakingPace)
    assert {brief.affect for brief in briefs} == set(Affect)
    assert {condition for brief in briefs for condition in brief.required_conditions} == {
        "completion",
        "hold",
        "non_floor_feedback",
        "response_floor_claim",
        "interruption_floor_claim",
    }


@pytest.mark.parametrize("count", [9, 21])
def test_representative_briefs_reject_non_pilot_size(count: int) -> None:
    with pytest.raises(ValueError, match="10 to 20"):
        representative_conversation_briefs(count=count, seed=41)


def test_set_id_is_validated_before_model_generation() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        validate_conversation_prompt_set_id("invalid-pilot")


def test_generation_instruction_fixes_semantics_without_timestamps() -> None:
    brief = representative_conversation_briefs(count=10, seed=41)[0]

    instruction = conversation_generation_instruction(brief)

    assert brief.plan_id in instruction
    assert brief.domain.value in instruction
    assert "English-only" in instruction
    assert "not fabricated audio timestamps" in instruction
    assert "will not be synthesized" in instruction
    assert '"discriminator":{"mapping"' in instruction


def test_plan_rejects_unknown_assistant_reference() -> None:
    values = _plan().model_dump()
    values["user_prompts"][1]["during_assistant_turn_id"] = "assistant_9"

    with pytest.raises(ValidationError, match="unknown assistant turn"):
        EnglishConversationPromptPlan.model_validate(values)


def test_plan_is_english_only_by_schema_and_validation() -> None:
    values = _plan().model_dump()
    values["language"] = "es"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnglishConversationPromptPlan.model_validate(values)

    values = _plan().model_dump()
    values["user_prompts"][0]["text"] = "Todav\u00eda no est\u00e1 listo."
    with pytest.raises(ValidationError, match="unaccented Latin letters"):
        EnglishConversationPromptPlan.model_validate(values)


def test_plan_accepts_english_typographic_punctuation() -> None:
    values = _plan().model_dump()
    values["user_prompts"][0]["text"] = "That's useful\u2014I'd try it tomorrow."

    plan = EnglishConversationPromptPlan.model_validate(values)

    assert "\u2014" in plan.user_prompts[0].text


def test_prompt_set_rejects_duplicate_spoken_text() -> None:
    first = _plan()
    second = first.model_copy(update={"plan_id": "pilot_second"})

    with pytest.raises(ValidationError, match="texts must be unique"):
        EnglishConversationPromptSet(
            set_id="pilot",
            provenance=_provenance(),
            plans=(first, second),
        )


def test_qwen_instruction_combines_stable_voice_and_segment_delivery() -> None:
    plan = _plan()

    instruction = qwen_voice_instruction(
        plan.base_user_voice,
        plan.user_prompts[0].delivery,
    )

    assert "general american" in instruction
    assert "fast pace" in instruction
    assert "engaging affect" in instruction
    assert "clean, close-mic" in instruction


def _plan() -> EnglishConversationPromptPlan:
    delivery = SegmentDelivery(
        pace=SpeakingPace.FAST,
        energy=Energy.ANIMATED,
        affect=Affect.ENGAGING,
    )
    return EnglishConversationPromptPlan(
        plan_id="pilot_first",
        seed=41,
        domain=TopicDomain.TECHNOLOGY,
        topic="Choosing a useful home automation routine",
        target_duration_seconds=24.0,
        base_user_voice=BaseUserVoice(
            perceived_age=PerceivedAge.ADULT,
            accent=EnglishAccent.GENERAL_AMERICAN,
            pitch=VocalPitch.MEDIUM,
            vocal_weight=VocalWeight.MEDIUM,
        ),
        assistant_turns=(
            AssistantTurnPrompt(
                turn_id="assistant_1",
                sequence_index=1,
                text="Which routine would save you the most time each day?",
                speaking_rate_words_per_minute=175,
                punctuation_pause_seconds=0.2,
            ),
        ),
        user_prompts=(
            CompletionUserPrompt(
                unit_id="user_1",
                sequence_index=0,
                speech_act=SpeechAct.OPINION,
                delivery=delivery,
                text="I would start with the lights because I adjust them every evening.",
            ),
            NonFloorFeedbackUserPrompt(
                unit_id="user_2",
                sequence_index=2,
                speech_act=SpeechAct.ANSWER,
                delivery=delivery,
                text="Right.",
                feedback_kind=NonFloorFeedbackKind.BACKCHANNEL,
                during_assistant_turn_id="assistant_1",
            ),
            HoldUserPrompt(
                unit_id="user_3",
                sequence_index=3,
                speech_act=SpeechAct.EXPLANATION,
                delivery=delivery,
                text_before_pause="The morning routine could also help",
                text_after_pause="but I would need to test the timing first.",
                pause_duration_seconds=0.8,
            ),
        ),
    )


def _provenance() -> ConversationPromptGeneratorProvenance:
    return ConversationPromptGeneratorProvenance(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="a" * 40,
        runtime_version="5.0.0",
        seed=41,
        requested_plan_count=20,
    )
