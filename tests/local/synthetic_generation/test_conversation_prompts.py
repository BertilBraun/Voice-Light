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
    EnglishConversationContentDraft,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    HoldUserPrompt,
    MicroBackchannel,
    NonFloorFeedbackUserPrompt,
    PerceivedAge,
    SegmentDelivery,
    SpeakingPace,
    SpeechAct,
    TopicDomain,
    VocalPitch,
    VocalWeight,
    assemble_conversation_prompt_plan,
    conversation_generation_instruction,
    qwen_voice_instruction,
    representative_conversation_briefs,
    validate_conversation_prompt_set_id,
)
from app.local.synthetic_generation.generate_conversation_prompts import _repair_instruction


def test_representative_briefs_are_deterministic_and_cover_dimensions() -> None:
    briefs = representative_conversation_briefs(count=10, seed=41)

    assert briefs == representative_conversation_briefs(count=10, seed=41)
    assert len({brief.domain for brief in briefs}) == 10
    assert {brief.pace for brief in briefs} == set(SpeakingPace)
    assert {brief.affect for brief in briefs} == set(Affect)
    assert {condition for brief in briefs for condition in brief.required_conditions} == {
        "completion",
        "hold",
        "non_floor_feedback",
        "response_floor_claim",
        "interruption_floor_claim",
    }


@pytest.mark.parametrize("count", [4, 11])
def test_representative_briefs_reject_non_pilot_size(count: int) -> None:
    with pytest.raises(ValueError, match="5 to 10"):
        representative_conversation_briefs(count=count, seed=41)


def test_set_id_is_validated_before_model_generation() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        validate_conversation_prompt_set_id("invalid-pilot")


def test_generation_instruction_requests_only_natural_content() -> None:
    brief = representative_conversation_briefs(count=10, seed=41)[0]

    instruction = conversation_generation_instruction(brief)

    assert brief.domain.value in instruction
    assert "natural English content" in instruction
    assert "will add all IDs" in instruction
    assert "2-11" in instruction
    assert "45-90 words" in instruction
    assert "Do not output backchannels" in instruction
    assert '"opening_user_turn"' in instruction
    assert "hesitation pause near its midpoint" in instruction


def test_repair_instruction_returns_validation_errors_to_the_model() -> None:
    values = _content_draft().model_dump()
    values["brief_user_turn"] = "word"
    with pytest.raises(ValidationError) as captured:
        EnglishConversationContentDraft.model_validate(values)

    instruction = _repair_instruction(captured.value)

    assert "failed typed validation" in instruction
    assert "Brief user turn" in instruction
    assert "corrected complete JSON object" in instruction


def test_repair_instruction_returns_semantic_errors_to_the_model() -> None:
    instruction = _repair_instruction(
        ValueError("Generated conversation omitted conditions: hold.")
    )

    assert "omitted conditions: hold" in instruction
    assert "corrected complete JSON object" in instruction
    assert "word count" in instruction
    assert "Do not add IDs" in instruction


@pytest.mark.parametrize("target_duration_seconds", [59.9, 120.1])
def test_plan_rejects_source_duration_outside_contract(
    target_duration_seconds: float,
) -> None:
    values = _plan().model_dump()
    values["target_duration_seconds"] = target_duration_seconds

    with pytest.raises(ValidationError):
        EnglishConversationPromptPlan.model_validate(values)


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


def test_backchannel_rejects_propositional_phrase() -> None:
    values = _plan().model_dump()
    values["user_prompts"][1]["text"] = "that sounds right"

    with pytest.raises(ValidationError, match="enum"):
        EnglishConversationPromptPlan.model_validate(values)


def test_plan_requires_every_user_turn_length_band() -> None:
    values = _plan().model_dump()
    values["user_prompts"] = tuple(
        prompt for index, prompt in enumerate(values["user_prompts"]) if index != 2
    )

    with pytest.raises(ValidationError, match="brief, normal, and extended"):
        EnglishConversationPromptPlan.model_validate(values)


def test_llm_content_schema_excludes_structural_plan_fields() -> None:
    properties = EnglishConversationContentDraft.model_json_schema()["properties"]

    assert set(properties) == {
        "topic",
        "voice_reference_text",
        "opening_user_turn",
        "brief_user_turn",
        "normal_user_turn",
        "extended_user_turn",
        "assistant_turns",
    }
    schema_text = str(properties)
    for forbidden_name in (
        "plan_id",
        "sequence_index",
        "condition",
        "delivery",
        "duration",
        "accent",
    ):
        assert forbidden_name not in schema_text

    values = _content_draft().model_dump()
    values["plan_id"] = "model_owned_id"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnglishConversationContentDraft.model_validate(values)


def test_deterministic_assembly_produces_complete_valid_plan() -> None:
    brief = representative_conversation_briefs(count=10, seed=41)[0]

    plan = assemble_conversation_prompt_plan(_content_draft(), brief)

    sequence_indices = tuple(turn.sequence_index for turn in plan.assistant_turns) + tuple(
        prompt.sequence_index for prompt in plan.user_prompts
    )
    assert sorted(sequence_indices) == list(range(len(sequence_indices)))
    assert len(plan.assistant_turns) == 3
    assert plan.plan_id == brief.plan_id


def test_deterministic_assembly_extends_short_voice_reference_text() -> None:
    brief = representative_conversation_briefs(count=10, seed=41)[0]
    values = _content_draft().model_dump()
    values["voice_reference_text"] = "This neutral sentence provides a short reference for cloning."

    plan = assemble_conversation_prompt_plan(
        EnglishConversationContentDraft.model_validate(values), brief
    )

    assert 14 <= len(plan.voice_reference_text.split()) <= 24
    assert plan.voice_reference_text.startswith("This neutral sentence")
    assert plan.seed == brief.seed
    assert plan.domain is brief.domain
    assert 60.0 <= plan.target_duration_seconds <= 120.0
    assert {prompt.condition for prompt in plan.user_prompts}.issuperset(brief.required_conditions)


def test_assembly_is_deterministic_and_pilot_covers_all_conditions() -> None:
    briefs = representative_conversation_briefs(count=5, seed=41)
    plans = tuple(assemble_conversation_prompt_plan(_content_draft(), brief) for brief in briefs)

    assert plans == tuple(
        assemble_conversation_prompt_plan(_content_draft(), brief) for brief in briefs
    )
    assert {prompt.condition for plan in plans for prompt in plan.user_prompts} == {
        "completion",
        "hold",
        "non_floor_feedback",
        "response_floor_claim",
        "interruption_floor_claim",
    }
    for plan in plans:
        element_ids = (
            *(turn.turn_id for turn in plan.assistant_turns),
            *(prompt.unit_id for prompt in plan.user_prompts),
        )
        assert len(element_ids) == len(set(element_ids))


def test_duration_estimate_scales_with_content_and_stays_in_contract() -> None:
    brief = representative_conversation_briefs(count=5, seed=41)[0]
    short_plan = assemble_conversation_prompt_plan(_content_draft(), brief)
    values = _content_draft().model_dump()
    values["extended_user_turn"] = " ".join(["thoughtful"] * 90)
    values["opening_user_turn"] = " ".join(["opening"] * 44)
    values["normal_user_turn"] = " ".join(["detail"] * 44)
    values["assistant_turns"] = tuple(" ".join(["reply"] * 25) for _ in range(3))
    long_plan = assemble_conversation_prompt_plan(
        EnglishConversationContentDraft.model_validate(values), brief
    )

    assert short_plan.target_duration_seconds < long_plan.target_duration_seconds
    assert long_plan.target_duration_seconds <= 120.0


def test_plan_requires_short_exact_voice_reference_text() -> None:
    values = _plan().model_dump()
    values["voice_reference_text"] = "Too short."

    with pytest.raises(ValidationError, match="14 to 24 words"):
        EnglishConversationPromptPlan.model_validate(values)


def test_plan_accepts_english_typographic_punctuation() -> None:
    values = _plan().model_dump()
    values["user_prompts"][0]["text"] = (
        "I would start with the lights\u2014because I adjust them every evening "
        "and always forget the ideal setting."
    )

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
        target_duration_seconds=75.0,
        base_user_voice=BaseUserVoice(
            perceived_age=PerceivedAge.ADULT,
            accent=EnglishAccent.GENERAL_AMERICAN,
            pitch=VocalPitch.MEDIUM,
            vocal_weight=VocalWeight.MEDIUM,
        ),
        voice_reference_text=(
            "Every clear morning brings a fresh chance to notice something useful nearby during "
            "an ordinary walk."
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
                text=(
                    "I would start with the lights because I adjust them every evening and always "
                    "forget the ideal setting."
                ),
            ),
            NonFloorFeedbackUserPrompt(
                unit_id="user_2",
                sequence_index=2,
                delivery=delivery,
                text=MicroBackchannel.RIGHT,
                during_assistant_turn_id="assistant_1",
            ),
            HoldUserPrompt(
                unit_id="user_3",
                sequence_index=4,
                speech_act=SpeechAct.EXPLANATION,
                delivery=delivery,
                text_before_pause=(
                    "The morning routine could also help because the hallway gets surprisingly "
                    "dark before sunrise, and finding the switch while carrying coffee is awkward"
                ),
                text_after_pause=(
                    "but I would want to test several schedules, keep the weekends flexible, and "
                    "make sure the lights never wake anyone who decided to sleep late."
                ),
                pause_duration_seconds=0.8,
            ),
            CompletionUserPrompt(
                unit_id="user_4",
                sequence_index=3,
                speech_act=SpeechAct.ANSWER,
                delivery=delivery,
                text="That would save time every morning.",
            ),
        ),
    )


def _content_draft() -> EnglishConversationContentDraft:
    return EnglishConversationContentDraft(
        topic="Choosing a useful home automation routine",
        voice_reference_text=(
            "Every clear morning brings a fresh chance to notice something useful nearby during "
            "an ordinary walk."
        ),
        opening_user_turn=(
            "I started looking at simple home routines because the evening lighting is never "
            "quite right when I arrive."
        ),
        brief_user_turn="That sounds genuinely useful.",
        normal_user_turn=(
            "I would probably begin with one hallway sensor and test its schedule for a week "
            "before changing anything else."
        ),
        extended_user_turn=(
            "The real test would be whether the routine stays helpful on weekends, when everyone "
            "wakes at different times and moves through the house less predictably. I would keep "
            "a manual switch available, watch for false triggers, and ask everyone whether the "
            "automatic light feels convenient or merely intrusive before adding more rooms."
        ),
        assistant_turns=(
            "Which part of the evening routine causes the most frustration?",
            "Starting with one location should make the result easier to judge.",
            "How would you decide whether the experiment was successful?",
        ),
    )


def _provenance() -> ConversationPromptGeneratorProvenance:
    return ConversationPromptGeneratorProvenance(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="a" * 40,
        runtime_version="5.0.0",
        seed=41,
        requested_plan_count=10,
    )
