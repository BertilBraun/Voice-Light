from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.local.synthetic_generation.conversation_corpus_audit import audit_conversation_corpus
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
    MatchedPrefixAmbiguity,
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
    matched = tuple(
        brief.prefix_ambiguity
        for brief in briefs
        if isinstance(brief.prefix_ambiguity, MatchedPrefixAmbiguity)
    )
    assert len(matched) == 2
    assert matched[0].pair_id == matched[1].pair_id
    assert {item.outcome for item in matched} == {"completion", "continuation"}
    assert briefs[0].pace is briefs[1].pace
    assert briefs[0].affect is briefs[1].affect
    assert briefs[0].domain is not briefs[1].domain


@pytest.mark.parametrize("count", [4, 2_001])
def test_representative_briefs_reject_non_pilot_size(count: int) -> None:
    with pytest.raises(ValueError, match="5 to 2,000"):
        representative_conversation_briefs(count=count, seed=41)


def test_small_pilot_does_not_overrepresent_matched_ambiguity() -> None:
    briefs = representative_conversation_briefs(count=5, seed=41)

    assert all(brief.prefix_ambiguity.kind == "independent" for brief in briefs)


def test_corpus_briefs_allocate_twenty_percent_matched_ambiguity() -> None:
    briefs = representative_conversation_briefs(count=100, seed=41)

    matched = tuple(brief for brief in briefs if brief.prefix_ambiguity.kind == "matched")
    assert len(matched) == 20
    assert len({brief.prefix_ambiguity.pair_id for brief in matched}) == 10
    for offset in range(0, len(matched), 2):
        first, second = matched[offset : offset + 2]
        assert first.prefix_ambiguity.pair_id == second.prefix_ambiguity.pair_id
        assert first.pace is second.pace
        assert first.affect is second.affect


def test_prompt_only_corpus_audit_reports_duration_and_dimensions(tmp_path: Path) -> None:
    brief = representative_conversation_briefs(count=5, seed=41)[0]
    plan = assemble_conversation_prompt_plan(_content_draft(), brief)
    prompt_set = EnglishConversationPromptSet(
        set_id="audit_test",
        provenance=_provenance().model_copy(update={"requested_plan_count": 5}),
        plans=(plan,),
    )
    path = tmp_path / "prompts.json"
    path.write_text(prompt_set.model_dump_json(indent=2), encoding="utf-8")

    audit = audit_conversation_corpus(path)

    assert audit.plan_count == 1
    assert audit.planned_conversation_hours == plan.target_duration_seconds / 3600.0
    assert sum(item.count for item in audit.turn_lengths) == 4
    assert {item.value for item in audit.turn_lengths} == {"brief", "normal", "extended"}
    assert audit.measured_conversation_hours is None
    assert audit.quality_flags == ()


def test_set_id_is_validated_before_model_generation() -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        validate_conversation_prompt_set_id("invalid-pilot")


def test_generation_instruction_requests_only_natural_content() -> None:
    brief = representative_conversation_briefs(count=10, seed=41)[0]

    instruction = conversation_generation_instruction(brief)

    assert brief.domain.value in instruction
    assert "natural English content" in instruction
    assert "will add all IDs" in instruction
    assert "assistant never initiates" in instruction
    assert "initial conversational direction" in instruction
    assert "not exact word-count requirements" in instruction
    assert "Do not output backchannels" in instruction
    assert '"opening_user_turn"' in instruction
    assert "naturally invites a hesitation" in instruction
    assert "distribution-matched ambiguity pair" in instruction


def test_extended_turn_rotates_across_conversation_positions() -> None:
    draft = _content_draft()
    plans = tuple(
        assemble_conversation_prompt_plan(draft, brief)
        for brief in representative_conversation_briefs(count=10, seed=41)[:6]
    )

    positions = []
    for plan in plans:
        ordered_floor_turns = tuple(
            prompt
            for prompt in sorted(plan.user_prompts, key=lambda prompt: prompt.sequence_index)
            if prompt.text
            in {
                draft.brief_user_turn,
                draft.normal_user_turn,
                draft.extended_user_turn,
            }
        )
        positions.append(
            next(
                index
                for index, prompt in enumerate(ordered_floor_turns)
                if prompt.text == draft.extended_user_turn
            )
        )

    assert set(positions) == {0, 1, 2}


@pytest.mark.parametrize(
    "extended_user_turn",
    [
        " ".join(["single"] * 45) + ".",
        " ".join(["long"] * 100) + ".",
        "A concise answer can still be useful.",
    ],
)
def test_extended_turn_length_is_guidance(extended_user_turn: str) -> None:
    values = _content_draft().model_dump()
    values["extended_user_turn"] = extended_user_turn

    draft = EnglishConversationContentDraft.model_validate(values)

    assert draft.extended_user_turn == extended_user_turn


def test_matched_ambiguity_branches_use_the_same_turn_band() -> None:
    completion_brief, continuation_brief = representative_conversation_briefs(count=10, seed=41)[:2]

    completion_plan = assemble_conversation_prompt_plan(_content_draft(), completion_brief)
    continuation_plan = assemble_conversation_prompt_plan(_content_draft(), continuation_brief)
    completion_prompt = next(
        prompt
        for prompt in completion_plan.user_prompts
        if prompt.text == _content_draft().normal_user_turn
    )
    continuation_prompt = next(
        prompt
        for prompt in continuation_plan.user_prompts
        if prompt.text == _content_draft().normal_user_turn
    )

    assert completion_prompt.condition == "completion"
    assert continuation_prompt.condition == "hold"
    assert completion_prompt.speech_act is continuation_prompt.speech_act


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


def test_plan_rejects_assistant_initiated_conversation() -> None:
    values = _plan().model_dump()
    values["assistant_turns"][0]["sequence_index"] = 0
    values["user_prompts"][0]["sequence_index"] = 1

    with pytest.raises(ValidationError, match="begin with a floor-owning user turn"):
        EnglishConversationPromptPlan.model_validate(values)


def test_plan_rejects_assistant_turn_after_backchannel() -> None:
    values = _plan().model_dump()
    values["assistant_turns"][0]["sequence_index"] = 3
    values["user_prompts"][3]["sequence_index"] = 1

    with pytest.raises(ValidationError, match="directly reply to a user turn"):
        EnglishConversationPromptPlan.model_validate(values)


def test_plan_has_no_model_generated_language_field() -> None:
    values = _plan().model_dump()
    values["language"] = "es"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnglishConversationPromptPlan.model_validate(values)


def test_backchannel_rejects_propositional_phrase() -> None:
    values = _plan().model_dump()
    values["user_prompts"][1]["text"] = "that sounds right"

    with pytest.raises(ValidationError, match="enum"):
        EnglishConversationPromptPlan.model_validate(values)


def test_plan_accepts_natural_turn_lengths_outside_target_bands() -> None:
    values = _plan().model_dump()
    values["user_prompts"][0]["text"] = "No."
    values["user_prompts"][2]["text"] = " ".join(["detail"] * 120) + "."

    plan = EnglishConversationPromptPlan.model_validate(values)

    assert plan.user_prompts[0].text == "No."


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


def test_deterministic_assembly_preserves_short_voice_reference_text() -> None:
    brief = representative_conversation_briefs(count=10, seed=41)[0]
    values = _content_draft().model_dump()
    values["voice_reference_text"] = "This neutral sentence provides a short reference for cloning."

    plan = assemble_conversation_prompt_plan(
        EnglishConversationContentDraft.model_validate(values), brief
    )

    assert plan.voice_reference_text == values["voice_reference_text"]
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
    values["extended_user_turn"] = " ".join(["thoughtful"] * 30) + ". "
    values["extended_user_turn"] += " ".join(["considered"] * 30) + ". "
    values["extended_user_turn"] += " ".join(["carefully"] * 30) + "."
    values["opening_user_turn"] = " ".join(["opening"] * 44)
    values["normal_user_turn"] = " ".join(["detail"] * 44)
    values["assistant_turns"] = tuple(" ".join(["reply"] * 25) for _ in range(3))
    long_plan = assemble_conversation_prompt_plan(
        EnglishConversationContentDraft.model_validate(values), brief
    )

    assert short_plan.target_duration_seconds < long_plan.target_duration_seconds
    assert long_plan.target_duration_seconds <= 120.0


def test_plan_accepts_short_voice_reference_text() -> None:
    values = _plan().model_dump()
    values["voice_reference_text"] = "Too short."

    plan = EnglishConversationPromptPlan.model_validate(values)

    assert plan.voice_reference_text == "Too short."


def test_plan_accepts_english_typographic_punctuation() -> None:
    values = _plan().model_dump()
    values["user_prompts"][0]["text"] = (
        "I would start with the lights\u2014because I adjust them every evening "
        "and always forget the ideal setting."
    )

    plan = EnglishConversationPromptPlan.model_validate(values)

    assert "\u2014" in plan.user_prompts[0].text


def test_prompt_set_accepts_repeated_everyday_phrases() -> None:
    first = _plan()
    second = first.model_copy(update={"plan_id": "pilot_second"})

    prompt_set = EnglishConversationPromptSet(
        set_id="pilot",
        provenance=_provenance(),
        plans=(first, second),
    )

    assert len(prompt_set.plans) == 2


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
                text=(
                    "The morning routine could also help because the hallway gets surprisingly "
                    "dark before sunrise, and finding the switch while carrying coffee is awkward, "
                    "but I would want to test several schedules, keep the weekends flexible, and "
                    "make sure the lights never wake anyone who decided to sleep late."
                ),
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
