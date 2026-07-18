from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

from app.training.tool_use.generation import (
    DatasetGenerationResult,
    RolloutGenerationConfig,
    generate_dataset,
    generate_record,
)
from app.training.tool_use.protocol import (
    AssistantGroundingFinding,
    AssistantStepEnvelope,
    CalculateCall,
    FinalResponseStep,
    FinalResponseStepEnvelope,
    GeneratedUserTurn,
    GeneratedUserTurnEnvelope,
    GroundingVerdict,
    RecordGroundingAssessment,
    RecordGroundingAssessmentEnvelope,
    RecordQualityAssessment,
    RecordQualityAssessmentEnvelope,
    RecordQualityIssue,
    SearchCall,
    SyntheticSearchResult,
    SyntheticSearchResultEnvelope,
    TeacherChatMessage,
    TeacherSamplingMode,
    ToolActionStep,
    UserTurnScopeAssessment,
    UserTurnScopeAssessmentEnvelope,
)
from app.training.tool_use.scenario import (
    AssistantResponseMode,
    AssistantTurnPlan,
    FollowUpKind,
    PlannedToolStep,
    ScenarioSpec,
    ToolName,
    UtteranceForm,
)
from app.training.tool_use.schema import (
    AssistantMessage,
    DatasetSplit,
    LengthBand,
    SpeechStyle,
    SuccessOutcome,
    ToolDialogueRecord,
    ToolResultMessage,
    ToolUseBaseModel,
)
from app.training.tool_use.vllm_client import StructuredGenerationResult

GeneratedValue = TypeVar("GeneratedValue", bound=ToolUseBaseModel)
TIME_REFERENCE = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


class ScriptedGenerator:
    def __init__(self, values: Sequence[ToolUseBaseModel]) -> None:
        self.values = list(values)
        self.requests: list[tuple[TeacherChatMessage, ...]] = []

    async def generate(
        self,
        messages: Sequence[TeacherChatMessage],
        response_type: type[GeneratedValue],
        random_seed: int,
        sampling_mode: TeacherSamplingMode = TeacherSamplingMode.CREATIVE,
    ) -> StructuredGenerationResult[GeneratedValue]:
        del random_seed
        del sampling_mode
        self.requests.append(tuple(messages))
        if not self.values:
            raise AssertionError("Scripted generator received an unexpected request.")
        value = self.values.pop(0)
        if type(value) is not response_type:
            raise AssertionError(
                f"Expected request for {type(value).__name__}, received {response_type.__name__}."
            )
        return StructuredGenerationResult[GeneratedValue](
            value=cast(GeneratedValue, value),
            prompt_tokens=10,
            completion_tokens=5,
        )


def two_call_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="scenario-test-001",
        random_seed=41,
        family="lookup_then_calculate",
        topic="a current ticket price",
        length_band=LengthBand.MEDIUM,
        speech_style=SpeechStyle.CASUAL,
        turns=(
            AssistantTurnPlan(
                user_instruction="Ask for a current price and then double it.",
                tool_steps=(
                    PlannedToolStep(tool_name=ToolName.SEARCH),
                    PlannedToolStep(tool_name=ToolName.CALCULATE),
                ),
                follow_up_kind=FollowUpKind.NONE,
                utterance_form=UtteranceForm.REQUEST,
            ),
            AssistantTurnPlan(
                user_instruction="Ask a pronoun-based follow-up.",
                tool_steps=(),
                follow_up_kind=FollowUpKind.PRONOUN,
                utterance_form=UtteranceForm.FRAGMENT,
            ),
        ),
        split=DatasetSplit.TRAIN,
        leakage_group_id="lookup-price-search-calculate",
    )


def calculate_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="scenario-test-calculate",
        random_seed=53,
        family="precise_arithmetic",
        topic="basic arithmetic",
        length_band=LengthBand.SHORT,
        speech_style=SpeechStyle.CASUAL,
        turns=(
            AssistantTurnPlan(
                user_instruction="Ask for twelve times eight.",
                tool_steps=(PlannedToolStep(tool_name=ToolName.CALCULATE),),
                follow_up_kind=FollowUpKind.NONE,
                utterance_form=UtteranceForm.REQUEST,
            ),
        ),
        split=DatasetSplit.TRAIN,
        leakage_group_id="calculate-test",
    )


def clarified_search_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="scenario-test-clarified-search",
        random_seed=61,
        family="search_clarify_then_call",
        topic="a local public event",
        length_band=LengthBand.MEDIUM,
        speech_style=SpeechStyle.CASUAL,
        turns=(
            AssistantTurnPlan(
                user_instruction="Ask for a local event without naming the city.",
                tool_steps=(),
                follow_up_kind=FollowUpKind.NONE,
                utterance_form=UtteranceForm.CONTEXT_FIRST,
                response_mode=AssistantResponseMode.CLARIFY_SEARCH,
            ),
            AssistantTurnPlan(
                user_instruction="Clarify that the city is Berlin.",
                tool_steps=(),
                follow_up_kind=FollowUpKind.CLARIFICATION,
                utterance_form=UtteranceForm.FRAGMENT,
                response_mode=AssistantResponseMode.CONFIRM_SEARCH,
            ),
            AssistantTurnPlan(
                user_instruction="Confirm the proposed lookup.",
                tool_steps=(PlannedToolStep(tool_name=ToolName.SEARCH),),
                follow_up_kind=FollowUpKind.DELAYED_REQUEST,
                utterance_form=UtteranceForm.REACTION,
            ),
        ),
        split=DatasetSplit.TRAIN,
        leakage_group_id="clarified-search-test",
    )


def scripted_two_call_values() -> tuple[ToolUseBaseModel, ...]:
    return (
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="what's the current ticket price and double it",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        accepted_user_scope((ToolName.SEARCH, ToolName.CALCULATE)),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="I'll check the current price.",
                call=SearchCall(query="current example ticket price"),
            )
        ),
        SyntheticSearchResultEnvelope(
            result=SyntheticSearchResult(content="The current ticket price is 24 euros.")
        ),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="It's 24 euros. I'll double that.",
                call=CalculateCall(expression="24 * 2"),
            )
        ),
        FinalResponseStepEnvelope(
            step=FinalResponseStep(audible_text="Twice the current price is 48 euros.")
        ),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="and does that include fees",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        accepted_user_scope(()),
        FinalResponseStepEnvelope(
            step=FinalResponseStep(
                audible_text="The search result didn't say whether fees were included."
            )
        ),
        accepted_quality_assessment(),
        accepted_grounding_assessment(
            "scenario-test-001-message-2",
            "scenario-test-001-message-4",
            "scenario-test-001-message-6",
            "scenario-test-001-message-8",
        ),
    )


def accepted_user_scope(
    required_tools: tuple[ToolName, ...],
) -> UserTurnScopeAssessmentEnvelope:
    return UserTurnScopeAssessmentEnvelope(
        assessment=UserTurnScopeAssessment(
            required_tools=required_tools,
            scope_is_clear=True,
            request_is_supported=True,
            repeats_prior_user_turn=False,
            voice_style_is_natural=True,
        )
    )


def user_scope(
    required_tools: tuple[ToolName, ...],
    scope_is_clear: bool,
) -> UserTurnScopeAssessmentEnvelope:
    return UserTurnScopeAssessmentEnvelope(
        assessment=UserTurnScopeAssessment(
            required_tools=required_tools,
            scope_is_clear=scope_is_clear,
            request_is_supported=True,
            repeats_prior_user_turn=False,
            voice_style_is_natural=True,
        )
    )


def accepted_quality_assessment() -> RecordQualityAssessmentEnvelope:
    return RecordQualityAssessmentEnvelope(
        assessment=RecordQualityAssessment(
            tool_plan_is_sufficient=True,
            no_premature_tool_results=True,
            assistant_claims_are_grounded=True,
            tool_results_are_used_correctly=True,
            user_turns_are_distinct=True,
            sequential_calls_are_justified=True,
            no_unsupported_action_commitments=True,
            conversation_is_coherent=True,
            voice_style_is_natural=True,
            tone_is_appropriate=True,
            issues=(),
        )
    )


def accepted_grounding_assessment(
    *message_ids: str,
) -> RecordGroundingAssessmentEnvelope:
    return RecordGroundingAssessmentEnvelope(
        assessment=RecordGroundingAssessment(
            findings=tuple(
                AssistantGroundingFinding(
                    message_id=message_id,
                    verdict=GroundingVerdict.GROUNDED,
                    unsupported_excerpts=(),
                )
                for message_id in message_ids
            )
        )
    )


def rejected_grounding_assessment(
    *message_ids: str,
    unsupported_message_id: str,
    unsupported_excerpt: str,
) -> RecordGroundingAssessmentEnvelope:
    return RecordGroundingAssessmentEnvelope(
        assessment=RecordGroundingAssessment(
            findings=tuple(
                AssistantGroundingFinding(
                    message_id=message_id,
                    verdict=(
                        GroundingVerdict.UNSUPPORTED
                        if message_id == unsupported_message_id
                        else GroundingVerdict.GROUNDED
                    ),
                    unsupported_excerpts=(
                        (unsupported_excerpt,) if message_id == unsupported_message_id else ()
                    ),
                )
                for message_id in message_ids
            )
        )
    )


def rejected_quality_assessment(
    issue: RecordQualityIssue,
) -> RecordQualityAssessmentEnvelope:
    return RecordQualityAssessmentEnvelope(
        assessment=RecordQualityAssessment(
            tool_plan_is_sufficient=issue is not RecordQualityIssue.UNPLANNED_TOOL_NEED,
            no_premature_tool_results=issue is not RecordQualityIssue.PREMATURE_TOOL_RESULT,
            assistant_claims_are_grounded=(
                issue is not RecordQualityIssue.UNGROUNDED_ASSISTANT_CLAIM
            ),
            tool_results_are_used_correctly=(issue is not RecordQualityIssue.INCORRECT_RESULT_USE),
            user_turns_are_distinct=issue is not RecordQualityIssue.DUPLICATE_USER_TURN,
            sequential_calls_are_justified=issue is not RecordQualityIssue.REDUNDANT_TOOL_CALL,
            no_unsupported_action_commitments=(issue is not RecordQualityIssue.UNSUPPORTED_ACTION),
            conversation_is_coherent=issue is not RecordQualityIssue.INCOHERENT_CONVERSATION,
            voice_style_is_natural=issue is not RecordQualityIssue.UNNATURAL_VOICE_STYLE,
            tone_is_appropriate=issue is not RecordQualityIssue.INAPPROPRIATE_TONE,
            issues=(issue,),
        )
    )


def test_staged_rollout_appends_each_call_and_result_before_continuing() -> None:
    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        generator = ScriptedGenerator(scripted_two_call_values())
        generated = await generate_record(
            scenario=two_call_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())
    ToolDialogueRecord.model_validate_json(record.model_dump_json())
    assert request_count == 11
    assert not generator.values
    assert "Requested utterance form: request" in generator.requests[0][1].content
    assert "The current ticket price is 24 euros." in generator.requests[4][1].content
    assert "This is a sequential call." in generator.requests[4][1].content

    assistant_calls = tuple(
        message
        for message in record.messages
        if isinstance(message, AssistantMessage) and message.tool_calls
    )
    tool_results = tuple(
        message for message in record.messages if isinstance(message, ToolResultMessage)
    )
    assert tuple(call.tool_calls[0].tool_name for call in assistant_calls) == (
        "search",
        "calculate",
    )
    assert tuple(result.call_id for result in tool_results) == (
        "scenario-test-001-call-1",
        "scenario-test-001-call-2",
    )
    assert isinstance(tool_results[0].outcome, SuccessOutcome)
    assert tool_results[0].outcome.content == "The current ticket price is 24 euros."
    assert isinstance(tool_results[1].outcome, SuccessOutcome)
    assert tool_results[1].outcome.content == "48"
    assert "Audit this completed candidate record" in generator.requests[-2][1].content
    assert "Audit the assistant messages" in generator.requests[-1][1].content


def test_clarified_search_waits_for_confirmation_and_uses_resolved_query() -> None:
    values: tuple[ToolUseBaseModel, ...] = (
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="Anything interesting happening nearby this weekend?",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        user_scope((ToolName.SEARCH, ToolName.SEARCH), scope_is_clear=False),
        FinalResponseStepEnvelope(
            step=FinalResponseStep(audible_text="Which city should I check?")
        ),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="Berlin, preferably something outdoors.",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        user_scope((ToolName.SEARCH,), scope_is_clear=True),
        FinalResponseStepEnvelope(
            step=FinalResponseStep(audible_text="Outdoor events in Berlin this weekend—search now?")
        ),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="Yeah, go for it.",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        accepted_user_scope((ToolName.SEARCH,)),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="Alright, looking that up.",
                call=SearchCall(query="outdoor events in Berlin this weekend"),
            )
        ),
        SyntheticSearchResultEnvelope(
            result=SyntheticSearchResult(
                content="Berlin has an outdoor film screening in Tempelhof this Saturday."
            )
        ),
        FinalResponseStepEnvelope(
            step=FinalResponseStep(
                audible_text=("There’s an outdoor film screening in Tempelhof this Saturday.")
            )
        ),
        accepted_quality_assessment(),
        accepted_grounding_assessment(
            "scenario-test-clarified-search-message-2",
            "scenario-test-clarified-search-message-4",
            "scenario-test-clarified-search-message-6",
            "scenario-test-clarified-search-message-8",
        ),
    )

    async def exercise() -> ToolDialogueRecord:
        generated = await generate_record(
            scenario=clarified_search_scenario(),
            generator=ScriptedGenerator(values),
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
            ),
        )
        return generated.record

    record = asyncio.run(exercise())
    assistant_messages = tuple(
        message for message in record.messages if isinstance(message, AssistantMessage)
    )

    assert not assistant_messages[0].tool_calls
    assert not assistant_messages[1].tool_calls
    assert assistant_messages[2].tool_calls[0].tool_name == "search"
    assert (
        assistant_messages[2].tool_calls[0].arguments.fields[0].value.value
        == "outdoor events in Berlin this weekend"
    )


def test_record_quality_rejection_regenerates_the_entire_candidate() -> None:
    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        first_candidate = scripted_two_call_values()[:-2]
        second_candidate = scripted_two_call_values()[:-2]
        generator = ScriptedGenerator(
            (
                *first_candidate,
                rejected_quality_assessment(RecordQualityIssue.UNGROUNDED_ASSISTANT_CLAIM),
                *second_candidate,
                accepted_quality_assessment(),
                accepted_grounding_assessment(
                    "scenario-test-001-message-2",
                    "scenario-test-001-message-4",
                    "scenario-test-001-message-6",
                    "scenario-test-001-message-8",
                ),
            )
        )
        generated = await generate_record(
            scenario=two_call_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert record.record_id == "scenario-test-001"
    assert request_count == 21
    assert not generator.values


def test_grounding_rejection_regenerates_the_entire_candidate() -> None:
    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        first_candidate = scripted_two_call_values()[:-2]
        second_candidate = scripted_two_call_values()[:-2]
        message_ids = (
            "scenario-test-001-message-2",
            "scenario-test-001-message-4",
            "scenario-test-001-message-6",
            "scenario-test-001-message-8",
        )
        generator = ScriptedGenerator(
            (
                *first_candidate,
                accepted_quality_assessment(),
                rejected_grounding_assessment(
                    *message_ids,
                    unsupported_message_id="scenario-test-001-message-8",
                    unsupported_excerpt="fees are definitely included",
                ),
                *second_candidate,
                accepted_quality_assessment(),
                accepted_grounding_assessment(*message_ids),
            )
        )
        generated = await generate_record(
            scenario=two_call_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert record.record_id == "scenario-test-001"
    assert request_count == 22
    assert not generator.values


def test_calculate_bridge_cannot_reveal_the_pending_result() -> None:
    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        user_turn = GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="quick one, twelve times eight",
                speech_style=SpeechStyle.CASUAL,
            )
        )
        final_response = FinalResponseStepEnvelope(step=FinalResponseStep(audible_text="It's 96."))
        generator = ScriptedGenerator(
            (
                user_turn,
                accepted_user_scope((ToolName.CALCULATE,)),
                AssistantStepEnvelope(
                    step=ToolActionStep(
                        audible_text="That's 96.",
                        call=CalculateCall(expression="12 * 8"),
                    )
                ),
                final_response,
                user_turn,
                accepted_user_scope((ToolName.CALCULATE,)),
                AssistantStepEnvelope(
                    step=ToolActionStep(
                        audible_text="One sec.",
                        call=CalculateCall(expression="12 * 8"),
                    )
                ),
                final_response,
                accepted_quality_assessment(),
                accepted_grounding_assessment(
                    "scenario-test-calculate-message-2",
                    "scenario-test-calculate-message-4",
                ),
            )
        )
        generated = await generate_record(
            scenario=calculate_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert request_count == 10
    assert record.messages[2].audible_text == "One sec."
    assert not generator.values


def test_spoken_turns_over_one_hundred_words_are_regenerated() -> None:
    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        valid_user_turn = GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="quick one, twelve times eight",
                speech_style=SpeechStyle.CASUAL,
            )
        )
        generator = ScriptedGenerator(
            (
                GeneratedUserTurnEnvelope(
                    turn=GeneratedUserTurn(
                        text=" ".join("a" for _ in range(101)),
                        speech_style=SpeechStyle.CASUAL,
                    )
                ),
                valid_user_turn,
                accepted_user_scope((ToolName.CALCULATE,)),
                AssistantStepEnvelope(
                    step=ToolActionStep(
                        audible_text="One sec.",
                        call=CalculateCall(expression="12 * 8"),
                    )
                ),
                FinalResponseStepEnvelope(step=FinalResponseStep(audible_text="It's 96.")),
                accepted_quality_assessment(),
                accepted_grounding_assessment(
                    "scenario-test-calculate-message-2",
                    "scenario-test-calculate-message-4",
                ),
            )
        )
        generated = await generate_record(
            scenario=calculate_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert request_count == 7
    assert record.messages[1].text == "quick one, twelve times eight"
    assert not generator.values


def test_calculate_response_must_preserve_the_exact_result() -> None:
    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        user_turn = GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="quick one, twelve times eight",
                speech_style=SpeechStyle.CASUAL,
            )
        )
        tool_step = AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="One sec.",
                call=CalculateCall(expression="12 * 8"),
            )
        )
        generator = ScriptedGenerator(
            (
                user_turn,
                accepted_user_scope((ToolName.CALCULATE,)),
                tool_step,
                FinalResponseStepEnvelope(
                    step=FinalResponseStep(audible_text="That's about a hundred.")
                ),
                user_turn,
                accepted_user_scope((ToolName.CALCULATE,)),
                tool_step,
                FinalResponseStepEnvelope(
                    step=FinalResponseStep(audible_text="That's exactly 96.")
                ),
                accepted_quality_assessment(),
                accepted_grounding_assessment(
                    "scenario-test-calculate-message-2",
                    "scenario-test-calculate-message-4",
                ),
            )
        )
        generated = await generate_record(
            scenario=calculate_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert request_count == 10
    assert record.messages[4].audible_text == "That's exactly 96."
    assert not generator.values


def test_user_scope_mismatch_regenerates_before_assistant_generation() -> None:
    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        generator = ScriptedGenerator(
            (
                GeneratedUserTurnEnvelope(
                    turn=GeneratedUserTurn(
                        text="what time is it",
                        speech_style=SpeechStyle.CASUAL,
                    )
                ),
                accepted_user_scope((ToolName.GET_TIME,)),
                GeneratedUserTurnEnvelope(
                    turn=GeneratedUserTurn(
                        text="quick one, twelve times eight",
                        speech_style=SpeechStyle.CASUAL,
                    )
                ),
                accepted_user_scope((ToolName.CALCULATE,)),
                AssistantStepEnvelope(
                    step=ToolActionStep(
                        audible_text="One sec.",
                        call=CalculateCall(expression="12 * 8"),
                    )
                ),
                FinalResponseStepEnvelope(step=FinalResponseStep(audible_text="It's 96.")),
                accepted_quality_assessment(),
                accepted_grounding_assessment(
                    "scenario-test-calculate-message-2",
                    "scenario-test-calculate-message-4",
                ),
            )
        )
        generated = await generate_record(
            scenario=calculate_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert request_count == 8
    assert record.messages[1].text == "quick one, twelve times eight"
    assert not generator.values


def test_dataset_generation_is_append_only_and_resumable(tmp_path: Path) -> None:
    async def exercise() -> tuple[
        DatasetGenerationResult,
        DatasetGenerationResult,
        ScriptedGenerator,
    ]:
        scenarios = (two_call_scenario(),)
        config = RolloutGenerationConfig(
            model_identifier="Qwen/Qwen3.6-27B-FP8",
            model_revision="test-revision",
            quantization="fp8",
            time_reference=TIME_REFERENCE,
        )
        first_generator = ScriptedGenerator(scripted_two_call_values())
        first = await generate_dataset(
            scenarios=scenarios,
            generator=first_generator,
            config=config,
            output_path=output_path,
            failure_path=failure_path,
            manifest_path=manifest_path,
        )
        second_generator = ScriptedGenerator(())
        second = await generate_dataset(
            scenarios=scenarios,
            generator=second_generator,
            config=config,
            output_path=output_path,
            failure_path=failure_path,
            manifest_path=manifest_path,
        )
        return first, second, second_generator

    output_path = tmp_path / "records.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    manifest_path = tmp_path / "manifests.jsonl"
    first, second, second_generator = asyncio.run(exercise())
    assert first.manifest.accepted_records == 1
    assert second.manifest.previously_completed == 1
    assert second.manifest.accepted_records == 0
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 2
    assert not second_generator.requests
