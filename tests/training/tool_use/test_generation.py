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
from app.training.tool_use.prompts import (
    teacher_led_assistant_step_messages,
    teacher_led_user_turn_messages,
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
    ScenarioGenerationMode,
    ScenarioSpec,
    TeacherLedConversationKind,
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


def teacher_led_scenario() -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="scenario-test-teacher-led",
        random_seed=67,
        family="teacher_led_tool_use",
        topic=(
            "Compare the population of two cities and preserve the comparison metric in "
            "follow-up questions."
        ),
        length_band=LengthBand.LONG,
        speech_style=SpeechStyle.CASUAL,
        turns=tuple(
            AssistantTurnPlan(
                user_instruction="Continue the same conversation naturally.",
                tool_steps=(),
                follow_up_kind=FollowUpKind.NONE,
                utterance_form=UtteranceForm.CONTEXT_FIRST,
            )
            for _ in range(4)
        ),
        split=DatasetSplit.TRAIN,
        leakage_group_id="teacher-led-city-population",
        generation_mode=ScenarioGenerationMode.TEACHER_LED,
        teacher_led_kind=TeacherLedConversationKind.TOOL_RICH,
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
                tool_steps=(PlannedToolStep(tool_name=ToolName.SEARCH),),
                follow_up_kind=FollowUpKind.CLARIFICATION,
                utterance_form=UtteranceForm.FRAGMENT,
            ),
        ),
        split=DatasetSplit.TRAIN,
        leakage_group_id="clarified-search-test",
    )


def test_teacher_led_no_tool_prompts_keep_requests_answerable_from_context() -> None:
    scenario = teacher_led_scenario().model_copy(
        update={
            "family": "teacher_led_no_tool",
            "teacher_led_kind": TeacherLedConversationKind.NO_TOOL,
        }
    )

    first_user_prompt = teacher_led_user_turn_messages(
        scenario=scenario,
        turn_index=0,
        public_history=(),
    )[1].content
    assistant_prompt = teacher_led_assistant_step_messages(
        scenario=scenario,
        public_history=(),
    )[1].content

    assert "Supply every detail" in first_user_prompt
    assert "Do not request current time" in first_user_prompt
    assert "Normally give a final spoken response" in assistant_prompt
    assert "Do not call a tool merely because one is available" in assistant_prompt


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
            advice_is_constructive=True,
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
            advice_is_constructive=issue is not RecordQualityIssue.COUNTERPRODUCTIVE_ADVICE,
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
                semantic_audits_enabled=True,
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


def test_staged_rollout_skips_subjective_semantic_audits() -> None:
    excluded_types = (
        UserTurnScopeAssessmentEnvelope,
        RecordQualityAssessmentEnvelope,
        RecordGroundingAssessmentEnvelope,
    )
    values = tuple(
        value for value in scripted_two_call_values() if type(value) not in excluded_types
    )

    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        generator = ScriptedGenerator(values)
        generated = await generate_record(
            scenario=two_call_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
                semantic_audits_enabled=False,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    ToolDialogueRecord.model_validate_json(record.model_dump_json())
    assert request_count == 7
    assert not generator.values
    assert not any(
        "Audit" in message.content for request in generator.requests for message in request
    )


def test_teacher_led_rollout_chooses_tools_and_preserves_causal_results() -> None:
    values: tuple[ToolUseBaseModel, ...] = (
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="How many people live in London these days?",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="I’ll check London’s latest figure.",
                call=SearchCall(query="current population of London"),
            )
        ),
        SyntheticSearchResultEnvelope(
            result=SyntheticSearchResult(content="London has 9.0 million residents.")
        ),
        AssistantStepEnvelope(
            step=FinalResponseStep(audible_text="London has about 9.0 million residents.")
        ),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="And how does that compare with New York?",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="I’ll get New York’s matching figure.",
                call=SearchCall(query="current population of New York City"),
            )
        ),
        SyntheticSearchResultEnvelope(
            result=SyntheticSearchResult(content="New York City has 8.3 million residents.")
        ),
        AssistantStepEnvelope(
            step=FinalResponseStep(
                audible_text="New York City has 8.3 million, so London is larger."
            )
        ),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="Roughly how many more people is that?",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="I’ll work out the difference.",
                call=CalculateCall(expression="9000000 - 8300000"),
            )
        ),
        AssistantStepEnvelope(
            step=FinalResponseStep(audible_text="London has roughly 700,000 more residents.")
        ),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="That’s closer than I expected.",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        AssistantStepEnvelope(
            step=FinalResponseStep(
                audible_text=(
                    "Yeah, they’re much closer in population than their reputations suggest."
                )
            )
        ),
    )

    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        generator = ScriptedGenerator(values)
        generated = await generate_record(
            scenario=teacher_led_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
                semantic_audits_enabled=False,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert request_count == 13
    assert not generator.values
    assert "Choose the action yourself" in generator.requests[1][1].content
    assert "London has about 9.0 million residents." in generator.requests[4][1].content
    assert '"kind": "tool_result"' not in generator.requests[4][1].content
    assert "New York City has 8.3 million residents." in generator.requests[7][1].content
    assert record.metadata.scenario.expected_tool_names == (
        "search",
        "search",
        "calculate",
    )
    assert record.metadata.scenario.tool_need == "multiple_required"
    assert record.metadata.scenario.user_turn_count == 4


def test_teacher_led_rollout_allows_more_than_two_dependent_calls() -> None:
    values: tuple[ToolUseBaseModel, ...] = (
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="Can you calculate three steps for me?",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="Starting with the first step.",
                call=CalculateCall(expression="2 + 2"),
            )
        ),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="Using that for the second step.",
                call=CalculateCall(expression="4 * 3"),
            )
        ),
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="And now the final calculation.",
                call=CalculateCall(expression="12 - 5"),
            )
        ),
        AssistantStepEnvelope(
            step=FinalResponseStep(audible_text="The three results are 4, 12, and 7.")
        ),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(text="Great, thanks.", speech_style=SpeechStyle.CASUAL)
        ),
        AssistantStepEnvelope(step=FinalResponseStep(audible_text="You’re welcome.")),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(text="That was all I needed.", speech_style=SpeechStyle.CASUAL)
        ),
        AssistantStepEnvelope(step=FinalResponseStep(audible_text="All set.")),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(text="Perfect.", speech_style=SpeechStyle.CASUAL)
        ),
        AssistantStepEnvelope(step=FinalResponseStep(audible_text="Perfect.")),
    )

    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord]:
        generator = ScriptedGenerator(values)
        generated = await generate_record(
            scenario=teacher_led_scenario(),
            generator=generator,
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
                time_reference=TIME_REFERENCE,
                semantic_audits_enabled=False,
            ),
        )
        return generator, generated.record

    generator, record = asyncio.run(exercise())

    assert not generator.values
    assert record.metadata.scenario.expected_tool_names[:3] == (
        "calculate",
        "calculate",
        "calculate",
    )
    assert "additional tool calls are available" not in generator.requests[1][1].content


def test_clarified_search_calls_automatically_after_resolving_query() -> None:
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
        AssistantStepEnvelope(
            step=ToolActionStep(
                audible_text="Checking outdoor events in Berlin.",
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
                semantic_audits_enabled=True,
            ),
        )
        return generated.record

    record = asyncio.run(exercise())
    assistant_messages = tuple(
        message for message in record.messages if isinstance(message, AssistantMessage)
    )

    assert not assistant_messages[0].tool_calls
    assert assistant_messages[1].tool_calls[0].tool_name == "search"
    assert (
        assistant_messages[1].tool_calls[0].arguments.fields[0].value.value
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
                semantic_audits_enabled=True,
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
                semantic_audits_enabled=True,
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
                semantic_audits_enabled=True,
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
                semantic_audits_enabled=True,
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
                semantic_audits_enabled=True,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert request_count == 10
    assert record.messages[4].audible_text == "That's exactly 96."
    assert not generator.values


def test_generation_without_audits_keeps_long_bridge_and_spoken_calculation() -> None:
    async def exercise() -> tuple[ScriptedGenerator, ToolDialogueRecord, int]:
        generator = ScriptedGenerator(
            (
                GeneratedUserTurnEnvelope(
                    turn=GeneratedUserTurn(
                        text="quick one, twelve times eight",
                        speech_style=SpeechStyle.CASUAL,
                    )
                ),
                AssistantStepEnvelope(
                    step=ToolActionStep(
                        audible_text="Sure, give me a moment while I work that out.",
                        call=CalculateCall(expression="12 * 8"),
                    )
                ),
                FinalResponseStepEnvelope(
                    step=FinalResponseStep(audible_text="That's ninety-six.")
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
                semantic_audits_enabled=False,
            ),
        )
        return generator, generated.record, generated.request_count

    generator, record, request_count = asyncio.run(exercise())

    assert request_count == 3
    assert record.messages[2].audible_text == "Sure, give me a moment while I work that out."
    assert record.messages[4].audible_text == "That's ninety-six."
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
                semantic_audits_enabled=True,
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
            semantic_audits_enabled=True,
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
