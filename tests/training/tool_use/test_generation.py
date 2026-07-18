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
    AssistantStepEnvelope,
    CalculateCall,
    FinalResponseStep,
    GeneratedUserTurn,
    GeneratedUserTurnEnvelope,
    SearchCall,
    SyntheticSearchResult,
    SyntheticSearchResultEnvelope,
    TeacherChatMessage,
    ToolActionStep,
)
from app.training.tool_use.scenario import (
    AssistantTurnPlan,
    FollowUpKind,
    PlannedToolStep,
    ScenarioSpec,
    ToolName,
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
    ) -> StructuredGenerationResult[GeneratedValue]:
        del random_seed
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
            ),
            AssistantTurnPlan(
                user_instruction="Ask a pronoun-based follow-up.",
                tool_steps=(),
                follow_up_kind=FollowUpKind.PRONOUN,
            ),
        ),
        split=DatasetSplit.TRAIN,
        leakage_group_id="lookup-price-search-calculate",
    )


def scripted_two_call_values() -> tuple[ToolUseBaseModel, ...]:
    return (
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="what's the current ticket price and double it",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
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
        AssistantStepEnvelope(
            step=FinalResponseStep(audible_text="Twice the current price is 48 euros.")
        ),
        GeneratedUserTurnEnvelope(
            turn=GeneratedUserTurn(
                text="and does that include fees",
                speech_style=SpeechStyle.CASUAL,
            )
        ),
        AssistantStepEnvelope(
            step=FinalResponseStep(
                audible_text="The search result didn't say whether fees were included."
            )
        ),
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
    assert request_count == 7
    assert not generator.values
    assert "The current ticket price is 24 euros." in generator.requests[3][1].content

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
