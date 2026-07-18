from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import AwareDatetime, Field

from app.training.tool_use.prompts import (
    PROMPT_REVISION,
    assistant_step_messages,
    synthetic_search_messages,
    user_turn_messages,
)
from app.training.tool_use.protocol import (
    AssistantStep,
    AssistantStepEnvelope,
    AssistantStepValidation,
    CalculateCall,
    FinalResponseStep,
    GeneratedToolCall,
    GeneratedUserTurnEnvelope,
    GetTimeCall,
    SearchCall,
    SyntheticSearchResultEnvelope,
    TeacherChatMessage,
    ToolActionStep,
    generated_call_tool_name,
)
from app.training.tool_use.scenario import (
    PlannedOutcome,
    PlannedToolStep,
    ScenarioSpec,
    ToolName,
)
from app.training.tool_use.schema import (
    AssistantMessage,
    AuditMetadata,
    ConversationMessage,
    EmptyOutcome,
    FailureOutcome,
    GenerationProvenance,
    NamedParameterSchema,
    ObjectParameterSchema,
    RecordMetadata,
    ScenarioMetadata,
    SpeechStyle,
    SplitMetadata,
    StringParameterSchema,
    SuccessOutcome,
    SystemMessage,
    TimeoutOutcome,
    ToolCall,
    ToolDefinition,
    ToolDialogueRecord,
    ToolLatency,
    ToolOutcome,
    ToolResultMessage,
    ToolUseBaseModel,
    UserMessage,
    append_record,
    canonical_json,
    completed_record_ids,
    contains_protocol_markup,
)
from app.training.tool_use.tools import calculate_expression, call_arguments, seeded_time
from app.training.tool_use.vllm_client import StructuredGenerationResult

GeneratedValue = TypeVar("GeneratedValue", bound=ToolUseBaseModel)


class StructuredGenerator(Protocol):
    async def generate(
        self,
        messages: Sequence[TeacherChatMessage],
        response_type: type[GeneratedValue],
        random_seed: int,
    ) -> StructuredGenerationResult[GeneratedValue]: ...


class RolloutGenerationConfig(ToolUseBaseModel):
    model_identifier: str
    model_revision: str
    quantization: str
    time_reference: AwareDatetime
    maximum_concurrency: int = Field(default=128, ge=1)
    maximum_semantic_attempts: int = Field(default=3, ge=1)
    bridge_word_limit: int = Field(default=8, ge=1)
    system_instruction: str = (
        "Speak naturally. Use supplied tools only when needed. Never invent a pending result."
    )


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class GeneratedRecord:
    record: ToolDialogueRecord
    usage: TokenUsage
    request_count: int


class GenerationFailure(ToolUseBaseModel):
    scenario_id: str
    error_type: str
    message: str
    recorded_at: str


class GenerationRunManifest(ToolUseBaseModel):
    schema_version: str = "voice-light.generation-run/v1"
    started_at: str
    completed_at: str
    model_identifier: str
    model_revision: str
    quantization: str
    time_reference: AwareDatetime
    prompt_revision: str
    planned_scenarios: int = Field(ge=0)
    previously_completed: int = Field(ge=0)
    accepted_records: int = Field(ge=0)
    failed_records: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    requests: int = Field(ge=0)


@dataclass(frozen=True)
class DatasetGenerationResult:
    manifest: GenerationRunManifest
    records: tuple[ToolDialogueRecord, ...]
    failures: tuple[GenerationFailure, ...]


class RecordHashPayload(ToolUseBaseModel):
    tools: tuple[ToolDefinition, ...]
    messages: tuple[ConversationMessage, ...]
    scenario_id: str


def runtime_tool_definitions() -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name=ToolName.SEARCH.value,
            description="Search for current or externally verifiable information.",
            latency=ToolLatency.LATENCY_BEARING,
            parameters=ObjectParameterSchema(
                description="Search arguments.",
                properties=(
                    NamedParameterSchema(
                        name="query",
                        required=True,
                        value_schema=StringParameterSchema(
                            description="A concise search query.",
                            min_length=1,
                            max_length=240,
                        ),
                    ),
                ),
            ),
            result_description="One concise search-result string.",
        ),
        ToolDefinition(
            name=ToolName.CALCULATE.value,
            description="Evaluate a basic numeric arithmetic expression.",
            latency=ToolLatency.IMMEDIATE,
            parameters=ObjectParameterSchema(
                description="Calculation arguments.",
                properties=(
                    NamedParameterSchema(
                        name="expression",
                        required=True,
                        value_schema=StringParameterSchema(
                            description="Numeric literals and basic arithmetic operators.",
                            min_length=1,
                            max_length=160,
                        ),
                    ),
                ),
            ),
            result_description="The exact calculation result as a string.",
        ),
        ToolDefinition(
            name=ToolName.GET_TIME.value,
            description="Get the current local date and time.",
            latency=ToolLatency.IMMEDIATE,
            parameters=ObjectParameterSchema(
                description="This tool takes no arguments.",
                properties=(),
            ),
            result_description="A local ISO 8601 timestamp with UTC offset.",
        ),
    )


async def generate_dataset(
    scenarios: Sequence[ScenarioSpec],
    generator: StructuredGenerator,
    config: RolloutGenerationConfig,
    output_path: Path,
    failure_path: Path,
    manifest_path: Path,
) -> DatasetGenerationResult:
    started_at = _utc_now()
    already_completed = completed_record_ids(output_path)
    pending_scenarios = tuple(
        scenario for scenario in scenarios if scenario.scenario_id not in already_completed
    )
    semaphore = asyncio.Semaphore(config.maximum_concurrency)

    async def generate_with_limit(scenario: ScenarioSpec) -> GeneratedRecord:
        async with semaphore:
            return await generate_record(scenario, generator, config)

    tasks = tuple(
        asyncio.create_task(generate_with_limit(scenario)) for scenario in pending_scenarios
    )
    task_scenarios = tuple(zip(tasks, pending_scenarios, strict=True))
    generated_records: list[ToolDialogueRecord] = []
    failures: list[GenerationFailure] = []
    usage = TokenUsage()
    request_count = 0
    for task, scenario in task_scenarios:
        try:
            generated = await task
        except Exception as error:
            failure = GenerationFailure(
                scenario_id=scenario.scenario_id,
                error_type=type(error).__name__,
                message=str(error),
                recorded_at=_utc_now(),
            )
            _append_failure(failure_path, failure)
            failures.append(failure)
            continue
        append_record(output_path, generated.record)
        generated_records.append(generated.record)
        usage = usage.add(generated.usage)
        request_count += generated.request_count

    manifest = GenerationRunManifest(
        started_at=started_at,
        completed_at=_utc_now(),
        model_identifier=config.model_identifier,
        model_revision=config.model_revision,
        quantization=config.quantization,
        time_reference=config.time_reference,
        prompt_revision=PROMPT_REVISION,
        planned_scenarios=len(scenarios),
        previously_completed=len(already_completed),
        accepted_records=len(generated_records),
        failed_records=len(failures),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        requests=request_count,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(manifest.model_dump_json())
        manifest_file.write("\n")
    return DatasetGenerationResult(
        manifest=manifest,
        records=tuple(generated_records),
        failures=tuple(failures),
    )


async def generate_record(
    scenario: ScenarioSpec,
    generator: StructuredGenerator,
    config: RolloutGenerationConfig,
) -> GeneratedRecord:
    tools = runtime_tool_definitions()
    messages: list[ConversationMessage] = [
        SystemMessage(
            message_id=f"{scenario.scenario_id}-system",
            text=config.system_instruction,
        )
    ]
    usage = TokenUsage()
    request_count = 0
    call_index = 0
    message_index = 0

    for turn_index, turn_plan in enumerate(scenario.turns):
        user_result, user_usage, user_requests = await _generate_validated(
            generator=generator,
            response_type=GeneratedUserTurnEnvelope,
            messages=user_turn_messages(scenario, turn_plan, messages),
            random_seed=_request_seed(scenario.random_seed, turn_index, 0),
            maximum_attempts=config.maximum_semantic_attempts,
            validator=lambda envelope: _validate_user_turn(
                envelope,
                scenario.speech_style,
            ),
        )
        usage = usage.add(user_usage)
        request_count += user_requests
        message_index += 1
        messages.append(
            UserMessage(
                message_id=f"{scenario.scenario_id}-message-{message_index}",
                text=user_result.turn.text,
                speech_style=scenario.speech_style,
            )
        )

        for step_index, planned_step in enumerate(turn_plan.tool_steps):
            assistant_result, assistant_usage, assistant_requests = await _generate_validated(
                generator=generator,
                response_type=AssistantStepEnvelope,
                messages=assistant_step_messages(
                    scenario,
                    messages,
                    planned_step.tool_name,
                ),
                random_seed=_request_seed(scenario.random_seed, turn_index, step_index + 1),
                maximum_attempts=config.maximum_semantic_attempts,
                validator=lambda envelope, expected=planned_step.tool_name: (
                    _validate_assistant_step(
                        envelope.step,
                        expected,
                        config.bridge_word_limit,
                    )
                ),
            )
            usage = usage.add(assistant_usage)
            request_count += assistant_requests
            match assistant_result.step:
                case ToolActionStep(call=generated_call, audible_text=audible_text):
                    pass
                case FinalResponseStep():
                    raise AssertionError(
                        "Validated tool step unexpectedly became a final response."
                    )
            call_index += 1
            call_id = f"{scenario.scenario_id}-call-{call_index}"
            message_index += 1
            messages.append(
                AssistantMessage(
                    message_id=f"{scenario.scenario_id}-message-{message_index}",
                    audible_text=audible_text,
                    tool_calls=(_canonical_tool_call(call_id, generated_call),),
                )
            )
            outcome, tool_usage, tool_requests = await _execute_tool(
                planned_step=planned_step,
                call=generated_call,
                generator=generator,
                random_seed=_request_seed(scenario.random_seed, turn_index, step_index + 101),
                maximum_attempts=config.maximum_semantic_attempts,
                time_reference=config.time_reference,
            )
            usage = usage.add(tool_usage)
            request_count += tool_requests
            message_index += 1
            messages.append(
                ToolResultMessage(
                    message_id=f"{scenario.scenario_id}-message-{message_index}",
                    call_id=call_id,
                    outcome=outcome,
                )
            )

        final_result, final_usage, final_requests = await _generate_validated(
            generator=generator,
            response_type=AssistantStepEnvelope,
            messages=assistant_step_messages(scenario, messages, None),
            random_seed=_request_seed(scenario.random_seed, turn_index, 999),
            maximum_attempts=config.maximum_semantic_attempts,
            validator=lambda envelope: _validate_assistant_step(
                envelope.step,
                None,
                config.bridge_word_limit,
            ),
        )
        usage = usage.add(final_usage)
        request_count += final_requests
        match final_result.step:
            case FinalResponseStep(audible_text=audible_text):
                message_index += 1
                messages.append(
                    AssistantMessage(
                        message_id=f"{scenario.scenario_id}-message-{message_index}",
                        audible_text=audible_text,
                    )
                )
            case ToolActionStep():
                raise AssertionError("Validated final response unexpectedly became a tool step.")

    metadata = _record_metadata(
        scenario=scenario,
        tools=tools,
        messages=tuple(messages),
        config=config,
    )
    record = ToolDialogueRecord(
        record_id=scenario.scenario_id,
        tools=tools,
        messages=tuple(messages),
        metadata=metadata,
    )
    return GeneratedRecord(record=record, usage=usage, request_count=request_count)


async def _generate_validated(
    generator: StructuredGenerator,
    response_type: type[GeneratedValue],
    messages: Sequence[TeacherChatMessage],
    random_seed: int,
    maximum_attempts: int,
    validator: Callable[[GeneratedValue], None],
) -> tuple[GeneratedValue, TokenUsage, int]:
    usage = TokenUsage()
    last_error: ValueError | None = None
    for attempt in range(maximum_attempts):
        try:
            result = await generator.generate(
                messages=messages,
                response_type=response_type,
                random_seed=random_seed + attempt,
            )
        except ValueError as error:
            last_error = error
            continue
        usage = usage.add(
            TokenUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )
        )
        try:
            validator(result.value)
        except ValueError as error:
            last_error = error
            continue
        return result.value, usage, attempt + 1
    assert last_error is not None
    raise last_error


def _validate_user_turn(
    envelope: GeneratedUserTurnEnvelope,
    expected_speech_style: SpeechStyle,
) -> None:
    if envelope.turn.speech_style is not expected_speech_style:
        raise ValueError(
            f"Expected {expected_speech_style.value} user style, "
            f"got {envelope.turn.speech_style.value}."
        )
    if contains_protocol_markup(envelope.turn.text):
        raise ValueError("Generated user text contains protocol markup.")


def _validate_assistant_step(
    step: AssistantStep,
    expected_tool_name: ToolName | None,
    bridge_word_limit: int,
) -> None:
    AssistantStepValidation(
        expected_tool_name=expected_tool_name,
        maximum_bridge_words=bridge_word_limit,
    ).validate_step(step)
    match step:
        case FinalResponseStep(audible_text=audible_text):
            if contains_protocol_markup(audible_text):
                raise ValueError("Generated assistant text contains protocol markup.")
        case ToolActionStep(audible_text=audible_text, call=CalculateCall(expression=expression)):
            if contains_protocol_markup(audible_text):
                raise ValueError("Generated tool bridge contains protocol markup.")
            calculate_expression(expression)
        case ToolActionStep(audible_text=audible_text):
            if contains_protocol_markup(audible_text):
                raise ValueError("Generated tool bridge contains protocol markup.")


async def _execute_tool(
    planned_step: PlannedToolStep,
    call: GeneratedToolCall,
    generator: StructuredGenerator,
    random_seed: int,
    maximum_attempts: int,
    time_reference: datetime,
) -> tuple[ToolOutcome, TokenUsage, int]:
    match planned_step.outcome:
        case PlannedOutcome.EMPTY:
            return EmptyOutcome(message="The tool returned no useful result."), TokenUsage(), 0
        case PlannedOutcome.FAILURE:
            return FailureOutcome(message="The tool request failed."), TokenUsage(), 0
        case PlannedOutcome.TIMEOUT:
            return TimeoutOutcome(message="The tool request timed out."), TokenUsage(), 0
        case PlannedOutcome.SUCCESS:
            pass

    match call:
        case SearchCall(query=query):
            search_result, usage, requests = await _generate_validated(
                generator=generator,
                response_type=SyntheticSearchResultEnvelope,
                messages=synthetic_search_messages(query),
                random_seed=random_seed,
                maximum_attempts=maximum_attempts,
                validator=_validate_search_result,
            )
            return SuccessOutcome(content=search_result.result.content), usage, requests
        case CalculateCall(expression=expression):
            return SuccessOutcome(content=calculate_expression(expression)), TokenUsage(), 0
        case GetTimeCall():
            return (
                SuccessOutcome(content=seeded_time(random_seed, time_reference)),
                TokenUsage(),
                0,
            )


def _validate_search_result(envelope: SyntheticSearchResultEnvelope) -> None:
    if contains_protocol_markup(envelope.result.content):
        raise ValueError("Synthetic search result contains protocol markup.")


def _canonical_tool_call(call_id: str, generated_call: GeneratedToolCall) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        tool_name=generated_call_tool_name(generated_call).value,
        arguments=call_arguments(generated_call),
    )


def _record_metadata(
    scenario: ScenarioSpec,
    tools: tuple[ToolDefinition, ...],
    messages: tuple[ConversationMessage, ...],
    config: RolloutGenerationConfig,
) -> RecordMetadata:
    expected_steps = tuple(step for turn in scenario.turns for step in turn.tool_steps)
    expected_names = tuple(step.tool_name.value for step in expected_steps)
    outcomes = tuple(step.outcome.value for step in expected_steps)
    content_hash = hashlib.sha256(
        canonical_json(
            RecordHashPayload(
                tools=tools,
                messages=messages,
                scenario_id=scenario.scenario_id,
            )
        ).encode("utf-8")
    ).hexdigest()
    return RecordMetadata(
        generation=GenerationProvenance(
            method="synthetic",
            model_identifier=config.model_identifier,
            model_revision=config.model_revision,
            quantization=config.quantization,
            prompt_revision=PROMPT_REVISION,
            random_seed=scenario.random_seed,
            time_reference=config.time_reference.isoformat(),
        ),
        scenario=ScenarioMetadata(
            family=scenario.family,
            tool_need=(
                "not_needed"
                if not expected_steps
                else "required"
                if len(expected_steps) == 1
                else "multiple_required"
            ),
            expected_tool_names=expected_names,
            outcome=",".join(outcomes) if outcomes else "not_applicable",
            length_band=scenario.length_band,
            speech_style=scenario.speech_style,
            user_turn_count=len(scenario.turns),
        ),
        split=SplitMetadata(
            name=scenario.split,
            policy_revision="family-combination-holdout-v1",
            leakage_group_id=scenario.leakage_group_id,
        ),
        audit=AuditMetadata(
            source_license="project-synthetic",
            intended_license="project-owned",
            human_review="pending",
            content_sha256=content_hash,
        ),
    )


def _request_seed(base_seed: int, turn_index: int, step_index: int) -> int:
    return (base_seed * 1_000_003 + turn_index * 10_007 + step_index) % (2**31)


def _append_failure(path: Path, failure: GenerationFailure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(failure.model_dump_json())
        output_file.write("\n")


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
