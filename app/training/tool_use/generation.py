from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import AwareDatetime, Field

from app.training.tool_use.prompts import (
    PROMPT_REVISION,
    assistant_step_messages,
    record_grounding_messages,
    record_quality_messages,
    synthetic_search_messages,
    user_turn_messages,
    user_turn_scope_messages,
)
from app.training.tool_use.protocol import (
    AssistantStep,
    AssistantStepEnvelope,
    AssistantStepValidation,
    CalculateCall,
    FinalResponseStep,
    FinalResponseStepEnvelope,
    GeneratedToolCall,
    GeneratedUserTurnEnvelope,
    GetTimeCall,
    GroundingVerdict,
    RecordGroundingAssessment,
    RecordGroundingAssessmentEnvelope,
    RecordQualityAssessment,
    RecordQualityAssessmentEnvelope,
    SearchCall,
    SyntheticSearchResultEnvelope,
    TeacherChatMessage,
    TeacherSamplingMode,
    ToolActionStep,
    UserTurnScopeAssessment,
    UserTurnScopeAssessmentEnvelope,
    generated_call_tool_name,
)
from app.training.tool_use.scenario import (
    AssistantTurnPlan,
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
from app.training.tool_use.tools import (
    calculate_expression,
    call_arguments,
    seeded_time,
    seeded_time_before_deadline,
)
from app.training.tool_use.vllm_client import StructuredGenerationResult

GeneratedValue = TypeVar("GeneratedValue", bound=ToolUseBaseModel)
MAXIMUM_SPOKEN_TURN_WORDS = 100


class StructuredGenerator(Protocol):
    async def generate(
        self,
        messages: Sequence[TeacherChatMessage],
        response_type: type[GeneratedValue],
        random_seed: int,
        sampling_mode: TeacherSamplingMode = TeacherSamplingMode.CREATIVE,
    ) -> StructuredGenerationResult[GeneratedValue]: ...


class RolloutGenerationConfig(ToolUseBaseModel):
    model_identifier: str
    model_revision: str
    quantization: str
    time_reference: AwareDatetime
    maximum_concurrency: int = Field(default=128, ge=1)
    maximum_semantic_attempts: int = Field(default=3, ge=1)
    maximum_record_attempts: int = Field(default=3, ge=1)
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


class GenerationAttemptError(ValueError):
    def __init__(self, message: str, usage: TokenUsage, request_count: int) -> None:
        super().__init__(message)
        self.usage = usage
        self.request_count = request_count


@dataclass(frozen=True)
class GeneratedRecord:
    record: ToolDialogueRecord
    usage: TokenUsage
    request_count: int
    rejected_candidate_count: int


class GenerationFailure(ToolUseBaseModel):
    scenario_id: str
    error_type: str
    message: str
    rejected_candidates: int = Field(ge=0)
    recorded_at: str


class GenerationRunManifest(ToolUseBaseModel):
    schema_version: str = "voice-light.generation-run/v2"
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
    rejected_candidates: int = Field(ge=0)
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


class RecordGenerationError(ValueError):
    def __init__(self, message: str, rejected_candidate_count: int) -> None:
        super().__init__(message)
        self.rejected_candidate_count = rejected_candidate_count


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
    rejected_candidate_count = 0
    for task, scenario in task_scenarios:
        try:
            generated = await task
        except Exception as error:
            match error:
                case RecordGenerationError():
                    rejected_candidates = error.rejected_candidate_count
                case _:
                    rejected_candidates = 0
            failure = GenerationFailure(
                scenario_id=scenario.scenario_id,
                error_type=type(error).__name__,
                message=str(error),
                rejected_candidates=rejected_candidates,
                recorded_at=_utc_now(),
            )
            _append_failure(failure_path, failure)
            failures.append(failure)
            rejected_candidate_count += rejected_candidates
            continue
        append_record(output_path, generated.record)
        generated_records.append(generated.record)
        usage = usage.add(generated.usage)
        request_count += generated.request_count
        rejected_candidate_count += generated.rejected_candidate_count

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
        rejected_candidates=rejected_candidate_count,
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
    usage = TokenUsage()
    request_count = 0
    last_error: ValueError | None = None
    rejected_candidate_count = 0
    for record_attempt in range(config.maximum_record_attempts):
        try:
            candidate = await _generate_record_candidate(
                scenario=scenario,
                generator=generator,
                config=config,
                seed_salt=record_attempt * 1_000_003,
            )
        except GenerationAttemptError as error:
            usage = usage.add(error.usage)
            request_count += error.request_count
            last_error = error
            rejected_candidate_count += 1
            continue
        except ValueError as error:
            last_error = error
            rejected_candidate_count += 1
            continue
        usage = usage.add(candidate.usage)
        request_count += candidate.request_count
        try:
            _validate_candidate_record(candidate.record)
            critique_result = await generator.generate(
                messages=record_quality_messages(scenario, candidate.record.messages),
                response_type=RecordQualityAssessmentEnvelope,
                random_seed=_request_seed(
                    scenario.random_seed + record_attempt,
                    len(scenario.turns),
                    10_001,
                ),
                sampling_mode=TeacherSamplingMode.DETERMINISTIC,
            )
            usage = usage.add(
                TokenUsage(
                    prompt_tokens=critique_result.prompt_tokens,
                    completion_tokens=critique_result.completion_tokens,
                )
            )
            request_count += 1
            _validate_record_quality(critique_result.value.assessment)
            grounding_result = await generator.generate(
                messages=record_grounding_messages(scenario, candidate.record.messages),
                response_type=RecordGroundingAssessmentEnvelope,
                random_seed=_request_seed(
                    scenario.random_seed + record_attempt,
                    len(scenario.turns),
                    10_002,
                ),
                sampling_mode=TeacherSamplingMode.DETERMINISTIC,
            )
            usage = usage.add(
                TokenUsage(
                    prompt_tokens=grounding_result.prompt_tokens,
                    completion_tokens=grounding_result.completion_tokens,
                )
            )
            request_count += 1
            _validate_record_grounding(
                candidate.record,
                grounding_result.value.assessment,
            )
        except ValueError as error:
            last_error = error
            rejected_candidate_count += 1
            continue
        return GeneratedRecord(
            record=candidate.record,
            usage=usage,
            request_count=request_count,
            rejected_candidate_count=rejected_candidate_count,
        )
    assert last_error is not None
    raise RecordGenerationError(
        str(last_error),
        rejected_candidate_count=rejected_candidate_count,
    ) from last_error


async def _generate_record_candidate(
    scenario: ScenarioSpec,
    generator: StructuredGenerator,
    config: RolloutGenerationConfig,
    seed_salt: int,
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
        try:
            user_result, user_usage, user_requests = await _generate_scoped_user_turn(
                generator=generator,
                scenario=scenario,
                turn_plan=turn_plan,
                public_history=messages,
                random_seed=_request_seed(scenario.random_seed + seed_salt, turn_index, 0),
                maximum_attempts=config.maximum_semantic_attempts,
            )
        except GenerationAttemptError as error:
            raise GenerationAttemptError(
                str(error),
                usage=usage.add(error.usage),
                request_count=request_count + error.request_count,
            ) from error
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
                    tuple(
                        later_step.tool_name
                        for later_step in turn_plan.tool_steps[step_index + 1 :]
                    ),
                ),
                random_seed=_request_seed(
                    scenario.random_seed + seed_salt,
                    turn_index,
                    step_index + 1,
                ),
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
                random_seed=_request_seed(
                    scenario.random_seed + seed_salt,
                    turn_index,
                    step_index + 101,
                ),
                maximum_attempts=config.maximum_semantic_attempts,
                time_reference=config.time_reference,
                public_history=messages,
                constrain_time_before_deadline=tuple(
                    step.tool_name for step in turn_plan.tool_steps
                )
                == (ToolName.GET_TIME, ToolName.CALCULATE),
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
            response_type=FinalResponseStepEnvelope,
            messages=assistant_step_messages(
                scenario,
                messages,
                None,
                tuple(
                    future_step.tool_name
                    for future_turn in scenario.turns[turn_index + 1 :]
                    for future_step in future_turn.tool_steps
                )
                if call_index == 0 and not turn_plan.tool_steps
                else (),
            ),
            random_seed=_request_seed(scenario.random_seed + seed_salt, turn_index, 999),
            maximum_attempts=config.maximum_semantic_attempts,
            validator=lambda envelope: _validate_assistant_step(
                envelope.step,
                None,
                config.bridge_word_limit,
            ),
        )
        usage = usage.add(final_usage)
        request_count += final_requests
        message_index += 1
        messages.append(
            AssistantMessage(
                message_id=f"{scenario.scenario_id}-message-{message_index}",
                audible_text=final_result.step.audible_text,
            )
        )

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
    return GeneratedRecord(
        record=record,
        usage=usage,
        request_count=request_count,
        rejected_candidate_count=0,
    )


async def _generate_scoped_user_turn(
    generator: StructuredGenerator,
    scenario: ScenarioSpec,
    turn_plan: AssistantTurnPlan,
    public_history: Sequence[ConversationMessage],
    random_seed: int,
    maximum_attempts: int,
) -> tuple[GeneratedUserTurnEnvelope, TokenUsage, int]:
    usage = TokenUsage()
    request_count = 0
    last_error: ValueError | None = None
    expected_tools = tuple(step.tool_name for step in turn_plan.tool_steps)
    for attempt in range(maximum_attempts):
        try:
            result = await generator.generate(
                messages=user_turn_messages(scenario, turn_plan, public_history),
                response_type=GeneratedUserTurnEnvelope,
                random_seed=random_seed + attempt * 2,
            )
            usage = usage.add(
                TokenUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
            )
            request_count += 1
            _validate_user_turn(result.value, scenario.speech_style)
            assessment_result = await generator.generate(
                messages=user_turn_scope_messages(public_history, result.value.turn.text),
                response_type=UserTurnScopeAssessmentEnvelope,
                random_seed=random_seed + attempt * 2 + 1,
                sampling_mode=TeacherSamplingMode.DETERMINISTIC,
            )
            usage = usage.add(
                TokenUsage(
                    prompt_tokens=assessment_result.prompt_tokens,
                    completion_tokens=assessment_result.completion_tokens,
                )
            )
            request_count += 1
            _validate_user_turn_scope(
                assessment=assessment_result.value.assessment,
                expected_tools=expected_tools,
            )
        except ValueError as error:
            last_error = error
            continue
        return result.value, usage, request_count
    assert last_error is not None
    raise GenerationAttemptError(
        str(last_error),
        usage=usage,
        request_count=request_count,
    ) from last_error


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
    if len(envelope.turn.text.split()) > MAXIMUM_SPOKEN_TURN_WORDS:
        raise ValueError("Generated user turn exceeds 100 words.")


def _validate_user_turn_scope(
    assessment: UserTurnScopeAssessment,
    expected_tools: tuple[ToolName, ...],
) -> None:
    if not assessment.scope_is_clear:
        raise ValueError("Generated user turn has unclear information scope.")
    if not assessment.request_is_supported:
        raise ValueError("Generated user turn requests an unsupported action.")
    if assessment.repeats_prior_user_turn:
        raise ValueError("Generated user turn repeats an earlier user turn.")
    if not assessment.voice_style_is_natural:
        raise ValueError("Generated user turn is not natural spoken language.")
    if assessment.required_tools != expected_tools:
        expected_names = ", ".join(tool.value for tool in expected_tools) or "none"
        actual_names = ", ".join(tool.value for tool in assessment.required_tools) or "none"
        raise ValueError(f"Generated user turn requires {actual_names}; expected {expected_names}.")


def _validate_assistant_step(
    step: AssistantStep,
    expected_tool_name: ToolName | None,
    bridge_word_limit: int,
) -> None:
    AssistantStepValidation(
        expected_tool_name=expected_tool_name,
        maximum_bridge_words=bridge_word_limit,
    ).validate_step(step)
    if len(step.audible_text.split()) > MAXIMUM_SPOKEN_TURN_WORDS:
        raise ValueError("Generated assistant response exceeds 100 words.")
    match step:
        case FinalResponseStep(audible_text=audible_text):
            if contains_protocol_markup(audible_text):
                raise ValueError("Generated assistant text contains protocol markup.")
        case ToolActionStep(audible_text=audible_text, call=CalculateCall(expression=expression)):
            if contains_protocol_markup(audible_text):
                raise ValueError("Generated tool bridge contains protocol markup.")
            if audible_text.rstrip().endswith("?"):
                raise ValueError("Generated tool bridge is phrased as a question.")
            calculate_expression(expression)
        case ToolActionStep(audible_text=audible_text):
            if contains_protocol_markup(audible_text):
                raise ValueError("Generated tool bridge contains protocol markup.")
            if audible_text.rstrip().endswith("?"):
                raise ValueError("Generated tool bridge is phrased as a question.")


async def _execute_tool(
    planned_step: PlannedToolStep,
    call: GeneratedToolCall,
    generator: StructuredGenerator,
    random_seed: int,
    maximum_attempts: int,
    time_reference: datetime,
    public_history: Sequence[ConversationMessage],
    constrain_time_before_deadline: bool,
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
                messages=synthetic_search_messages(query, public_history),
                random_seed=random_seed,
                maximum_attempts=maximum_attempts,
                validator=_validate_search_result,
            )
            return SuccessOutcome(content=search_result.result.content), usage, requests
        case CalculateCall(expression=expression):
            return SuccessOutcome(content=calculate_expression(expression)), TokenUsage(), 0
        case GetTimeCall():
            if constrain_time_before_deadline:
                deadline_minutes = _deadline_minutes_from_history(public_history)
                return (
                    SuccessOutcome(
                        content=seeded_time_before_deadline(
                            random_seed,
                            time_reference,
                            deadline_minutes,
                        )
                    ),
                    TokenUsage(),
                    0,
                )
            return (
                SuccessOutcome(content=seeded_time(random_seed, time_reference)),
                TokenUsage(),
                0,
            )


def _validate_search_result(envelope: SyntheticSearchResultEnvelope) -> None:
    if contains_protocol_markup(envelope.result.content):
        raise ValueError("Synthetic search result contains protocol markup.")


def _validate_candidate_record(record: ToolDialogueRecord) -> None:
    seen_user_turns: set[str] = set()
    for message_index, message in enumerate(record.messages):
        match message:
            case UserMessage(text=text):
                normalized_text = _normalized_spoken_text(text)
                if normalized_text in seen_user_turns:
                    raise ValueError("Candidate repeats an earlier user utterance.")
                seen_user_turns.add(normalized_text)
            case AssistantMessage(audible_text=audible_text, tool_calls=tool_calls) if tool_calls:
                result = _result_after_call(record.messages, message_index, tool_calls[0].call_id)
                _validate_bridge_does_not_reveal_result(
                    audible_text=audible_text,
                    tool_name=tool_calls[0].tool_name,
                    outcome=result.outcome,
                )
                if tool_calls[0].tool_name == ToolName.CALCULATE.value:
                    response = _assistant_after_result(record.messages, result.call_id)
                    _validate_calculate_response(result.outcome, response.audible_text)
            case _:
                pass
    _validate_time_then_calculate_turns(record)


def _deadline_minutes_from_history(public_history: Sequence[ConversationMessage]) -> int:
    for message in reversed(public_history):
        match message:
            case UserMessage(text=text):
                return _deadline_minutes(text)
            case _:
                pass
    raise ValueError("Time-calculation scenario has no preceding user turn.")


def _deadline_minutes(text: str) -> int:
    matches = tuple(
        re.finditer(
            r"\b(?P<hour>1[0-2]|0?[1-9])"
            r"(?::(?P<minute>[0-5]\d))?\s*"
            r"(?P<period>a\.?m\.?|p\.?m\.?)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if len(matches) != 1:
        raise ValueError("Expected exactly one numeric AM/PM deadline.")
    match = matches[0]
    hour = int(match.group("hour"))
    minute_text = match.group("minute")
    minute = int(minute_text) if minute_text is not None else 0
    if match.group("period").casefold().startswith("p") and hour != 12:
        hour += 12
    if match.group("period").casefold().startswith("a") and hour == 12:
        hour = 0
    deadline_minutes = hour * 60 + minute
    if not 13 * 60 <= deadline_minutes < 24 * 60:
        raise ValueError("Time-calculation deadline must be between 1:00 PM and midnight.")
    return deadline_minutes


def _validate_time_then_calculate_turns(record: ToolDialogueRecord) -> None:
    for message_index, message in enumerate(record.messages):
        match message:
            case UserMessage(text=user_text):
                turn_messages = _messages_until_next_user(record.messages, message_index + 1)
                _validate_time_calculation_turn(user_text, turn_messages)
            case _:
                pass


def _messages_until_next_user(
    messages: tuple[ConversationMessage, ...],
    start_index: int,
) -> tuple[ConversationMessage, ...]:
    turn_messages: list[ConversationMessage] = []
    for message in messages[start_index:]:
        match message:
            case UserMessage():
                break
            case _:
                turn_messages.append(message)
    return tuple(turn_messages)


def _validate_time_calculation_turn(
    user_text: str,
    turn_messages: tuple[ConversationMessage, ...],
) -> None:
    tool_assistant_messages = tuple(
        (message_index, message)
        for message_index, message in enumerate(turn_messages)
        if isinstance(message, AssistantMessage) and message.tool_calls
    )
    tool_names = tuple(message.tool_calls[0].tool_name for _, message in tool_assistant_messages)
    if tool_names != (ToolName.GET_TIME.value, ToolName.CALCULATE.value):
        return

    deadline_minutes = _deadline_minutes(user_text)
    time_message_index, time_message = tool_assistant_messages[0]
    time_result = _result_after_call(
        turn_messages,
        time_message_index,
        time_message.tool_calls[0].call_id,
    )
    match time_result.outcome:
        case SuccessOutcome(content=timestamp):
            parsed_time = datetime.fromisoformat(timestamp)
            current_minutes = parsed_time.hour * 60 + parsed_time.minute
        case _:
            return

    calculation_message_index, calculation_message = tool_assistant_messages[1]
    calculation_result = _result_after_call(
        turn_messages,
        calculation_message_index,
        calculation_message.tool_calls[0].call_id,
    )
    match calculation_result.outcome:
        case SuccessOutcome(content=result_content):
            expected_result = str(deadline_minutes - current_minutes)
            if result_content != expected_result:
                raise ValueError("Time calculation does not match the sampled time and deadline.")
        case _:
            pass


def _result_after_call(
    messages: tuple[ConversationMessage, ...],
    assistant_message_index: int,
    call_id: str,
) -> ToolResultMessage:
    for later_message in messages[assistant_message_index + 1 :]:
        match later_message:
            case ToolResultMessage(call_id=result_call_id) if result_call_id == call_id:
                return later_message
            case _:
                pass
    raise AssertionError("Canonical record omitted the result for a tool call.")


def _validate_bridge_does_not_reveal_result(
    audible_text: str,
    tool_name: str,
    outcome: ToolOutcome,
) -> None:
    match outcome:
        case SuccessOutcome(content=result_content):
            pass
        case _:
            return
    normalized_bridge = _normalized_spoken_text(audible_text)
    normalized_result = _normalized_spoken_text(result_content)
    if (
        tool_name == ToolName.CALCULATE.value
        and normalized_result
        and re.search(rf"\b{re.escape(normalized_result)}\b", normalized_bridge)
    ):
        raise ValueError("Calculate bridge reveals the pending result.")
    if tool_name == ToolName.GET_TIME.value and re.search(
        r"\b(?:[0-2]?\d:[0-5]\d|a\.?m\.?|p\.?m\.?|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\b",
        audible_text.casefold(),
    ):
        raise ValueError("Get-time bridge states a pending date or time.")


def _assistant_after_result(
    messages: tuple[ConversationMessage, ...],
    call_id: str,
) -> AssistantMessage:
    result_seen = False
    for message in messages:
        match message:
            case ToolResultMessage(call_id=result_call_id) if result_call_id == call_id:
                result_seen = True
            case AssistantMessage() if result_seen:
                return message
            case _:
                pass
    raise AssertionError("Canonical record omitted the assistant response after a tool result.")


def _validate_calculate_response(outcome: ToolOutcome, audible_text: str) -> None:
    match outcome:
        case SuccessOutcome(content=result_content):
            pass
        case _:
            return
    if not re.search(
        rf"(?<![\d.]){re.escape(result_content)}(?!\d|[.,]\d)",
        audible_text,
    ):
        raise ValueError("Assistant response does not state the exact calculate result.")


def _validate_record_quality(assessment: RecordQualityAssessment) -> None:
    checks = (
        assessment.tool_plan_is_sufficient,
        assessment.no_premature_tool_results,
        assessment.assistant_claims_are_grounded,
        assessment.tool_results_are_used_correctly,
        assessment.user_turns_are_distinct,
        assessment.sequential_calls_are_justified,
        assessment.no_unsupported_action_commitments,
        assessment.conversation_is_coherent,
        assessment.voice_style_is_natural,
        assessment.tone_is_appropriate,
    )
    if not all(checks) or assessment.issues:
        issue_names = ", ".join(issue.value for issue in assessment.issues) or "unspecified"
        raise ValueError(f"Record quality audit rejected candidate: {issue_names}.")


def _validate_record_grounding(
    record: ToolDialogueRecord,
    assessment: RecordGroundingAssessment,
) -> None:
    expected_message_ids = tuple(
        message.message_id for message in record.messages if isinstance(message, AssistantMessage)
    )
    actual_message_ids = tuple(finding.message_id for finding in assessment.findings)
    if actual_message_ids != expected_message_ids:
        raise ValueError("Grounding audit did not assess every assistant message in order.")
    unsupported_findings = tuple(
        finding
        for finding in assessment.findings
        if finding.verdict is GroundingVerdict.UNSUPPORTED or finding.unsupported_excerpts
    )
    if unsupported_findings:
        excerpts = tuple(
            excerpt for finding in unsupported_findings for excerpt in finding.unsupported_excerpts
        )
        detail = "; ".join(excerpts) or "unspecified assistant claim"
        raise ValueError(f"Grounding audit rejected candidate: {detail}.")


def _normalized_spoken_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


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
