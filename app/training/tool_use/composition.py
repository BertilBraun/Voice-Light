from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Literal

from pydantic import Field

from app.training.tool_use.scenario import SegmentBucket
from app.training.tool_use.schema import (
    AssistantMessage,
    AuditMetadata,
    ConversationMessage,
    GenerationProvenance,
    LengthBand,
    RecordMetadata,
    ScenarioMetadata,
    SpeechStyle,
    SplitMetadata,
    SystemMessage,
    ToolDefinition,
    ToolDialogueRecord,
    ToolResultMessage,
    ToolUseBaseModel,
    UserMessage,
    canonical_json,
)

COMPOSITION_REVISION = "segment-permutation-v1"


class CompositionPlan(ToolUseBaseModel):
    schema_version: Literal["voice-light.composition-plan/v1"] = "voice-light.composition-plan/v1"
    composition_id: str
    random_seed: int
    segment_record_ids: tuple[str, ...] = Field(min_length=2)


class CompositionHashPayload(ToolUseBaseModel):
    tools: tuple[ToolDefinition, ...]
    messages: tuple[ConversationMessage, ...]
    segment_record_ids: tuple[str, ...]


def compose_segment_record(
    plan: CompositionPlan,
    source_records: tuple[ToolDialogueRecord, ...],
) -> ToolDialogueRecord:
    actual_ids = tuple(record.record_id for record in source_records)
    if actual_ids != plan.segment_record_ids:
        raise ValueError("Source record order must match the composition plan.")
    first = source_records[0]
    if any(record.tools != first.tools for record in source_records[1:]):
        raise ValueError("Composed segments must advertise identical runtime tools.")
    if any(
        record.metadata.scenario.speech_style is not first.metadata.scenario.speech_style
        for record in source_records[1:]
    ):
        raise ValueError("Composed segments must use one speech style.")
    if any(
        record.metadata.split.name is not first.metadata.split.name for record in source_records[1:]
    ):
        raise ValueError("Composed segments must belong to one dataset split.")

    match first.messages[0]:
        case SystemMessage(text=system_text):
            pass
        case _:
            raise AssertionError("Canonical source records always begin with a system message.")
    for record in source_records[1:]:
        match record.messages[0]:
            case SystemMessage(text=candidate_text):
                if candidate_text != system_text:
                    raise ValueError("Composed segments must use the same system instruction.")
            case _:
                raise AssertionError("Canonical source records always begin with a system message.")

    messages = (
        SystemMessage(
            message_id=f"{plan.composition_id}-system",
            text=system_text,
        ),
        *(message for record in source_records for message in record.messages[1:]),
    )
    expected_tool_names = tuple(
        tool_call.tool_name
        for message in messages
        if isinstance(message, AssistantMessage)
        for tool_call in message.tool_calls
    )
    outcomes = tuple(
        message.outcome.status for message in messages if isinstance(message, ToolResultMessage)
    )
    user_turn_count = sum(isinstance(message, UserMessage) for message in messages)
    if not 8 <= user_turn_count <= 16:
        raise ValueError("Calibration compositions must contain 8 to 16 user rounds.")

    content_hash = hashlib.sha256(
        canonical_json(
            CompositionHashPayload(
                tools=first.tools,
                messages=messages,
                segment_record_ids=plan.segment_record_ids,
            )
        ).encode("utf-8")
    ).hexdigest()
    lineage_hash = hashlib.sha256(
        "\n".join(sorted(plan.segment_record_ids)).encode("utf-8")
    ).hexdigest()[:20]
    return ToolDialogueRecord(
        record_id=plan.composition_id,
        tools=first.tools,
        messages=messages,
        metadata=RecordMetadata(
            generation=GenerationProvenance(
                method="synthetic",
                model_identifier=first.metadata.generation.model_identifier,
                model_revision=first.metadata.generation.model_revision,
                quantization=first.metadata.generation.quantization,
                prompt_revision=COMPOSITION_REVISION,
                random_seed=plan.random_seed,
                time_reference=first.metadata.generation.time_reference,
            ),
            scenario=ScenarioMetadata(
                family="composed_segments",
                tool_need=(
                    "not_needed"
                    if not expected_tool_names
                    else "required"
                    if len(expected_tool_names) == 1
                    else "multiple_required"
                ),
                expected_tool_names=expected_tool_names,
                outcome=",".join(outcomes) if outcomes else "not_applicable",
                length_band=LengthBand.LONG,
                speech_style=first.metadata.scenario.speech_style,
                user_turn_count=user_turn_count,
            ),
            split=SplitMetadata(
                name=first.metadata.split.name,
                policy_revision="segment-lineage-permutation-holdout-v1",
                leakage_group_id=f"composition-{lineage_hash}",
            ),
            audit=AuditMetadata(
                source_license=first.metadata.audit.source_license,
                intended_license=first.metadata.audit.intended_license,
                human_review="pending",
                content_sha256=content_hash,
            ),
        ),
    )


def build_bucket_calibration_compositions(
    source_records: tuple[ToolDialogueRecord, ...],
    random_seed: int,
) -> tuple[tuple[CompositionPlan, ToolDialogueRecord], ...]:
    no_tool_buckets = (
        SegmentBucket.NATURAL_CONVERSATION,
        SegmentBucket.DRAFTING,
        SegmentBucket.BRAINSTORM_DECISION,
    )
    single_tool_buckets = (
        SegmentBucket.SEARCH,
        SegmentBucket.CALCULATE,
        SegmentBucket.GET_TIME,
    )
    compositions: list[tuple[CompositionPlan, ToolDialogueRecord]] = []
    for style_index, speech_style in enumerate(SpeechStyle):
        selected = (
            _record_for_bucket_and_style(
                source_records,
                no_tool_buckets[style_index % len(no_tool_buckets)],
                speech_style,
            ),
            _record_for_bucket_and_style(
                source_records,
                single_tool_buckets[style_index % len(single_tool_buckets)],
                speech_style,
            ),
            _record_for_bucket_and_style(
                source_records,
                SegmentBucket.SEQUENTIAL_TOOLS,
                speech_style,
            ),
            _record_for_bucket_and_style(
                source_records,
                SegmentBucket.CORRECTION_RECOVERY,
                speech_style,
            ),
        )
        generator = random.Random(random_seed + style_index)
        permutations: list[tuple[int, ...]] = []
        while len(permutations) < 2:
            indexes = list(range(len(selected)))
            generator.shuffle(indexes)
            permutation = tuple(indexes)
            if permutation not in permutations:
                permutations.append(permutation)
        for permutation_index, permutation in enumerate(permutations):
            ordered_records = tuple(selected[index] for index in permutation)
            composition_index = style_index * 2 + permutation_index
            plan = CompositionPlan(
                composition_id=f"composition-{random_seed}-{composition_index:03d}",
                random_seed=random_seed + composition_index,
                segment_record_ids=tuple(record.record_id for record in ordered_records),
            )
            compositions.append((plan, compose_segment_record(plan, ordered_records)))
    return tuple(compositions)


def select_bucket_calibration_segments(
    candidate_records: tuple[ToolDialogueRecord, ...],
) -> tuple[ToolDialogueRecord, ...]:
    return tuple(
        _record_for_bucket_and_style(candidate_records, bucket, speech_style)
        for bucket in SegmentBucket
        for speech_style in SpeechStyle
    )


def write_composition_plans(
    path: Path,
    plans: tuple[CompositionPlan, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(plan.model_dump_json() for plan in plans)
    path.write_text(f"{content}\n", encoding="utf-8")


def write_composed_records(
    path: Path,
    records: tuple[ToolDialogueRecord, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(record.model_dump_json() for record in records)
    path.write_text(f"{content}\n", encoding="utf-8")


def _record_for_bucket_and_style(
    records: tuple[ToolDialogueRecord, ...],
    bucket: SegmentBucket,
    speech_style: SpeechStyle,
) -> ToolDialogueRecord:
    matches = tuple(
        record
        for record in records
        if record.metadata.scenario.family == bucket.value
        and record.metadata.scenario.speech_style is speech_style
    )
    if not matches:
        raise ValueError(f"No accepted {bucket.value}/{speech_style.value} segment was found.")
    return sorted(matches, key=lambda record: record.record_id)[0]
