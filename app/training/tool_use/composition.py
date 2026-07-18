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
    DatasetSplit,
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
    logical_epoch: int = Field(ge=0)
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
                logical_epoch=0,
                segment_record_ids=tuple(record.record_id for record in ordered_records),
            )
            compositions.append((plan, compose_segment_record(plan, ordered_records)))
    return tuple(compositions)


def build_training_compositions(
    source_records: tuple[ToolDialogueRecord, ...],
    split: DatasetSplit,
    logical_epoch: int,
    random_seed: int,
    minimum_user_turns: int,
    maximum_user_turns: int,
) -> tuple[tuple[CompositionPlan, ToolDialogueRecord], ...]:
    if minimum_user_turns < 1:
        raise ValueError("Minimum user turns must be positive.")
    if maximum_user_turns < minimum_user_turns:
        raise ValueError("Maximum user turns cannot be smaller than the minimum.")
    split_records = tuple(
        record for record in source_records if record.metadata.split.name is split
    )
    if not split_records:
        raise ValueError(f"No source records belong to the {split.value} split.")

    compositions: list[tuple[CompositionPlan, ToolDialogueRecord]] = []
    composition_index = 0
    for speech_style in SpeechStyle:
        style_records = tuple(
            record
            for record in split_records
            if record.metadata.scenario.speech_style is speech_style
        )
        if not style_records:
            continue
        style_groups = _group_style_segments(
            records=style_records,
            random_seed=random_seed + logical_epoch * 10_000 + len(compositions),
            minimum_user_turns=minimum_user_turns,
            maximum_user_turns=maximum_user_turns,
        )
        for group in style_groups:
            plan_seed = random_seed + logical_epoch * 1_000_000 + composition_index
            plan = CompositionPlan(
                composition_id=(
                    f"training-{split.value}-e{logical_epoch:02d}-"
                    f"{speech_style.value}-{composition_index:05d}"
                ),
                random_seed=plan_seed,
                logical_epoch=logical_epoch,
                segment_record_ids=tuple(record.record_id for record in group),
            )
            compositions.append((plan, compose_segment_record(plan, group)))
            composition_index += 1
    if not compositions:
        raise ValueError(f"No {split.value} compositions could satisfy the turn bounds.")
    return tuple(compositions)


def select_bucket_calibration_segments(
    candidate_records: tuple[ToolDialogueRecord, ...],
) -> tuple[ToolDialogueRecord, ...]:
    return tuple(
        _record_for_bucket_and_style(candidate_records, bucket, speech_style)
        for bucket in SegmentBucket
        for speech_style in SpeechStyle
    )


def select_review_records(
    candidate_records: tuple[ToolDialogueRecord, ...],
    count: int,
    random_seed: int,
) -> tuple[ToolDialogueRecord, ...]:
    if count < 1:
        raise ValueError("Review record count must be positive.")
    if count > len(candidate_records):
        raise ValueError("Review record count exceeds the available records.")
    generator = random.Random(random_seed)
    selected: list[ToolDialogueRecord] = []
    remaining = list(candidate_records)
    if count >= len(SegmentBucket):
        for bucket in SegmentBucket:
            bucket_candidates = tuple(
                record for record in remaining if record.metadata.scenario.family == bucket.value
            )
            if not bucket_candidates:
                raise ValueError(f"No {bucket.value} record is available for review.")
            record = generator.choice(bucket_candidates)
            selected.append(record)
            remaining.remove(record)
    additional_count = count - len(selected)
    selected.extend(generator.sample(remaining, additional_count))
    generator.shuffle(selected)
    return tuple(selected)


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


def _group_style_segments(
    records: tuple[ToolDialogueRecord, ...],
    random_seed: int,
    minimum_user_turns: int,
    maximum_user_turns: int,
) -> tuple[tuple[ToolDialogueRecord, ...], ...]:
    generator = random.Random(random_seed)
    remaining = list(records)
    generator.shuffle(remaining)
    groups: list[list[ToolDialogueRecord]] = []
    while remaining:
        target_user_turns = generator.randint(minimum_user_turns, maximum_user_turns)
        group: list[ToolDialogueRecord] = []
        user_turns = 0
        used_buckets: set[str] = set()
        previous_bucket: str | None = None
        while user_turns < minimum_user_turns:
            candidate_index = _choose_segment_index(
                records=remaining,
                generator=generator,
                maximum_additional_turns=maximum_user_turns - user_turns,
                used_buckets=used_buckets,
                previous_bucket=previous_bucket,
            )
            if candidate_index is None:
                break
            candidate = remaining.pop(candidate_index)
            group.append(candidate)
            user_turns += candidate.metadata.scenario.user_turn_count
            previous_bucket = candidate.metadata.scenario.family
            used_buckets.add(previous_bucket)
        if user_turns < minimum_user_turns or len(used_buckets) < 2:
            remaining.extend(group)
            break
        while user_turns < target_user_turns:
            candidate_index = _choose_segment_index(
                records=remaining,
                generator=generator,
                maximum_additional_turns=maximum_user_turns - user_turns,
                used_buckets=used_buckets,
                previous_bucket=previous_bucket,
            )
            if candidate_index is None:
                break
            candidate = remaining.pop(candidate_index)
            group.append(candidate)
            user_turns += candidate.metadata.scenario.user_turn_count
            previous_bucket = candidate.metadata.scenario.family
            used_buckets.add(previous_bucket)
        groups.append(group)
    generator.shuffle(remaining)
    _distribute_remaining_segments(
        remaining=remaining,
        groups=groups,
        maximum_user_turns=maximum_user_turns,
    )
    return tuple(tuple(group) for group in groups)


def _choose_segment_index(
    records: list[ToolDialogueRecord],
    generator: random.Random,
    maximum_additional_turns: int,
    used_buckets: set[str],
    previous_bucket: str | None,
) -> int | None:
    fitting_indexes = tuple(
        index
        for index, record in enumerate(records)
        if record.metadata.scenario.user_turn_count <= maximum_additional_turns
    )
    if not fitting_indexes:
        return None
    new_bucket_indexes = tuple(
        index
        for index in fitting_indexes
        if records[index].metadata.scenario.family not in used_buckets
    )
    different_bucket_indexes = tuple(
        index
        for index in fitting_indexes
        if records[index].metadata.scenario.family != previous_bucket
    )
    candidates = new_bucket_indexes or different_bucket_indexes or fitting_indexes
    return generator.choice(candidates)


def _distribute_remaining_segments(
    remaining: list[ToolDialogueRecord],
    groups: list[list[ToolDialogueRecord]],
    maximum_user_turns: int,
) -> None:
    unplaced: list[ToolDialogueRecord] = []
    while remaining:
        record = remaining.pop()
        fitting_group_indexes = tuple(
            index
            for index, group in enumerate(groups)
            if _user_turn_count(group) + record.metadata.scenario.user_turn_count
            <= maximum_user_turns
        )
        if not fitting_group_indexes:
            unplaced.append(record)
            continue
        new_bucket_group_indexes = tuple(
            index
            for index in fitting_group_indexes
            if record.metadata.scenario.family
            not in {candidate.metadata.scenario.family for candidate in groups[index]}
        )
        candidates = new_bucket_group_indexes or fitting_group_indexes
        target_index = max(
            candidates,
            key=lambda index: maximum_user_turns - _user_turn_count(groups[index]),
        )
        groups[target_index].append(record)
    remaining.extend(unplaced)


def _user_turn_count(records: list[ToolDialogueRecord]) -> int:
    return sum(record.metadata.scenario.user_turn_count for record in records)
