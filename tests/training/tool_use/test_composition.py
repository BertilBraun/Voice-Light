from __future__ import annotations

from app.training.tool_use.composition import (
    build_bucket_calibration_compositions,
    build_training_compositions,
    select_bucket_calibration_segments,
    select_review_records,
)
from app.training.tool_use.generation import runtime_tool_definitions
from app.training.tool_use.scenario import SegmentBucket
from app.training.tool_use.schema import (
    AssistantMessage,
    AuditMetadata,
    DatasetSplit,
    GenerationProvenance,
    LengthBand,
    RecordMetadata,
    ScenarioMetadata,
    SpeechStyle,
    SplitMetadata,
    SystemMessage,
    ToolDialogueRecord,
    UserMessage,
)


def test_bucket_compositions_create_two_permutations_for_each_style() -> None:
    source_records = tuple(
        _source_record(
            record_id=f"{bucket.value}-{speech_style.value}",
            family=bucket.value,
            speech_style=speech_style,
            split=_split_for_style(speech_style),
        )
        for bucket in SegmentBucket
        for speech_style in SpeechStyle
    )

    compositions = build_bucket_calibration_compositions(
        source_records=source_records,
        random_seed=67,
    )

    assert len(compositions) == 10
    assert all(record.metadata.scenario.user_turn_count == 8 for _, record in compositions)
    for style_index, speech_style in enumerate(SpeechStyle):
        first_plan, first_record = compositions[style_index * 2]
        second_plan, second_record = compositions[style_index * 2 + 1]
        assert set(first_plan.segment_record_ids) == set(second_plan.segment_record_ids)
        assert first_plan.segment_record_ids != second_plan.segment_record_ids
        assert (
            first_record.metadata.split.leakage_group_id
            == second_record.metadata.split.leakage_group_id
        )
        assert first_record.metadata.scenario.speech_style is speech_style
        assert second_record.metadata.scenario.speech_style is speech_style


def test_bucket_selection_keeps_one_record_per_bucket_and_style() -> None:
    source_records = tuple(
        _source_record(
            record_id=f"{bucket.value}-{speech_style.value}-{candidate_index}",
            family=bucket.value,
            speech_style=speech_style,
            split=_split_for_style(speech_style),
        )
        for bucket in SegmentBucket
        for speech_style in SpeechStyle
        for candidate_index in range(2)
    )

    selected = select_bucket_calibration_segments(source_records)

    assert len(selected) == 40
    assert all(record.record_id.endswith("-0") for record in selected)


def test_training_compositions_mix_buckets_without_reusing_source_segments() -> None:
    source_records = tuple(
        _source_record(
            record_id=f"{bucket.value}-train-{candidate_index}",
            family=bucket.value,
            speech_style=SpeechStyle.CLEAN,
            split=DatasetSplit.TRAIN,
        )
        for bucket in SegmentBucket
        for candidate_index in range(4)
    )

    compositions = build_training_compositions(
        source_records=source_records,
        split=DatasetSplit.TRAIN,
        logical_epoch=1,
        random_seed=83,
        minimum_user_turns=8,
        maximum_user_turns=16,
    )

    selected_ids = tuple(
        source_id for plan, _ in compositions for source_id in plan.segment_record_ids
    )
    assert len(selected_ids) == len(set(selected_ids))
    assert set(selected_ids) == {record.record_id for record in source_records}
    assert all(plan.logical_epoch == 1 for plan, _ in compositions)
    assert all(8 <= record.metadata.scenario.user_turn_count <= 16 for _, record in compositions)
    source_by_id = {record.record_id: record for record in source_records}
    assert all(
        len(
            {
                source_by_id[source_id].metadata.scenario.family
                for source_id in plan.segment_record_ids
            }
        )
        >= 2
        for plan, _ in compositions
    )


def test_training_compositions_are_reproducible_and_change_across_epochs() -> None:
    source_records = tuple(
        _source_record(
            record_id=f"{bucket.value}-train-{candidate_index}",
            family=bucket.value,
            speech_style=SpeechStyle.CLEAN,
            split=DatasetSplit.TRAIN,
        )
        for bucket in SegmentBucket
        for candidate_index in range(3)
    )

    first = build_training_compositions(
        source_records=source_records,
        split=DatasetSplit.TRAIN,
        logical_epoch=0,
        random_seed=89,
        minimum_user_turns=8,
        maximum_user_turns=16,
    )
    repeated = build_training_compositions(
        source_records=source_records,
        split=DatasetSplit.TRAIN,
        logical_epoch=0,
        random_seed=89,
        minimum_user_turns=8,
        maximum_user_turns=16,
    )
    next_epoch = build_training_compositions(
        source_records=source_records,
        split=DatasetSplit.TRAIN,
        logical_epoch=1,
        random_seed=89,
        minimum_user_turns=8,
        maximum_user_turns=16,
    )

    assert tuple(plan for plan, _ in first) == tuple(plan for plan, _ in repeated)
    assert tuple(plan.segment_record_ids for plan, _ in first) != tuple(
        plan.segment_record_ids for plan, _ in next_epoch
    )


def test_review_selection_covers_every_behavior_bucket() -> None:
    source_records = tuple(
        _source_record(
            record_id=f"{bucket.value}-review-{candidate_index}",
            family=bucket.value,
            speech_style=SpeechStyle.CLEAN,
            split=DatasetSplit.TRAIN,
        )
        for bucket in SegmentBucket
        for candidate_index in range(2)
    )

    selected = select_review_records(
        candidate_records=source_records,
        count=10,
        random_seed=97,
    )
    repeated = select_review_records(
        candidate_records=source_records,
        count=10,
        random_seed=97,
    )

    assert selected == repeated
    assert len(selected) == 10
    assert {record.metadata.scenario.family for record in selected} == {
        bucket.value for bucket in SegmentBucket
    }


def _source_record(
    record_id: str,
    family: str,
    speech_style: SpeechStyle,
    split: DatasetSplit,
) -> ToolDialogueRecord:
    return ToolDialogueRecord(
        record_id=record_id,
        tools=runtime_tool_definitions(),
        messages=(
            SystemMessage(
                message_id=f"{record_id}-system",
                text="Be concise and conversational.",
            ),
            UserMessage(
                message_id=f"{record_id}-user-0",
                text="First thought.",
                speech_style=speech_style,
            ),
            AssistantMessage(
                message_id=f"{record_id}-assistant-0",
                audible_text="That makes sense.",
            ),
            UserMessage(
                message_id=f"{record_id}-user-1",
                text="One more detail.",
                speech_style=speech_style,
            ),
            AssistantMessage(
                message_id=f"{record_id}-assistant-1",
                audible_text="I’d keep that detail.",
            ),
        ),
        metadata=RecordMetadata(
            generation=GenerationProvenance(
                method="synthetic",
                model_identifier="teacher",
                model_revision="revision",
                quantization="fp8",
                prompt_revision="test",
                random_seed=1,
                time_reference="2026-07-18T12:00:00+00:00",
            ),
            scenario=ScenarioMetadata(
                family=family,
                tool_need="not_needed",
                expected_tool_names=(),
                outcome="not_applicable",
                length_band=LengthBand.MEDIUM,
                speech_style=speech_style,
                user_turn_count=2,
            ),
            split=SplitMetadata(
                name=split,
                policy_revision="test",
                leakage_group_id=record_id,
            ),
            audit=AuditMetadata(
                source_license="project-synthetic",
                intended_license="project-owned",
                human_review="pending",
                content_sha256="0" * 64,
            ),
        ),
    )


def _split_for_style(speech_style: SpeechStyle) -> DatasetSplit:
    match speech_style:
        case SpeechStyle.CLEAN | SpeechStyle.ASR_FRAGMENT | SpeechStyle.REPAIR:
            return DatasetSplit.TRAIN
        case SpeechStyle.CASUAL:
            return DatasetSplit.VALIDATION
        case SpeechStyle.FORMAL:
            return DatasetSplit.TEST
