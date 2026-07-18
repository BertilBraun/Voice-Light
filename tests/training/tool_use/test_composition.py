from __future__ import annotations

from app.training.tool_use.composition import (
    build_bucket_calibration_compositions,
    select_bucket_calibration_segments,
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
