from pathlib import Path
from uuid import uuid4

from app.local.alignment_migration.riff import PcmWaveMetadata, RiffChunk
from app.local.timeline_repair.materialize import materialized_segments, materialized_words
from app.local.timeline_repair.models import (
    TimelineRepairPlanRecord,
    TimelineRepairScope,
    TransitionLocationSource,
)
from app.shared.asr import TimestampedWord


def test_global_materialization_pads_and_crops_to_canonical_length() -> None:
    segments = materialized_segments(
        plan=_plan(
            scope=TimelineRepairScope.GLOBAL_OFFSET,
            first_shift_seconds=2.0,
            second_shift_seconds=2.0,
        ),
        metadata=_metadata(frame_count=10, sample_rate=1),
    )

    assert [(segment.output_frame_count, segment.source_start_frame) for segment in segments] == [
        (2, None),
        (8, 0),
    ]


def test_piecewise_materialization_uses_both_shifts_and_silences_exclusion() -> None:
    segments = materialized_segments(
        plan=_plan(
            scope=TimelineRepairScope.AFTER_CHANGE_POINT,
            first_shift_seconds=1.0,
            second_shift_seconds=-1.0,
            change_point_seconds=5.0,
            exclusion_start_seconds=4.0,
            exclusion_end_seconds=6.0,
        ),
        metadata=_metadata(frame_count=10, sample_rate=1),
    )

    assert [(segment.output_frame_count, segment.source_start_frame) for segment in segments] == [
        (1, None),
        (3, 0),
        (2, None),
        (3, 7),
        (1, None),
    ]


def test_materialized_words_shift_each_side_of_transition_and_drop_straddler() -> None:
    plan = _plan(
        scope=TimelineRepairScope.AFTER_CHANGE_POINT,
        first_shift_seconds=1.0,
        second_shift_seconds=-1.0,
        change_point_seconds=5.0,
        exclusion_start_seconds=4.0,
        exclusion_end_seconds=6.0,
    )

    words = materialized_words(
        (
            TimestampedWord(text="early", start_seconds=1.0, end_seconds=2.0),
            TimestampedWord(text="crossing", start_seconds=4.5, end_seconds=5.5),
            TimestampedWord(text="late", start_seconds=7.0, end_seconds=8.0),
        ),
        plan=plan,
        duration_seconds=10.0,
    )

    assert [(word.text, word.start_seconds, word.end_seconds) for word in words] == [
        ("early", 2.0, 3.0),
        ("late", 6.0, 7.0),
    ]


def _plan(
    scope: TimelineRepairScope,
    first_shift_seconds: float,
    second_shift_seconds: float,
    change_point_seconds: float | None = None,
    exclusion_start_seconds: float | None = None,
    exclusion_end_seconds: float | None = None,
) -> TimelineRepairPlanRecord:
    identifier = uuid4()
    return TimelineRepairPlanRecord(
        id=uuid4(),
        sample_id=uuid4(),
        external_id="sample_test",
        duration_seconds=10.0,
        plan_version="test",
        plan_fingerprint="a" * 64,
        repair_scope=scope,
        first_part_shift_seconds=first_shift_seconds,
        second_part_shift_seconds=second_shift_seconds,
        change_point_seconds=change_point_seconds,
        transition_location_source=(
            TransitionLocationSource.MANUAL
            if scope is TimelineRepairScope.AFTER_CHANGE_POINT
            else None
        ),
        exclusion_start_seconds=exclusion_start_seconds,
        exclusion_end_seconds=exclusion_end_seconds,
        speaker1_audio_sha256="b" * 64,
        speaker2_audio_sha256="c" * 64,
        quality_result_id=uuid4(),
        speaker1_parakeet_transcript_id=identifier,
        speaker2_parakeet_transcript_id=uuid4(),
        speaker1_canary_transcript_id=uuid4(),
        speaker2_canary_transcript_id=uuid4(),
        derived_annotation_version=None,
        derived_annotation=None,
        conversation_regions_version=None,
        conversation_regions=None,
        created_at="2026-01-01T00:00:00Z",
    )


def _metadata(frame_count: int, sample_rate: int) -> PcmWaveMetadata:
    data_chunk = RiffChunk(
        identifier=b"data",
        header_offset=36,
        payload_offset=44,
        payload_size=frame_count * 2,
    )
    return PcmWaveMetadata(
        path=Path("test.wav"),
        file_size=44 + frame_count * 2,
        format_tag=1,
        channel_count=1,
        sample_rate=sample_rate,
        byte_rate=sample_rate * 2,
        block_alignment=2,
        bits_per_sample=16,
        frame_count=frame_count,
        chunks=(data_chunk,),
        data_chunk=data_chunk,
    )
