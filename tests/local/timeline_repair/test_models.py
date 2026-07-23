from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from app.local.timeline_repair.models import (
    TimelineRepairPlanCreate,
    TimelineRepairScope,
    TimelineRepairSource,
    TransitionLocationSource,
)

SAMPLE_ID = UUID("00000000-0000-0000-0000-000000000001")
QUALITY_ID = UUID("00000000-0000-0000-0000-000000000002")
TRANSCRIPT_IDS = tuple(UUID(f"00000000-0000-0000-0000-{index:012d}") for index in range(3, 7))


def repair_source(**overrides: object) -> TimelineRepairSource:
    values: dict[str, object] = {
        "sample_id": SAMPLE_ID,
        "external_id": "sample_001",
        "duration_seconds": 1_000.0,
        "repair_scope": TimelineRepairScope.AFTER_CHANGE_POINT,
        "first_part_shift_seconds": 0.0,
        "second_part_shift_seconds": 1.5,
        "change_point_seconds": 500.0,
        "transition_location_source": TransitionLocationSource.MANUAL,
        "exclusion_start_seconds": 490.0,
        "exclusion_end_seconds": 510.0,
        "speaker1_audio_sha256": "a" * 64,
        "speaker2_audio_sha256": "b" * 64,
        "quality_result_id": QUALITY_ID,
        "speaker1_parakeet_transcript_id": TRANSCRIPT_IDS[0],
        "speaker2_parakeet_transcript_id": TRANSCRIPT_IDS[1],
        "speaker1_canary_transcript_id": TRANSCRIPT_IDS[2],
        "speaker2_canary_transcript_id": TRANSCRIPT_IDS[3],
    }
    values.update(overrides)
    return TimelineRepairSource.model_validate(values)


def test_fingerprint_ignores_display_metadata_but_covers_provenance() -> None:
    original = TimelineRepairPlanCreate(plan_version="timeline-v1", source=repair_source())
    renamed = TimelineRepairPlanCreate(
        plan_version="timeline-v1",
        source=repair_source(external_id="renamed", duration_seconds=2_000.0),
    )
    changed_hash = TimelineRepairPlanCreate(
        plan_version="timeline-v1",
        source=repair_source(speaker2_audio_sha256="c" * 64),
    )

    assert original.fingerprint() == renamed.fingerprint()
    assert original.fingerprint() != changed_hash.fingerprint()


def test_transition_requires_change_point_inside_exclusion() -> None:
    with pytest.raises(ValidationError, match="inside"):
        repair_source(change_point_seconds=520.0)


def test_global_repair_rejects_transition_fields() -> None:
    with pytest.raises(ValidationError, match="cannot define"):
        repair_source(
            repair_scope=TimelineRepairScope.GLOBAL_OFFSET,
            first_part_shift_seconds=1.5,
        )
