from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from app.local.conversation_regions.models import ConversationRegionAnalysis
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import ConversationAnnotation

TIMELINE_REPAIR_PLAN_VERSION = "immutable-virtual-timeline-v1"
TIMELINE_MATERIALIZATION_VERSION = "physical-timeline-v1"


class TimelineRepairScope(StrEnum):
    GLOBAL_OFFSET = "global_offset"
    AFTER_CHANGE_POINT = "after_change_point"


class TransitionLocationSource(StrEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class TimelineRepairSource(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    duration_seconds: float = Field(gt=0.0)
    repair_scope: TimelineRepairScope
    first_part_shift_seconds: float = Field(ge=-12.0, le=12.0)
    second_part_shift_seconds: float = Field(ge=-12.0, le=12.0)
    change_point_seconds: float | None
    transition_location_source: TransitionLocationSource | None
    exclusion_start_seconds: float | None
    exclusion_end_seconds: float | None
    speaker1_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker2_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_result_id: UUID
    speaker1_parakeet_transcript_id: UUID
    speaker2_parakeet_transcript_id: UUID
    speaker1_canary_transcript_id: UUID
    speaker2_canary_transcript_id: UUID

    @model_validator(mode="after")
    def validate_timeline(self) -> TimelineRepairSource:
        if self.repair_scope is TimelineRepairScope.GLOBAL_OFFSET:
            if self.first_part_shift_seconds != self.second_part_shift_seconds:
                raise ValueError("Global repair shifts must be identical.")
            if any(
                value is not None
                for value in (
                    self.change_point_seconds,
                    self.transition_location_source,
                    self.exclusion_start_seconds,
                    self.exclusion_end_seconds,
                )
            ):
                raise ValueError("Global repairs cannot define a transition.")
            return self
        transition_values = (
            self.change_point_seconds,
            self.transition_location_source,
            self.exclusion_start_seconds,
            self.exclusion_end_seconds,
        )
        if any(value is None for value in transition_values):
            raise ValueError("Transition repairs require a location and exclusion interval.")
        assert self.change_point_seconds is not None
        assert self.exclusion_start_seconds is not None
        assert self.exclusion_end_seconds is not None
        if (
            not self.exclusion_start_seconds
            <= self.change_point_seconds
            <= self.exclusion_end_seconds
        ):
            raise ValueError("The transition must fall inside its exclusion interval.")
        if self.exclusion_end_seconds > self.duration_seconds:
            raise ValueError("The exclusion interval exceeds the recording duration.")
        return self


class TimelineRepairPlanCreate(FrozenBaseModel):
    plan_version: str = Field(min_length=1, max_length=200)
    source: TimelineRepairSource

    def fingerprint(self) -> str:
        canonical = self.model_dump_json(exclude={"source": {"external_id", "duration_seconds"}})
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TimelineRepairDerivedArtifacts(FrozenBaseModel):
    annotation_version: str = Field(min_length=1, max_length=200)
    annotation: ConversationAnnotation
    conversation_regions_version: str = Field(min_length=1, max_length=200)
    conversation_regions: ConversationRegionAnalysis


class TimelineRepairPlanRecord(TimelineRepairSource):
    id: UUID
    plan_version: str
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    derived_annotation_version: str | None
    derived_annotation: ConversationAnnotation | None
    conversation_regions_version: str | None
    conversation_regions: ConversationRegionAnalysis | None
    materialization_version: str | None = None
    materialized_speaker2_audio_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    materialized_prepared_audio_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    materialized_at: datetime | None = None
    created_at: datetime

    @property
    def is_materialized(self) -> bool:
        return self.materialized_at is not None
