from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from app.shared.base_model import FrozenBaseModel

CONVERSATION_REGION_ANALYSIS_VERSION = "conversation-regions-v1"


class ConversationRegionReason(StrEnum):
    DUAL_SILENCE = "dual_silence"
    ONE_SIDED_ACTIVITY = "one_sided_activity"
    SLOW_TURN_EXCHANGE = "slow_turn_exchange"


class ConversationRegionConfig(FrozenBaseModel):
    minimum_dual_silence_seconds: float = Field(default=6.0, gt=0.0)
    maximum_one_sided_activity_seconds: float = Field(default=45.0, gt=0.0)
    minimum_present_speech_seconds: float = Field(default=3.0, gt=0.0)
    maximum_turn_exchange_gap_seconds: float = Field(default=3.0, gt=0.0)


class UnusableConversationRegion(FrozenBaseModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    reasons: tuple[ConversationRegionReason, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> UnusableConversationRegion:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Conversation region end must follow its start.")
        return self


class ConversationRegionAnalysis(FrozenBaseModel):
    analysis_version: str
    annotation_version: str
    config: ConversationRegionConfig
    duration_seconds: float = Field(gt=0.0)
    usable_duration_seconds: float = Field(ge=0.0)
    unusable_duration_seconds: float = Field(ge=0.0)
    usable_ratio: float = Field(ge=0.0, le=1.0)
    unusable_regions: tuple[UnusableConversationRegion, ...]

    @model_validator(mode="after")
    def validate_accounting(self) -> ConversationRegionAnalysis:
        previous_end_seconds = 0.0
        for region in self.unusable_regions:
            if region.start_seconds < previous_end_seconds:
                raise ValueError("Unusable conversation regions must be ordered and disjoint.")
            if region.end_seconds > self.duration_seconds + 0.08:
                raise ValueError("Unusable conversation region exceeds the recording duration.")
            previous_end_seconds = region.end_seconds
        calculated_unusable_seconds = sum(
            region.end_seconds - region.start_seconds for region in self.unusable_regions
        )
        if abs(calculated_unusable_seconds - self.unusable_duration_seconds) > 0.08:
            raise ValueError("Unusable conversation duration does not match its regions.")
        if (
            abs(
                self.usable_duration_seconds
                + self.unusable_duration_seconds
                - self.duration_seconds
            )
            > 0.08
        ):
            raise ValueError("Conversation region durations do not cover the recording.")
        calculated_ratio = self.usable_duration_seconds / self.duration_seconds
        if abs(calculated_ratio - self.usable_ratio) > 1e-6:
            raise ValueError("Conversation region usable ratio is inconsistent.")
        return self


class ConversationRegionResultRecord(FrozenBaseModel):
    sample_id: UUID
    analysis_version: str
    annotation_version: str
    analysis: ConversationRegionAnalysis
    created_at: datetime
    updated_at: datetime


class ConversationRegionReasonDuration(FrozenBaseModel):
    reason: ConversationRegionReason
    duration_seconds: float = Field(ge=0.0)


class ConversationRegionDatasetSummary(FrozenBaseModel):
    dataset_id: UUID | None
    analysis_version: str
    analyzed_sample_count: int = Field(ge=0)
    represented_duration_seconds: float = Field(ge=0.0)
    usable_duration_seconds: float = Field(ge=0.0)
    unusable_duration_seconds: float = Field(ge=0.0)
    usable_ratio: float = Field(ge=0.0, le=1.0)
    reason_durations: tuple[ConversationRegionReasonDuration, ...]
