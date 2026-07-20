from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

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
    reasons: tuple[ConversationRegionReason, ...]


class ConversationRegionAnalysis(FrozenBaseModel):
    analysis_version: str
    annotation_version: str
    config: ConversationRegionConfig
    duration_seconds: float = Field(gt=0.0)
    usable_duration_seconds: float = Field(ge=0.0)
    unusable_duration_seconds: float = Field(ge=0.0)
    usable_ratio: float = Field(ge=0.0, le=1.0)
    unusable_regions: tuple[UnusableConversationRegion, ...]


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
