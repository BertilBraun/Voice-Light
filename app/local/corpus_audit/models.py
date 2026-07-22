from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.local.training_samples.models import TrainingSamplePropositionKind
from app.shared.base_model import FrozenBaseModel


class CorpusAuditRejectionReason(StrEnum):
    MISSING_REGION_ANALYSIS = "missing_region_analysis"
    INSUFFICIENT_SUPERVISED_DURATION = "insufficient_supervised_duration"
    INSUFFICIENT_PRIMARY_SUPERVISION = "insufficient_primary_supervision"
    INSUFFICIENT_FUTURE_SUPERVISION = "insufficient_future_supervision"
    EXCESSIVE_MASKING = "excessive_masking"


class CorpusAuditRequest(FrozenBaseModel):
    dataset_ids: tuple[UUID, ...] = Field(min_length=1)
    minimum_quality: float = Field(default=0.95, ge=0.0, le=1.0)
    input_duration_seconds: float = Field(default=20.0, gt=0.0)
    burn_in_seconds: float = Field(default=4.0, ge=0.0)
    stride_seconds: float = Field(default=16.0, gt=0.0)
    minimum_supervised_seconds: float = Field(default=8.0, gt=0.0)
    minimum_primary_supervision_ratio: float = Field(default=0.9, ge=0.0, le=1.0)
    minimum_future_supervision_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    maximum_masked_ratio: float = Field(default=0.35, ge=0.0, le=1.0)


class CorpusAuditCategorySummary(FrozenBaseModel):
    kind: TrainingSamplePropositionKind
    accepted_window_count: int = Field(ge=0)
    available_oriented_event_count: int = Field(ge=0)
    covered_oriented_event_count: int = Field(ge=0)


class CorpusAuditRejectionSummary(FrozenBaseModel):
    reason: CorpusAuditRejectionReason
    window_count: int = Field(ge=0)


class CorpusAuditPhysicalEventCounts(FrozenBaseModel):
    turn_shift_count: int = Field(ge=0)
    pause_count: int = Field(ge=0)
    backchannel_count: int = Field(ge=0)
    interruption_count: int = Field(ge=0)
    overlap_count: int = Field(ge=0)


class CorpusAuditDatasetSummary(FrozenBaseModel):
    dataset_id: UUID
    dataset_name: str
    conversation_count: int = Field(ge=0)
    accepted_conversation_count: int = Field(ge=0)
    source_duration_seconds: float = Field(ge=0.0)
    usable_source_duration_seconds: float = Field(ge=0.0)
    candidate_window_count: int = Field(ge=0)
    accepted_window_count: int = Field(ge=0)
    input_duration_seconds: float = Field(ge=0.0)
    supervised_duration_seconds: float = Field(ge=0.0)
    effective_supervised_duration_seconds: float = Field(ge=0.0)
    masked_duration_seconds: float = Field(ge=0.0)
    physical_events: CorpusAuditPhysicalEventCounts


class CorpusAuditConversationSummary(FrozenBaseModel):
    dataset_id: UUID
    dataset_name: str
    sample_id: UUID
    external_id: str
    quality_score: float
    source_duration_seconds: float = Field(ge=0.0)
    usable_source_duration_seconds: float = Field(ge=0.0)
    candidate_window_count: int = Field(ge=0)
    accepted_window_count: int = Field(ge=0)
    effective_supervised_duration_seconds: float = Field(ge=0.0)
    masked_duration_seconds: float = Field(ge=0.0)


class CorpusAuditPilotMetric(FrozenBaseModel):
    label: str
    current: float = Field(ge=0.0)
    target: float = Field(gt=0.0)
    unit: str
    ready: bool


class CorpusAuditReport(FrozenBaseModel):
    generated_at: datetime
    config: CorpusAuditRequest
    dataset_summaries: tuple[CorpusAuditDatasetSummary, ...]
    conversations: tuple[CorpusAuditConversationSummary, ...]
    categories: tuple[CorpusAuditCategorySummary, ...]
    rejections: tuple[CorpusAuditRejectionSummary, ...]
    pilot_metrics: tuple[CorpusAuditPilotMetric, ...]
    conversation_count: int = Field(ge=0)
    accepted_conversation_count: int = Field(ge=0)
    candidate_window_count: int = Field(ge=0)
    accepted_window_count: int = Field(ge=0)
    input_duration_seconds: float = Field(ge=0.0)
    supervised_duration_seconds: float = Field(ge=0.0)
    effective_supervised_duration_seconds: float = Field(ge=0.0)
    masked_duration_seconds: float = Field(ge=0.0)
    physical_events: CorpusAuditPhysicalEventCounts
