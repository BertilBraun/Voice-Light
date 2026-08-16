from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.local.db.models import TrackSide
from app.shared.base_model import FrozenBaseModel


class CorpusReviewStatus(StrEnum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"


class CorpusReviewDatasetSelection(FrozenBaseModel):
    dataset_id: UUID
    minimum_quality: float = Field(ge=0.0, le=1.0)


class CorpusReviewSetRequest(FrozenBaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    seed: str = Field(min_length=1, max_length=200)
    items_per_dataset: int = Field(default=3, ge=1, le=10)
    datasets: tuple[CorpusReviewDatasetSelection, ...] = Field(min_length=1)


class PlannedCorpusReviewItem(FrozenBaseModel):
    dataset_id: UUID
    dataset_name: str
    sample_id: UUID
    external_id: str
    quality_score: float
    user_side: TrackSide
    start_seconds: float = Field(ge=0.0)


class CorpusReviewSetRecord(FrozenBaseModel):
    id: UUID
    name: str
    seed: str
    items_per_dataset: int
    config: CorpusReviewSetRequest
    created_at: datetime
    updated_at: datetime


class CorpusReviewItemRecord(FrozenBaseModel):
    id: UUID
    review_set_id: UUID
    dataset_id: UUID
    dataset_name: str
    sample_id: UUID
    external_id: str
    quality_score: float
    user_side: TrackSide
    start_seconds: float
    audio_status: CorpusReviewStatus
    annotation_status: CorpusReviewStatus
    label_status: CorpusReviewStatus
    overall_status: CorpusReviewStatus
    notes: str
    created_at: datetime
    updated_at: datetime


class CorpusReviewPlan(FrozenBaseModel):
    review_set: CorpusReviewSetRecord
    items: tuple[CorpusReviewItemRecord, ...]


class CorpusReviewDecision(FrozenBaseModel):
    audio_status: CorpusReviewStatus
    annotation_status: CorpusReviewStatus
    label_status: CorpusReviewStatus
    overall_status: CorpusReviewStatus
    notes: str = Field(max_length=4000)
