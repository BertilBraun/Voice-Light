from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from app.local.db.models import TrackSide
from app.shared.base_model import FrozenBaseModel


class CorpusReviewStatus(StrEnum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"


def validate_review_status_combination(
    audio_status: CorpusReviewStatus,
    annotation_status: CorpusReviewStatus,
    label_status: CorpusReviewStatus,
    overall_status: CorpusReviewStatus,
) -> None:
    component_statuses = (audio_status, annotation_status, label_status)
    if overall_status is CorpusReviewStatus.PASS and any(
        status is not CorpusReviewStatus.PASS for status in component_statuses
    ):
        raise ValueError("Overall pass requires all component decisions to pass.")
    if overall_status is CorpusReviewStatus.FAIL and all(
        status is not CorpusReviewStatus.FAIL for status in component_statuses
    ):
        raise ValueError("Overall failure requires at least one failed component.")


class CorpusReviewGateIssueCode(StrEnum):
    INCOMPLETE_SET = "incomplete_set"
    DUPLICATE_RECORDING = "duplicate_recording"
    INCOMPLETE_REVIEW = "incomplete_review"
    FAILED_REVIEW = "failed_review"
    STALE_PROVENANCE = "stale_provenance"


class CorpusReviewSelectionAlgorithm(StrEnum):
    DATASET_ID_SHA256_V1 = "dataset-id-sha256-v1"
    DATASET_NAME_SHA256_V2 = "dataset-name-sha256-v2"


class CorpusReviewDatasetSelection(FrozenBaseModel):
    dataset_id: UUID
    minimum_quality: float = Field(ge=0.0, le=1.0)


class CorpusReviewSetRequest(FrozenBaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    seed: str = Field(min_length=1, max_length=200)
    items_per_dataset: int = Field(default=3, ge=1, le=10)
    selection_algorithm: CorpusReviewSelectionAlgorithm = (
        CorpusReviewSelectionAlgorithm.DATASET_NAME_SHA256_V2
    )
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
    quality_result_id: UUID
    conversation_region_result_id: UUID
    speaker1_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker2_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_metric_version: str
    annotation_version: str
    region_analysis_version: str
    training_label_version: str
    input_duration_seconds: float = Field(gt=0.0)
    frame_seconds: float = Field(gt=0.0)
    provenance_current: bool
    user_side: TrackSide
    start_seconds: float
    audio_status: CorpusReviewStatus
    annotation_status: CorpusReviewStatus
    label_status: CorpusReviewStatus
    overall_status: CorpusReviewStatus
    notes: str
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_statuses(self) -> CorpusReviewItemRecord:
        validate_review_status_combination(
            audio_status=self.audio_status,
            annotation_status=self.annotation_status,
            label_status=self.label_status,
            overall_status=self.overall_status,
        )
        return self


class CorpusReviewPlan(FrozenBaseModel):
    review_set: CorpusReviewSetRecord
    items: tuple[CorpusReviewItemRecord, ...]


class CorpusReviewDecision(FrozenBaseModel):
    audio_status: CorpusReviewStatus
    annotation_status: CorpusReviewStatus
    label_status: CorpusReviewStatus
    overall_status: CorpusReviewStatus
    notes: str = Field(max_length=4000)

    @model_validator(mode="after")
    def validate_overall_status(self) -> CorpusReviewDecision:
        validate_review_status_combination(
            audio_status=self.audio_status,
            annotation_status=self.annotation_status,
            label_status=self.label_status,
            overall_status=self.overall_status,
        )
        return self


class CorpusReviewGateIssue(FrozenBaseModel):
    code: CorpusReviewGateIssueCode
    message: str


class CorpusReviewReadiness(FrozenBaseModel):
    review_set_name: str
    expected_item_count: int = Field(ge=0)
    item_count: int = Field(ge=0)
    passed_item_count: int = Field(ge=0)
    failed_item_count: int = Field(ge=0)
    pending_item_count: int = Field(ge=0)
    ready_to_publish: bool
    issues: tuple[CorpusReviewGateIssue, ...]
