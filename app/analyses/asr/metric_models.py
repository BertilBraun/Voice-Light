from __future__ import annotations

from pydantic import Field

from app.asr.alignment import WordErrorCounts
from app.asr.transcript import JsonObject, RuntimeStats
from app.frozen_base_config import FrozenBaseModel


class ClassMetric(FrozenBaseModel):
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


class TimestampMetric(FrozenBaseModel):
    count: int
    median_absolute_error: float | None
    mean_absolute_error: float | None
    p90_absolute_error: float | None
    within_50ms: float | None
    within_100ms: float | None
    within_200ms: float | None
    greater_than_500ms: float | None
    median_bias: float | None


class TimestampMetrics(FrozenBaseModel):
    start: TimestampMetric
    end: TimestampMetric


class DisfluencyMetrics(FrozenBaseModel):
    repetition_reference_events: int
    repetition_predicted_events: int
    repetition_recalled_events: int
    repetition_recall: float
    partial_reference_events: int
    partial_predicted_events: int
    partial_recalled_events: int
    partial_recall: float


class FileMetrics(FrozenBaseModel):
    model_name: str
    audio_path: str
    word_error_counts: WordErrorCounts
    filler_metrics: dict[str, ClassMetric]
    disfluency_metrics: DisfluencyMetrics
    timestamp_metrics: TimestampMetrics
    runtime: RuntimeStats
    largest_timestamp_errors: list[JsonObject] = Field(default_factory=list)
