from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

from pydantic import Field

from app.frozen_base_config import FrozenBaseModel

JsonScalar: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, JsonScalar]


class AlignmentOperation(StrEnum):
    EQUAL = "equal"
    SUBSTITUTE = "substitute"
    DELETE = "delete"
    INSERT = "insert"


class SpeakerTrack(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"


class Word(FrozenBaseModel):
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


class ReferenceTranscript(FrozenBaseModel):
    audio_path: str
    words: tuple[Word, ...]


class RuntimeStats(FrozenBaseModel):
    audio_duration_seconds: float
    processing_time_seconds: float
    model_loading_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    real_time_factor: float | None = None
    peak_gpu_memory_mb: float | None = None


class TranscriptionResult(FrozenBaseModel):
    model_name: str
    audio_path: str
    track: SpeakerTrack
    audio_duration_seconds: float
    processing_time_seconds: float
    words: tuple[Word, ...]
    raw_output: JsonObject = Field(default_factory=dict)
    model_identifier: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    model_loading_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    real_time_factor: float | None = None
    peak_gpu_memory_mb: float | None = None
    error: str | None = None


class AlignedWord(FrozenBaseModel):
    reference: Word | None
    prediction: Word | None
    operation: AlignmentOperation
    reference_token: str | None
    prediction_token: str | None


class WordErrorCounts(FrozenBaseModel):
    substitutions: int
    insertions: int
    deletions: int
    reference_words: int

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0
        return (self.substitutions + self.insertions + self.deletions) / self.reference_words


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
