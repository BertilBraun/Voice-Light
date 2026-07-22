from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.shared.base_model import FrozenBaseModel
from app.shared.language import LanguageProbeWindow, TrackLanguageStatus
from app.shared.quality import TrackVadResult


class DatasetStorageKind(StrEnum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


class TrackSide(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_ASR = "waiting_for_asr"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AsrRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SampleLanguageStatus(StrEnum):
    ENGLISH = "english"
    NON_ENGLISH = "non_english"
    INCONCLUSIVE = "inconclusive"


class TrackLanguageAssessment(FrozenBaseModel):
    sample_track_id: UUID
    source_audio_sha256: str
    assessment_version: str
    status: TrackLanguageStatus
    language_code: str | None
    confidence: float | None
    transcript_word_count: int
    transcript_text: str
    probe_windows: tuple[LanguageProbeWindow, ...]
    error: str | None


class DatasetCreate(FrozenBaseModel):
    name: str
    storage_kind: DatasetStorageKind = DatasetStorageKind.LOCAL
    root_uri: str
    description: str = ""


class DatasetRecord(FrozenBaseModel):
    id: UUID
    name: str
    storage_kind: DatasetStorageKind
    root_uri: str
    description: str
    created_at: datetime
    updated_at: datetime


class SampleRecord(FrozenBaseModel):
    id: UUID
    dataset_id: UUID
    external_id: str
    duration_seconds: float | None
    quality_score: float | None
    quality_flags: tuple[str, ...]
    is_unusable: bool
    created_at: datetime
    updated_at: datetime


class SampleTrackRecord(FrozenBaseModel):
    id: UUID
    sample_id: UUID
    side: TrackSide
    speaker_index: int
    storage_uri: str
    access_uri: str
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    sample_count: int | None
    audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int | None = None
    source_etag: str | None = None
    created_at: datetime
    updated_at: datetime


class AudioMetadataRecord(FrozenBaseModel):
    id: UUID
    sample_track_id: UUID
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_count: int | None
    payload: dict[str, object]
    created_at: datetime
    updated_at: datetime


class TrackVadRecord(FrozenBaseModel):
    sample_track_id: UUID
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vad_version: str
    result: TrackVadResult
    created_at: datetime
    updated_at: datetime


class QualityResultRecord(FrozenBaseModel):
    id: UUID
    sample_id: UUID
    metric_version: str
    status: str
    total_quality_score: float | None
    raw_quality_score: float | None
    interaction_density_score: float | None
    timing_reliability_score: float | None
    audio_quality_score: float | None
    conversation_quality_score: float | None
    interaction_count: int | None
    speech_segment_count: int | None
    turn_count: int | None
    turn_taking_count: int | None
    pause_count: int | None
    backchannel_count: int | None
    interruption_count: int | None
    usable_event_count: int | None
    annotation_duration_seconds: float | None
    represented_duration_seconds: float | None
    estimated_speech_segment_count: int | None
    estimated_interaction_count: int | None
    estimated_turn_count: int | None
    estimated_turn_taking_count: int | None
    estimated_pause_count: int | None
    estimated_backchannel_count: int | None
    estimated_interruption_count: int | None
    estimated_usable_event_count: int | None
    conversation_events_per_hour: float | None
    speech_ratio: float | None
    silence_ratio: float | None
    overlap_ratio: float | None
    duration_mismatch_seconds: float | None
    track_correlation: float | None
    energy_envelope_correlation: float | None
    speaker1_parakeet_full_asr_transcript_id: UUID | None
    speaker2_parakeet_full_asr_transcript_id: UUID | None
    speaker1_canary_full_asr_transcript_id: UUID | None
    speaker2_canary_full_asr_transcript_id: UUID | None
    flags: tuple[str, ...]
    payload: dict[str, object]
    created_at: datetime
    updated_at: datetime


class QualityResultSummaryRecord(FrozenBaseModel):
    id: UUID
    sample_id: UUID
    metric_version: str
    annotation_version: str | None
    status: str
    total_quality_score: float | None
    speech_ratio: float | None
    overlap_ratio: float | None
    has_parakeet_transcript_pair: bool
    has_canary_transcript_pair: bool
    created_at: datetime


class DashboardSampleSummary(FrozenBaseModel):
    sample: SampleRecord
    latest_quality: QualityResultSummaryRecord | None
    language_status: SampleLanguageStatus | None


class QualityVersionCount(FrozenBaseModel):
    metric_version: str
    annotation_version: str | None
    status: str
    sample_count: int


class FullAsrCoverageCount(FrozenBaseModel):
    cohort: str
    model_id: str
    side: TrackSide
    successful_transcript_count: int
    failed_transcript_count: int


class DatasetCompletenessSummary(FrozenBaseModel):
    dataset_id: UUID | None
    expected_metric_version: str
    expected_annotation_version: str
    sample_count: int
    current_quality_sample_count: int
    not_current_sample_count: int
    duration_excluded_sample_count: int
    reviewed_current_quality_sample_count: int
    unreviewed_current_quality_sample_count: int
    quality_versions: tuple[QualityVersionCount, ...]
    full_asr_coverage: tuple[FullAsrCoverageCount, ...]


class AsrRunRecord(FrozenBaseModel):
    id: UUID
    sample_id: UUID
    model_name: str
    status: AsrRunStatus
    runtime_seconds: float | None
    real_time_factor: float | None
    error: str | None
    payload: dict[str, object]
    created_at: datetime
    updated_at: datetime


class AsrWordRecord(FrozenBaseModel):
    id: UUID
    asr_run_id: UUID
    side: TrackSide
    word_index: int
    text: str
    start_seconds: float | None
    end_seconds: float | None
    confidence: float | None
    created_at: datetime


class AsrEvaluationRecord(FrozenBaseModel):
    id: UUID
    asr_run_id: UUID
    wer: float | None
    substitutions: int
    insertions: int
    deletions: int
    reference_word_count: int
    start_median_absolute_error: float | None
    start_mean_absolute_error: float | None
    start_p90_absolute_error: float | None
    end_median_absolute_error: float | None
    end_mean_absolute_error: float | None
    end_p90_absolute_error: float | None
    payload: dict[str, object]
    created_at: datetime
    updated_at: datetime


class IngestionJobRecord(FrozenBaseModel):
    id: UUID
    dataset_id: UUID | None
    status: JobStatus
    source_uri: str
    message: str
    total_samples: int
    processed_samples: int
    failed_samples: int
    error: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


class ConversationDatasetSummary(FrozenBaseModel):
    dataset_id: UUID | None
    analyzed_sample_count: int
    invalid_sample_count: int
    analyzed_duration_seconds: float
    represented_duration_seconds: float
    speech_segment_count: int
    interaction_count: int
    turn_count: int
    turn_taking_count: int
    pause_count: int
    backchannel_count: int
    interruption_count: int
    usable_event_count: int
    estimated_speech_segment_count: int
    estimated_interaction_count: int
    estimated_turn_count: int
    estimated_turn_taking_count: int
    estimated_pause_count: int
    estimated_backchannel_count: int
    estimated_interruption_count: int
    estimated_usable_event_count: int


class SampleListFilter(FrozenBaseModel):
    dataset_id: UUID | None = None
    quality_min: float | None = None
    quality_max: float | None = None
    duration_min: float | None = None
    duration_max: float | None = None
    flag: str | None = None
    speech_ratio_min: float | None = None
    speech_ratio_max: float | None = None
    overlap_ratio_min: float | None = None
    overlap_ratio_max: float | None = None
    silence_ratio_min: float | None = None
    silence_ratio_max: float | None = None
    asr_model: str | None = None
    wer_min: float | None = None
    wer_max: float | None = None
    timestamp_p90_max: float | None = None
    language_status: SampleLanguageStatus | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class DashboardSample(FrozenBaseModel):
    sample: SampleRecord
    tracks: tuple[SampleTrackRecord, ...]
    latest_quality: QualityResultRecord | None
    latest_asr_run: AsrRunRecord | None
    latest_asr_evaluation: AsrEvaluationRecord | None
    language_assessments: tuple[TrackLanguageAssessment, ...]
