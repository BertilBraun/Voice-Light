from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.db.models import (
    AsrEvaluationRecord,
    AsrRunRecord,
    AsrRunStatus,
    AsrWordRecord,
    AudioMetadataRecord,
    ConversationDatasetSummary,
    DashboardSample,
    DashboardSampleSummary,
    DatasetCompletenessSummary,
    DatasetCreate,
    DatasetRecord,
    FullAsrCoverageCount,
    IngestionJobRecord,
    JobStatus,
    QualityResultRecord,
    QualityResultSummaryRecord,
    QualityVersionCount,
    SampleLanguageStatus,
    SampleListFilter,
    SampleRecord,
    SampleTrackRecord,
    TrackLanguageAssessment,
    TrackSide,
)
from app.local.timeline_repair.models import TIMELINE_REPAIR_PLAN_VERSION
from app.shared.quality import METRIC_VERSION

INTERESTING_SAMPLE_POOL_SIZE = 64


@dataclass(frozen=True)
class SampleTrackInput:
    side: TrackSide
    speaker_index: int
    storage_uri: str
    access_uri: str
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    sample_count: int | None
    audio_sha256: str | None
    source_size_bytes: int | None = None
    source_etag: str | None = None


@dataclass(frozen=True)
class AudioMetadataInput:
    sample_track_id: UUID
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_count: int | None
    payload: dict[str, object]


@dataclass(frozen=True)
class QualityResultInput:
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


@dataclass(frozen=True)
class AsrRunInput:
    sample_id: UUID
    model_name: str
    status: AsrRunStatus
    runtime_seconds: float | None
    real_time_factor: float | None
    error: str | None
    payload: dict[str, object]


@dataclass(frozen=True)
class AsrWordInput:
    side: TrackSide
    word_index: int
    text: str
    start_seconds: float | None
    end_seconds: float | None
    confidence: float | None


@dataclass(frozen=True)
class AsrEvaluationInput:
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


@dataclass(frozen=True)
class DashboardFilterSql:
    where_clause: str
    parameters: tuple[object, ...]


@dataclass(frozen=True)
class AnnotatedSampleRecord:
    sample_id: UUID
    external_id: str
    represented_duration_seconds: float
    usable_event_count: int
    quality_score: float | None


class Repository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def upsert_dataset(self, payload: DatasetCreate) -> DatasetRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO datasets (name, storage_kind, root_uri, description, updated_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (name, root_uri) DO UPDATE
                SET storage_kind = EXCLUDED.storage_kind,
                    description = EXCLUDED.description,
                    updated_at = now()
                RETURNING *
                """,
                (payload.name, payload.storage_kind.value, payload.root_uri, payload.description),
            ).fetchone()
        assert row is not None
        return dataset_record(row)

    def list_datasets(self) -> list[DatasetRecord]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM datasets ORDER BY name, root_uri").fetchall()
        return [dataset_record(row) for row in rows]

    def get_dataset(self, dataset_id: UUID) -> DatasetRecord:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE id = %s", (dataset_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        return dataset_record(row)

    def create_ingestion_job(self, source_uri: str, message: str) -> IngestionJobRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO ingestion_jobs (status, source_uri, message)
                VALUES (%s, %s, %s)
                RETURNING *
                """,
                (JobStatus.QUEUED.value, source_uri, message),
            ).fetchone()
        assert row is not None
        return ingestion_job_record(row)

    def update_ingestion_job(
        self,
        job_id: UUID,
        status: JobStatus | None,
        message: str | None,
        dataset_id: UUID | None = None,
        total_samples: int | None = None,
        processed_samples: int | None = None,
        failed_samples: int | None = None,
        error: str | None = None,
    ) -> IngestionJobRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = COALESCE(%s, status),
                    message = COALESCE(%s, message),
                    dataset_id = COALESCE(%s, dataset_id),
                    total_samples = COALESCE(%s, total_samples),
                    processed_samples = COALESCE(%s, processed_samples),
                    failed_samples = COALESCE(%s, failed_samples),
                    error = COALESCE(%s, error),
                    updated_at = now(),
                    finished_at = CASE
                      WHEN %s IN (
                        'completed', 'failed', 'canceled', 'waiting_for_asr'
                      ) THEN now()
                      ELSE finished_at
                    END
                WHERE id = %s
                RETURNING *
                """,
                (
                    status.value if status is not None else None,
                    message,
                    dataset_id,
                    total_samples,
                    processed_samples,
                    failed_samples,
                    error,
                    status.value if status is not None else None,
                    job_id,
                ),
            ).fetchone()
        if row is None:
            raise ValueError(f"Ingestion job not found: {job_id}")
        return ingestion_job_record(row)

    def list_ingestion_jobs(self, limit: int) -> list[IngestionJobRecord]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [ingestion_job_record(row) for row in rows]

    def upsert_sample(self, dataset_id: UUID, external_id: str) -> SampleRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO samples (dataset_id, external_id, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (dataset_id, external_id) DO UPDATE
                SET updated_at = now()
                RETURNING *
                """,
                (dataset_id, external_id),
            ).fetchone()
        assert row is not None
        return sample_record(row)

    def completed_quality_sample_ids(
        self,
        dataset_id: UUID,
        metric_version: str,
    ) -> set[str]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT samples.external_id
                FROM samples
                JOIN quality_results ON quality_results.sample_id = samples.id
                WHERE samples.dataset_id = %s
                  AND quality_results.metric_version = %s
                  AND quality_results.status IN ('completed', 'invalid')
                """,
                (dataset_id, metric_version),
            ).fetchall()
        return {str(row["external_id"]) for row in rows}

    def update_sample_duration(self, sample_id: UUID, duration_seconds: float) -> SampleRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                UPDATE samples
                SET duration_seconds = %s, updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (duration_seconds, sample_id),
            ).fetchone()
        if row is None:
            raise ValueError(f"Sample not found: {sample_id}")
        return sample_record(row)

    def upsert_sample_track(self, sample_id: UUID, track: SampleTrackInput) -> SampleTrackRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO sample_tracks (
                  sample_id, side, speaker_index, storage_uri, access_uri,
                  duration_seconds, sample_rate, channels, sample_count, audio_sha256,
                  source_size_bytes, source_etag, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (sample_id, side) DO UPDATE
                SET speaker_index = EXCLUDED.speaker_index,
                    storage_uri = EXCLUDED.storage_uri,
                    access_uri = EXCLUDED.access_uri,
                    duration_seconds = COALESCE(
                      EXCLUDED.duration_seconds, sample_tracks.duration_seconds
                    ),
                    sample_rate = COALESCE(EXCLUDED.sample_rate, sample_tracks.sample_rate),
                    channels = COALESCE(EXCLUDED.channels, sample_tracks.channels),
                    sample_count = COALESCE(EXCLUDED.sample_count, sample_tracks.sample_count),
                    audio_sha256 = COALESCE(
                      EXCLUDED.audio_sha256, sample_tracks.audio_sha256
                    ),
                    source_size_bytes = COALESCE(
                      EXCLUDED.source_size_bytes, sample_tracks.source_size_bytes
                    ),
                    source_etag = COALESCE(EXCLUDED.source_etag, sample_tracks.source_etag),
                    updated_at = now()
                RETURNING *
                """,
                (
                    sample_id,
                    track.side.value,
                    track.speaker_index,
                    track.storage_uri,
                    track.access_uri,
                    track.duration_seconds,
                    track.sample_rate,
                    track.channels,
                    track.sample_count,
                    track.audio_sha256,
                    track.source_size_bytes,
                    track.source_etag,
                ),
            ).fetchone()
        assert row is not None
        return sample_track_record(row)

    def upsert_audio_metadata(self, metadata: AudioMetadataInput) -> AudioMetadataRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO audio_metadata (
                  sample_track_id, duration_seconds, sample_rate, channels,
                  sample_count, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (sample_track_id) DO UPDATE
                SET duration_seconds = EXCLUDED.duration_seconds,
                    sample_rate = EXCLUDED.sample_rate,
                    channels = EXCLUDED.channels,
                    sample_count = EXCLUDED.sample_count,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                RETURNING *
                """,
                (
                    metadata.sample_track_id,
                    metadata.duration_seconds,
                    metadata.sample_rate,
                    metadata.channels,
                    metadata.sample_count,
                    Jsonb(metadata.payload),
                ),
            ).fetchone()
        assert row is not None
        return audio_metadata_record(row)

    def insert_quality_result(self, quality_result: QualityResultInput) -> QualityResultRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO quality_results (
                  sample_id, metric_version, status, total_quality_score, raw_quality_score,
                  interaction_density_score, timing_reliability_score,
                  audio_quality_score, conversation_quality_score,
                  interaction_count, speech_segment_count, turn_count, turn_taking_count,
                  pause_count, backchannel_count, interruption_count, usable_event_count,
                  annotation_duration_seconds, represented_duration_seconds,
                  estimated_speech_segment_count, estimated_interaction_count,
                  estimated_turn_count, estimated_turn_taking_count, estimated_pause_count,
                  estimated_backchannel_count, estimated_interruption_count,
                  estimated_usable_event_count,
                  conversation_events_per_hour, speech_ratio, silence_ratio, overlap_ratio,
                  duration_mismatch_seconds, track_correlation, energy_envelope_correlation,
                  speaker1_parakeet_full_asr_transcript_id,
                  speaker2_parakeet_full_asr_transcript_id,
                  speaker1_canary_full_asr_transcript_id,
                  speaker2_canary_full_asr_transcript_id,
                  flags, payload
                )
                VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING *
                """,
                (
                    quality_result.sample_id,
                    quality_result.metric_version,
                    quality_result.status,
                    quality_result.total_quality_score,
                    quality_result.raw_quality_score,
                    quality_result.interaction_density_score,
                    quality_result.timing_reliability_score,
                    quality_result.audio_quality_score,
                    quality_result.conversation_quality_score,
                    quality_result.interaction_count,
                    quality_result.speech_segment_count,
                    quality_result.turn_count,
                    quality_result.turn_taking_count,
                    quality_result.pause_count,
                    quality_result.backchannel_count,
                    quality_result.interruption_count,
                    quality_result.usable_event_count,
                    quality_result.annotation_duration_seconds,
                    quality_result.represented_duration_seconds,
                    quality_result.estimated_speech_segment_count,
                    quality_result.estimated_interaction_count,
                    quality_result.estimated_turn_count,
                    quality_result.estimated_turn_taking_count,
                    quality_result.estimated_pause_count,
                    quality_result.estimated_backchannel_count,
                    quality_result.estimated_interruption_count,
                    quality_result.estimated_usable_event_count,
                    quality_result.conversation_events_per_hour,
                    quality_result.speech_ratio,
                    quality_result.silence_ratio,
                    quality_result.overlap_ratio,
                    quality_result.duration_mismatch_seconds,
                    quality_result.track_correlation,
                    quality_result.energy_envelope_correlation,
                    quality_result.speaker1_parakeet_full_asr_transcript_id,
                    quality_result.speaker2_parakeet_full_asr_transcript_id,
                    quality_result.speaker1_canary_full_asr_transcript_id,
                    quality_result.speaker2_canary_full_asr_transcript_id,
                    list(quality_result.flags),
                    Jsonb(quality_result.payload),
                ),
            ).fetchone()
            connection.execute(
                """
                UPDATE samples
                SET quality_score = %s,
                    quality_flags = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    quality_result.total_quality_score,
                    list(quality_result.flags),
                    quality_result.sample_id,
                ),
            )
        assert row is not None
        return quality_result_record(row)

    def insert_asr_run(self, asr_run: AsrRunInput) -> AsrRunRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO asr_runs (
                  sample_id, model_name, status, runtime_seconds,
                  real_time_factor, error, payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    asr_run.sample_id,
                    asr_run.model_name,
                    asr_run.status.value,
                    asr_run.runtime_seconds,
                    asr_run.real_time_factor,
                    asr_run.error,
                    Jsonb(asr_run.payload),
                ),
            ).fetchone()
        assert row is not None
        return asr_run_record(row)

    def insert_asr_words(
        self, asr_run_id: UUID, words: Sequence[AsrWordInput]
    ) -> list[AsrWordRecord]:
        if not words:
            return []
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO asr_words (
                      asr_run_id, side, word_index, text,
                      start_seconds, end_seconds, confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (asr_run_id, side, word_index) DO UPDATE
                    SET text = EXCLUDED.text,
                        start_seconds = EXCLUDED.start_seconds,
                        end_seconds = EXCLUDED.end_seconds,
                        confidence = EXCLUDED.confidence
                    """,
                    [
                        (
                            asr_run_id,
                            word.side.value,
                            word.word_index,
                            word.text,
                            word.start_seconds,
                            word.end_seconds,
                            word.confidence,
                        )
                        for word in words
                    ],
                )
            rows = connection.execute(
                "SELECT * FROM asr_words WHERE asr_run_id = %s ORDER BY side, word_index",
                (asr_run_id,),
            ).fetchall()
        return [asr_word_record(row) for row in rows]

    def upsert_asr_evaluation(self, evaluation: AsrEvaluationInput) -> AsrEvaluationRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO asr_evaluations (
                  asr_run_id, wer, substitutions, insertions, deletions, reference_word_count,
                  start_median_absolute_error, start_mean_absolute_error, start_p90_absolute_error,
                  end_median_absolute_error, end_mean_absolute_error, end_p90_absolute_error,
                  payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (asr_run_id) DO UPDATE
                SET wer = EXCLUDED.wer,
                    substitutions = EXCLUDED.substitutions,
                    insertions = EXCLUDED.insertions,
                    deletions = EXCLUDED.deletions,
                    reference_word_count = EXCLUDED.reference_word_count,
                    start_median_absolute_error = EXCLUDED.start_median_absolute_error,
                    start_mean_absolute_error = EXCLUDED.start_mean_absolute_error,
                    start_p90_absolute_error = EXCLUDED.start_p90_absolute_error,
                    end_median_absolute_error = EXCLUDED.end_median_absolute_error,
                    end_mean_absolute_error = EXCLUDED.end_mean_absolute_error,
                    end_p90_absolute_error = EXCLUDED.end_p90_absolute_error,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                RETURNING *
                """,
                (
                    evaluation.asr_run_id,
                    evaluation.wer,
                    evaluation.substitutions,
                    evaluation.insertions,
                    evaluation.deletions,
                    evaluation.reference_word_count,
                    evaluation.start_median_absolute_error,
                    evaluation.start_mean_absolute_error,
                    evaluation.start_p90_absolute_error,
                    evaluation.end_median_absolute_error,
                    evaluation.end_mean_absolute_error,
                    evaluation.end_p90_absolute_error,
                    Jsonb(evaluation.payload),
                ),
            ).fetchone()
        assert row is not None
        return asr_evaluation_record(row)

    def list_dashboard_samples(self, sample_filter: SampleListFilter) -> list[DashboardSample]:
        filter_sql = dashboard_filter_sql(sample_filter)
        parameters = (*filter_sql.parameters, sample_filter.limit, sample_filter.offset)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                  row_to_json(samples.*) AS sample_json,
                  COALESCE(track_rows.tracks_json, '[]'::json) AS tracks_json,
                  row_to_json(latest_quality.*) AS quality_json,
                  row_to_json(latest_asr_run.*) AS asr_run_json,
                  row_to_json(latest_asr_evaluation.*) AS asr_evaluation_json,
                  COALESCE(language_rows.assessments_json, '[]'::json)
                    AS language_assessments_json
                FROM samples
                LEFT JOIN LATERAL (
                  SELECT json_agg(sample_tracks.* ORDER BY speaker_index) AS tracks_json
                  FROM sample_tracks
                  WHERE sample_tracks.sample_id = samples.id
                ) AS track_rows ON true
                LEFT JOIN LATERAL (
                  SELECT *
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_quality ON true
                LEFT JOIN LATERAL (
                  SELECT *
                  FROM asr_runs
                  WHERE asr_runs.sample_id = samples.id
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_asr_run ON true
                LEFT JOIN asr_evaluations AS latest_asr_evaluation
                  ON latest_asr_evaluation.asr_run_id = latest_asr_run.id
                {language_assessments_lateral_sql()}
                {language_summary_lateral_sql()}
                {filter_sql.where_clause}
                ORDER BY samples.updated_at DESC, samples.external_id
                LIMIT %s OFFSET %s
                """,
                parameters,
            ).fetchall()
        return [dashboard_sample_from_row(row) for row in rows]

    def list_dashboard_sample_summaries(
        self,
        sample_filter: SampleListFilter,
    ) -> list[DashboardSampleSummary]:
        filter_sql = dashboard_filter_sql(sample_filter)
        parameters = (*filter_sql.parameters, sample_filter.limit, sample_filter.offset)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                  row_to_json(samples.*) AS sample_json,
                  latest_quality.id AS quality_id,
                  latest_quality.sample_id AS quality_sample_id,
                  latest_quality.metric_version,
                  latest_quality.payload -> 'conversation_annotation' ->> 'annotation_version'
                    AS annotation_version,
                  latest_quality.status AS quality_status,
                  latest_quality.total_quality_score,
                  latest_quality.speech_ratio,
                  latest_quality.overlap_ratio,
                  (
                    latest_quality.speaker1_parakeet_full_asr_transcript_id IS NOT NULL
                    AND latest_quality.speaker2_parakeet_full_asr_transcript_id IS NOT NULL
                  ) AS has_parakeet_transcript_pair,
                  (
                    latest_quality.speaker1_canary_full_asr_transcript_id IS NOT NULL
                    AND latest_quality.speaker2_canary_full_asr_transcript_id IS NOT NULL
                  ) AS has_canary_transcript_pair,
                  latest_quality.created_at AS quality_created_at,
                  language_summary.language_status
                FROM samples
                LEFT JOIN LATERAL (
                  SELECT *
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                  ORDER BY created_at DESC, id DESC
                  LIMIT 1
                ) AS latest_quality ON true
                LEFT JOIN LATERAL (
                  SELECT *
                  FROM asr_runs
                  WHERE asr_runs.sample_id = samples.id
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_asr_run ON true
                LEFT JOIN asr_evaluations AS latest_asr_evaluation
                  ON latest_asr_evaluation.asr_run_id = latest_asr_run.id
                {language_summary_lateral_sql()}
                {filter_sql.where_clause}
                ORDER BY samples.updated_at DESC, samples.external_id
                LIMIT %s OFFSET %s
                """,
                parameters,
            ).fetchall()
        return [dashboard_sample_summary_from_row(row) for row in rows]

    def dataset_completeness_summary(
        self,
        dataset_id: UUID | None,
        expected_metric_version: str,
        expected_annotation_version: str,
    ) -> DatasetCompletenessSummary:
        dataset_filter = "WHERE samples.dataset_id = %s" if dataset_id is not None else ""
        parameters: tuple[object, ...] = (dataset_id,) if dataset_id is not None else ()
        with self.connection() as connection:
            overview = connection.execute(
                f"""
                WITH selected_samples AS (
                  SELECT samples.id
                  FROM samples
                  {dataset_filter}
                ),
                current_quality AS (
                  SELECT DISTINCT quality_results.sample_id
                  FROM quality_results
                  JOIN selected_samples ON selected_samples.id = quality_results.sample_id
                  WHERE quality_results.metric_version = %s
                    AND quality_results.status = 'completed'
                ),
                duration_excluded AS (
                  SELECT DISTINCT quality_results.sample_id
                  FROM quality_results
                  JOIN selected_samples ON selected_samples.id = quality_results.sample_id
                  WHERE quality_results.status = 'invalid'
                    AND 'duration_mismatch_invalid' = ANY(quality_results.flags)
                )
                SELECT
                  COUNT(*)::integer AS sample_count,
                  COUNT(current_quality.sample_id)::integer AS current_quality_sample_count,
                  COUNT(*) FILTER (
                    WHERE current_quality.sample_id IS NULL
                  )::integer AS not_current_sample_count,
                  COUNT(duration_excluded.sample_id) FILTER (
                    WHERE current_quality.sample_id IS NULL
                  )::integer AS duration_excluded_sample_count,
                  COUNT(synchronization_reviews.sample_id) FILTER (
                    WHERE current_quality.sample_id IS NOT NULL
                  )::integer AS reviewed_current_quality_sample_count
                FROM selected_samples
                LEFT JOIN current_quality ON current_quality.sample_id = selected_samples.id
                LEFT JOIN duration_excluded ON duration_excluded.sample_id = selected_samples.id
                LEFT JOIN synchronization_reviews
                  ON synchronization_reviews.sample_id = selected_samples.id
                """,
                (*parameters, expected_metric_version),
            ).fetchone()
            assert overview is not None
            version_rows = connection.execute(
                f"""
                WITH selected_samples AS (
                  SELECT samples.id
                  FROM samples
                  {dataset_filter}
                ),
                latest_quality AS (
                  SELECT DISTINCT ON (quality_results.sample_id)
                    quality_results.sample_id,
                    quality_results.metric_version,
                    quality_results.status,
                    quality_results.payload
                      -> 'conversation_annotation'
                      ->> 'annotation_version' AS annotation_version
                  FROM quality_results
                  JOIN selected_samples ON selected_samples.id = quality_results.sample_id
                  ORDER BY quality_results.sample_id,
                           quality_results.created_at DESC,
                           quality_results.id DESC
                )
                SELECT metric_version, annotation_version, status, COUNT(*)::integer AS sample_count
                FROM latest_quality
                GROUP BY metric_version, annotation_version, status
                ORDER BY metric_version, annotation_version, status
                """,
                parameters,
            ).fetchall()
            coverage_rows = connection.execute(
                f"""
                WITH selected_samples AS (
                  SELECT samples.id
                  FROM samples
                  {dataset_filter}
                ),
                current_quality AS (
                  SELECT DISTINCT quality_results.sample_id
                  FROM quality_results
                  JOIN selected_samples ON selected_samples.id = quality_results.sample_id
                  WHERE quality_results.metric_version = %s
                    AND quality_results.status = 'completed'
                ),
                latest_transcripts AS (
                  SELECT DISTINCT ON (
                    transcripts.sample_track_id,
                    transcripts.model_id
                  )
                    sample_tracks.sample_id,
                    sample_tracks.side,
                    transcripts.model_id,
                    transcripts.error,
                    jsonb_array_length(transcripts.words) AS word_count
                  FROM full_recording_asr_transcripts AS transcripts
                  JOIN sample_tracks ON sample_tracks.id = transcripts.sample_track_id
                  JOIN selected_samples ON selected_samples.id = sample_tracks.sample_id
                  ORDER BY transcripts.sample_track_id,
                           transcripts.model_id,
                           transcripts.updated_at DESC,
                           transcripts.id DESC
                )
                SELECT
                  CASE
                    WHEN current_quality.sample_id IS NOT NULL THEN 'current_quality'
                    ELSE 'not_current'
                  END AS cohort,
                  latest_transcripts.model_id,
                  latest_transcripts.side,
                  COUNT(*) FILTER (
                    WHERE latest_transcripts.error IS NULL
                      AND latest_transcripts.word_count > 0
                  )::integer AS successful_transcript_count,
                  COUNT(*) FILTER (
                    WHERE latest_transcripts.error IS NOT NULL
                      OR latest_transcripts.word_count = 0
                  )::integer AS failed_transcript_count
                FROM latest_transcripts
                LEFT JOIN current_quality
                  ON current_quality.sample_id = latest_transcripts.sample_id
                GROUP BY cohort, latest_transcripts.model_id, latest_transcripts.side
                ORDER BY cohort, latest_transcripts.model_id, latest_transcripts.side
                """,
                (*parameters, expected_metric_version),
            ).fetchall()
        sample_count = int(overview["sample_count"])
        current_quality_sample_count = int(overview["current_quality_sample_count"])
        reviewed_count = int(overview["reviewed_current_quality_sample_count"])
        return DatasetCompletenessSummary(
            dataset_id=dataset_id,
            expected_metric_version=expected_metric_version,
            expected_annotation_version=expected_annotation_version,
            sample_count=sample_count,
            current_quality_sample_count=current_quality_sample_count,
            not_current_sample_count=int(overview["not_current_sample_count"]),
            duration_excluded_sample_count=int(overview["duration_excluded_sample_count"]),
            reviewed_current_quality_sample_count=reviewed_count,
            unreviewed_current_quality_sample_count=current_quality_sample_count - reviewed_count,
            quality_versions=tuple(QualityVersionCount.model_validate(row) for row in version_rows),
            full_asr_coverage=tuple(
                FullAsrCoverageCount.model_validate(row) for row in coverage_rows
            ),
        )

    def conversation_dataset_summary(
        self, sample_filter: SampleListFilter
    ) -> ConversationDatasetSummary:
        filter_sql = dashboard_filter_sql(sample_filter)
        query_parameters = (METRIC_VERSION, *filter_sql.parameters)
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT
                  COUNT(latest_quality.id)::integer AS analyzed_sample_count,
                  COUNT(latest_quality.id) FILTER (
                    WHERE latest_quality.status = 'invalid'
                  )::integer AS invalid_sample_count,
                  COALESCE(SUM(latest_quality.annotation_duration_seconds), 0.0)::double precision
                    AS analyzed_duration_seconds,
                  COALESCE(SUM(latest_quality.represented_duration_seconds), 0.0)::double precision
                    AS represented_duration_seconds,
                  COALESCE(SUM(latest_quality.speech_segment_count), 0)::integer
                    AS speech_segment_count,
                  COALESCE(SUM(latest_quality.interaction_count), 0)::integer
                    AS interaction_count,
                  COALESCE(SUM(latest_quality.turn_count), 0)::integer AS turn_count,
                  COALESCE(SUM(latest_quality.turn_taking_count), 0)::integer
                    AS turn_taking_count,
                  COALESCE(SUM(latest_quality.pause_count), 0)::integer AS pause_count,
                  COALESCE(SUM(latest_quality.backchannel_count), 0)::integer
                    AS backchannel_count,
                  COALESCE(SUM(latest_quality.interruption_count), 0)::integer
                    AS interruption_count,
                  COALESCE(SUM(latest_quality.usable_event_count), 0)::integer
                    AS usable_event_count,
                  COALESCE(SUM(latest_quality.estimated_speech_segment_count), 0)::integer
                    AS estimated_speech_segment_count,
                  COALESCE(SUM(latest_quality.estimated_interaction_count), 0)::integer
                    AS estimated_interaction_count,
                  COALESCE(SUM(latest_quality.estimated_turn_count), 0)::integer
                    AS estimated_turn_count,
                  COALESCE(SUM(latest_quality.estimated_turn_taking_count), 0)::integer
                    AS estimated_turn_taking_count,
                  COALESCE(SUM(latest_quality.estimated_pause_count), 0)::integer
                    AS estimated_pause_count,
                  COALESCE(SUM(latest_quality.estimated_backchannel_count), 0)::integer
                    AS estimated_backchannel_count,
                  COALESCE(SUM(latest_quality.estimated_interruption_count), 0)::integer
                    AS estimated_interruption_count,
                  COALESCE(SUM(latest_quality.estimated_usable_event_count), 0)::integer
                    AS estimated_usable_event_count
                FROM samples
                LEFT JOIN LATERAL (
                  SELECT *
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                    AND quality_results.metric_version = %s
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_quality ON true
                LEFT JOIN LATERAL (
                  SELECT *
                  FROM asr_runs
                  WHERE asr_runs.sample_id = samples.id
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_asr_run ON true
                LEFT JOIN asr_evaluations AS latest_asr_evaluation
                  ON latest_asr_evaluation.asr_run_id = latest_asr_run.id
                {language_summary_lateral_sql()}
                {filter_sql.where_clause}
                """,
                query_parameters,
            ).fetchone()
        assert row is not None
        return ConversationDatasetSummary(dataset_id=sample_filter.dataset_id, **row)

    def get_dashboard_sample(self, sample_id: UUID) -> DashboardSample:
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT
                  row_to_json(samples.*) AS sample_json,
                  COALESCE(track_rows.tracks_json, '[]'::json) AS tracks_json,
                  row_to_json(latest_quality.*) AS quality_json,
                  row_to_json(latest_asr_run.*) AS asr_run_json,
                  row_to_json(latest_asr_evaluation.*) AS asr_evaluation_json,
                  COALESCE(language_rows.assessments_json, '[]'::json)
                    AS language_assessments_json
                FROM samples
                LEFT JOIN LATERAL (
                  SELECT json_agg(sample_tracks.* ORDER BY speaker_index) AS tracks_json
                  FROM sample_tracks
                  WHERE sample_tracks.sample_id = samples.id
                ) AS track_rows ON true
                LEFT JOIN LATERAL (
                  SELECT *
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_quality ON true
                LEFT JOIN LATERAL (
                  SELECT *
                  FROM asr_runs
                  WHERE asr_runs.sample_id = samples.id
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_asr_run ON true
                LEFT JOIN asr_evaluations AS latest_asr_evaluation
                  ON latest_asr_evaluation.asr_run_id = latest_asr_run.id
                {language_assessments_lateral_sql()}
                WHERE samples.id = %s
                """,
                (sample_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Sample not found: {sample_id}")
        return dashboard_sample_from_row(row)

    def list_annotated_samples(
        self,
        dataset_id: UUID,
        metric_version: str,
        annotation_version: str,
        limit: int,
        minimum_quality: float | None,
    ) -> list[AnnotatedSampleRecord]:
        quality_filter, quality_parameters = minimum_quality_filter(minimum_quality)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                  samples.id AS sample_id,
                  samples.external_id,
                  COALESCE(samples.duration_seconds, 0.0)::double precision
                    AS represented_duration_seconds,
                  COALESCE(latest_quality.usable_event_count, 0)::integer
                    AS usable_event_count,
                  latest_quality.total_quality_score AS quality_score
                FROM samples
                JOIN LATERAL (
                  SELECT id, usable_event_count, total_quality_score, payload
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                    AND quality_results.metric_version = %s
                    AND quality_results.status = 'completed'
                    AND quality_results.payload
                      -> 'conversation_annotation'
                      ->> 'annotation_version' = %s
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_quality ON true
                WHERE samples.dataset_id = %s
                  AND samples.is_unusable = FALSE
                  {training_timeline_eligibility_sql()}
                  AND jsonb_typeof(
                    latest_quality.payload -> 'conversation_annotation'
                  ) = 'object'
                {quality_filter}
                ORDER BY samples.updated_at DESC, samples.external_id
                LIMIT %s
                """,
                (
                    metric_version,
                    annotation_version,
                    dataset_id,
                    *quality_parameters,
                    limit,
                ),
            ).fetchall()
        return [annotated_sample_record(row) for row in rows]

    def random_annotated_sample_id(
        self,
        dataset_id: UUID | None,
        metric_version: str,
        annotation_version: str,
        current_sample_id: UUID | None,
        minimum_quality: float | None,
    ) -> UUID:
        quality_filter, quality_parameters = minimum_quality_filter(minimum_quality)
        dataset_filter, dataset_parameters = optional_dataset_filter(dataset_id)
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT samples.id AS sample_id
                FROM samples
                JOIN LATERAL (
                  SELECT id, total_quality_score, payload
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                    AND quality_results.metric_version = %s
                    AND quality_results.status = 'completed'
                    AND quality_results.payload
                      -> 'conversation_annotation'
                      ->> 'annotation_version' = %s
                  ORDER BY created_at DESC
                  LIMIT 1
                ) AS latest_quality ON true
                WHERE samples.is_unusable = FALSE
                  {dataset_filter}
                  {training_timeline_eligibility_sql()}
                  AND jsonb_typeof(
                    latest_quality.payload -> 'conversation_annotation'
                  ) = 'object'
                {quality_filter}
                ORDER BY (samples.id = %s), random()
                LIMIT 1
                """,
                (
                    metric_version,
                    annotation_version,
                    *dataset_parameters,
                    *quality_parameters,
                    current_sample_id,
                ),
            ).fetchone()
        return annotated_sample_id(row)

    def interesting_annotated_sample_id(
        self,
        dataset_id: UUID | None,
        metric_version: str,
        annotation_version: str,
        current_sample_id: UUID | None,
        minimum_quality: float | None,
    ) -> UUID:
        quality_filter, quality_parameters = minimum_quality_filter(minimum_quality)
        dataset_filter, dataset_parameters = optional_dataset_filter(dataset_id)
        with self.connection() as connection:
            row = connection.execute(
                f"""
                WITH interesting_candidates AS (
                  SELECT
                    samples.id AS sample_id,
                    latest_quality.conversation_events_per_hour,
                    latest_quality.usable_event_count
                  FROM samples
                  JOIN LATERAL (
                    SELECT
                      id,
                      conversation_events_per_hour,
                      usable_event_count,
                      total_quality_score,
                      payload
                    FROM quality_results
                    WHERE quality_results.sample_id = samples.id
                      AND quality_results.metric_version = %s
                      AND quality_results.status = 'completed'
                      AND quality_results.payload
                        -> 'conversation_annotation'
                        ->> 'annotation_version' = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                  ) AS latest_quality ON true
                  WHERE samples.is_unusable = FALSE
                    {dataset_filter}
                    {training_timeline_eligibility_sql()}
                    AND jsonb_typeof(
                      latest_quality.payload -> 'conversation_annotation'
                    ) = 'object'
                  {quality_filter}
                  ORDER BY
                    latest_quality.conversation_events_per_hour DESC NULLS LAST,
                    latest_quality.usable_event_count DESC NULLS LAST
                  LIMIT %s
                )
                SELECT sample_id
                FROM interesting_candidates
                ORDER BY (sample_id = %s), random()
                LIMIT 1
                """,
                (
                    metric_version,
                    annotation_version,
                    *dataset_parameters,
                    *quality_parameters,
                    INTERESTING_SAMPLE_POOL_SIZE,
                    current_sample_id,
                ),
            ).fetchone()
        return annotated_sample_id(row)


def training_timeline_eligibility_sql() -> str:
    return f"""
      AND (
        NOT (
          EXISTS (
            SELECT 1
            FROM misalignment_lab_global_counterchecks AS global_countercheck
            WHERE global_countercheck.sample_id = samples.id
              AND global_countercheck.judgment = 'global_offset_confirmed'
          )
          OR EXISTS (
            SELECT 1
            FROM misalignment_lab_repair_reviews AS repair_review
            WHERE repair_review.sample_id = samples.id
              AND repair_review.repair_scope = 'after_change_point'
              AND repair_review.judgment = 'plausible'
              AND repair_review.transition_confirmed = TRUE
          )
        )
        OR EXISTS (
          SELECT 1
          FROM virtual_timeline_repair_plans AS repair_plan
          JOIN sample_tracks AS repaired_speaker1
            ON repaired_speaker1.sample_id = samples.id
           AND repaired_speaker1.side = 'speaker1'
          JOIN sample_tracks AS repaired_speaker2
            ON repaired_speaker2.sample_id = samples.id
           AND repaired_speaker2.side = 'speaker2'
          WHERE repair_plan.sample_id = samples.id
            AND repair_plan.plan_version = '{TIMELINE_REPAIR_PLAN_VERSION}'
            AND repair_plan.quality_result_id = latest_quality.id
            AND repair_plan.speaker1_audio_sha256 = repaired_speaker1.audio_sha256
            AND (
              (
                repair_plan.materialized_at IS NULL
                AND repair_plan.speaker2_audio_sha256 = repaired_speaker2.audio_sha256
              )
              OR
              (
                repair_plan.materialized_at IS NOT NULL
                AND repair_plan.materialized_speaker2_audio_sha256
                  = repaired_speaker2.audio_sha256
              )
            )
            AND repair_plan.derived_annotation IS NOT NULL
            AND repair_plan.conversation_regions IS NOT NULL
        )
      )
    """


def dashboard_filter_sql(sample_filter: SampleListFilter) -> DashboardFilterSql:
    filters: list[str] = []
    parameters: list[object] = []
    if sample_filter.dataset_id is not None:
        filters.append("samples.dataset_id = %s")
        parameters.append(sample_filter.dataset_id)
    if sample_filter.quality_min is not None:
        filters.append("latest_quality.total_quality_score >= %s")
        parameters.append(sample_filter.quality_min)
    if sample_filter.quality_max is not None:
        filters.append("latest_quality.total_quality_score <= %s")
        parameters.append(sample_filter.quality_max)
    if sample_filter.duration_min is not None:
        filters.append("samples.duration_seconds >= %s")
        parameters.append(sample_filter.duration_min)
    if sample_filter.duration_max is not None:
        filters.append("samples.duration_seconds <= %s")
        parameters.append(sample_filter.duration_max)
    if sample_filter.flag is not None:
        filters.append("%s = ANY(samples.quality_flags)")
        parameters.append(sample_filter.flag)
    if sample_filter.speech_ratio_min is not None:
        filters.append("latest_quality.speech_ratio >= %s")
        parameters.append(sample_filter.speech_ratio_min)
    if sample_filter.speech_ratio_max is not None:
        filters.append("latest_quality.speech_ratio <= %s")
        parameters.append(sample_filter.speech_ratio_max)
    if sample_filter.overlap_ratio_min is not None:
        filters.append("latest_quality.overlap_ratio >= %s")
        parameters.append(sample_filter.overlap_ratio_min)
    if sample_filter.overlap_ratio_max is not None:
        filters.append("latest_quality.overlap_ratio <= %s")
        parameters.append(sample_filter.overlap_ratio_max)
    if sample_filter.silence_ratio_min is not None:
        filters.append("latest_quality.silence_ratio >= %s")
        parameters.append(sample_filter.silence_ratio_min)
    if sample_filter.silence_ratio_max is not None:
        filters.append("latest_quality.silence_ratio <= %s")
        parameters.append(sample_filter.silence_ratio_max)
    if sample_filter.asr_model is not None:
        filters.append("latest_asr_run.model_name = %s")
        parameters.append(sample_filter.asr_model)
    if sample_filter.wer_min is not None:
        filters.append("latest_asr_evaluation.wer >= %s")
        parameters.append(sample_filter.wer_min)
    if sample_filter.wer_max is not None:
        filters.append("latest_asr_evaluation.wer <= %s")
        parameters.append(sample_filter.wer_max)
    if sample_filter.timestamp_p90_max is not None:
        filters.append(
            """
            LEAST(
              COALESCE(latest_asr_evaluation.start_p90_absolute_error, 'Infinity'::float),
              COALESCE(latest_asr_evaluation.end_p90_absolute_error, 'Infinity'::float)
            ) <= %s
            """
        )
        parameters.append(sample_filter.timestamp_p90_max)
    if sample_filter.language_status is not None:
        filters.append("language_summary.language_status = %s")
        parameters.append(sample_filter.language_status.value)
    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    return DashboardFilterSql(where_clause=where_clause, parameters=tuple(parameters))


def minimum_quality_filter(
    minimum_quality: float | None,
) -> tuple[str, tuple[object, ...]]:
    if minimum_quality is None:
        return "", ()
    return "AND latest_quality.total_quality_score > %s", (minimum_quality,)


def optional_dataset_filter(dataset_id: UUID | None) -> tuple[str, tuple[UUID, ...]]:
    if dataset_id is None:
        return "", ()
    return "AND samples.dataset_id = %s", (dataset_id,)


def annotated_sample_id(row: dict[str, object] | None) -> UUID:
    if row is None:
        raise ValueError("No annotated samples match the selected quality filter.")
    sample_id = row["sample_id"]
    assert isinstance(sample_id, UUID)
    return sample_id


def annotated_sample_record(row: dict[str, object]) -> AnnotatedSampleRecord:
    sample_id = row["sample_id"]
    external_id = row["external_id"]
    represented_duration_seconds = row["represented_duration_seconds"]
    usable_event_count = row["usable_event_count"]
    quality_score = row["quality_score"]
    assert isinstance(sample_id, UUID)
    assert isinstance(external_id, str)
    assert isinstance(represented_duration_seconds, float)
    assert isinstance(usable_event_count, int)
    assert quality_score is None or isinstance(quality_score, float)
    return AnnotatedSampleRecord(
        sample_id=sample_id,
        external_id=external_id,
        represented_duration_seconds=represented_duration_seconds,
        usable_event_count=usable_event_count,
        quality_score=quality_score,
    )


def dataset_record(row: dict[str, object]) -> DatasetRecord:
    return DatasetRecord.model_validate(row)


def sample_record(row: dict[str, object]) -> SampleRecord:
    row = normalize_array_row(row, "quality_flags")
    return SampleRecord.model_validate(row)


def sample_track_record(row: dict[str, object]) -> SampleTrackRecord:
    return SampleTrackRecord.model_validate(row)


def audio_metadata_record(row: dict[str, object]) -> AudioMetadataRecord:
    row = normalize_json_row(row, "payload")
    return AudioMetadataRecord.model_validate(row)


def quality_result_record(row: dict[str, object]) -> QualityResultRecord:
    row = normalize_array_row(normalize_json_row(row, "payload"), "flags")
    return QualityResultRecord.model_validate(row)


def asr_run_record(row: dict[str, object]) -> AsrRunRecord:
    row = normalize_json_row(row, "payload")
    return AsrRunRecord.model_validate(row)


def asr_word_record(row: dict[str, object]) -> AsrWordRecord:
    return AsrWordRecord.model_validate(row)


def asr_evaluation_record(row: dict[str, object]) -> AsrEvaluationRecord:
    row = normalize_json_row(row, "payload")
    return AsrEvaluationRecord.model_validate(row)


def ingestion_job_record(row: dict[str, object]) -> IngestionJobRecord:
    return IngestionJobRecord.model_validate(row)


def dashboard_sample_from_row(row: dict[str, object]) -> DashboardSample:
    sample_json = stable_json_value(row["sample_json"])
    tracks_json = stable_json_value(row["tracks_json"])
    quality_json = stable_json_value(row["quality_json"])
    asr_run_json = stable_json_value(row["asr_run_json"])
    asr_evaluation_json = stable_json_value(row["asr_evaluation_json"])
    language_assessments_json = stable_json_value(row["language_assessments_json"])
    return DashboardSample(
        sample=sample_record(dict(sample_json)),
        tracks=tuple(sample_track_record(dict(track_json)) for track_json in tracks_json),
        latest_quality=quality_result_record(dict(quality_json))
        if quality_json is not None
        else None,
        latest_asr_run=asr_run_record(dict(asr_run_json)) if asr_run_json is not None else None,
        latest_asr_evaluation=asr_evaluation_record(dict(asr_evaluation_json))
        if asr_evaluation_json is not None
        else None,
        language_assessments=tuple(
            TrackLanguageAssessment.model_validate(assessment)
            for assessment in language_assessments_json
        ),
    )


def dashboard_sample_summary_from_row(row: dict[str, object]) -> DashboardSampleSummary:
    sample_json = stable_json_value(row["sample_json"])
    quality_id = row["quality_id"]
    latest_quality = (
        QualityResultSummaryRecord(
            id=quality_id if isinstance(quality_id, UUID) else UUID(str(quality_id)),
            sample_id=(
                row["quality_sample_id"]
                if isinstance(row["quality_sample_id"], UUID)
                else UUID(str(row["quality_sample_id"]))
            ),
            metric_version=str(row["metric_version"]),
            annotation_version=(
                str(row["annotation_version"]) if row["annotation_version"] is not None else None
            ),
            status=str(row["quality_status"]),
            total_quality_score=optional_float(row["total_quality_score"]),
            speech_ratio=optional_float(row["speech_ratio"]),
            overlap_ratio=optional_float(row["overlap_ratio"]),
            has_parakeet_transcript_pair=bool(row["has_parakeet_transcript_pair"]),
            has_canary_transcript_pair=bool(row["has_canary_transcript_pair"]),
            created_at=datetime_from_value(row["quality_created_at"]),
        )
        if quality_id is not None
        else None
    )
    return DashboardSampleSummary(
        sample=sample_record(dict(sample_json)),
        latest_quality=latest_quality,
        language_status=(
            SampleLanguageStatus(str(row["language_status"]))
            if row["language_status"] is not None
            else None
        ),
    )


def language_assessments_lateral_sql() -> str:
    return """
        LEFT JOIN LATERAL (
          SELECT json_agg(
            current_language.assessment_json
            ORDER BY current_language.speaker_index
          ) AS assessments_json
          FROM (
            SELECT DISTINCT ON (sample_tracks.id)
              to_jsonb(track_language_assessments.*) AS assessment_json,
              sample_tracks.speaker_index
            FROM sample_tracks
            JOIN track_language_assessments
              ON track_language_assessments.sample_track_id = sample_tracks.id
             AND track_language_assessments.source_audio_sha256
                 = sample_tracks.audio_sha256
            WHERE sample_tracks.sample_id = samples.id
            ORDER BY
              sample_tracks.id,
              track_language_assessments.updated_at DESC,
              track_language_assessments.id DESC
          ) AS current_language
        ) AS language_rows ON true
    """


def language_summary_lateral_sql() -> str:
    return """
        LEFT JOIN LATERAL (
          SELECT CASE
            WHEN COUNT(current_language.status) = 0 THEN NULL
            WHEN BOOL_OR(current_language.status = 'non_english') THEN 'non_english'
            WHEN COUNT(current_language.status) < (
              SELECT COUNT(*)
              FROM sample_tracks
              WHERE sample_tracks.sample_id = samples.id
            ) OR BOOL_OR(current_language.status IN ('inconclusive', 'failed'))
              THEN 'inconclusive'
            ELSE 'english'
          END AS language_status
          FROM (
            SELECT DISTINCT ON (sample_tracks.id)
              sample_tracks.id AS sample_track_id,
              track_language_assessments.status
            FROM sample_tracks
            JOIN track_language_assessments
              ON track_language_assessments.sample_track_id = sample_tracks.id
             AND track_language_assessments.source_audio_sha256
                 = sample_tracks.audio_sha256
            WHERE sample_tracks.sample_id = samples.id
            ORDER BY
              sample_tracks.id,
              track_language_assessments.updated_at DESC,
              track_language_assessments.id DESC
          ) AS current_language
        ) AS language_summary ON true
    """


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    assert isinstance(value, int | float)
    return float(value)


def datetime_from_value(value: object) -> datetime:
    assert isinstance(value, datetime)
    return value


def normalize_json_row(row: dict[str, object], key: str) -> dict[str, object]:
    value = row[key]
    if isinstance(value, str):
        normalized = dict(row)
        normalized[key] = json.loads(value)
        return normalized
    return row


def normalize_array_row(row: dict[str, object], key: str) -> dict[str, object]:
    value = row[key]
    if isinstance(value, list):
        normalized = dict(row)
        normalized[key] = tuple(str(item) for item in value)
        return normalized
    return row


def stable_json_value(value: object) -> object:
    if value is None:
        return None
    return json.loads(json.dumps(value, default=json_default))


def json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")
