from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.db.models import (
    AsrEvaluationRecord,
    AsrRunRecord,
    AsrRunStatus,
    AsrWordRecord,
    AudioMetadataRecord,
    DashboardSample,
    DatasetCreate,
    DatasetRecord,
    IngestionJobRecord,
    JobStatus,
    QualityResultRecord,
    SampleListFilter,
    SampleRecord,
    SampleTrackRecord,
    TrackSide,
)


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
    calibrated_quality_score: float | None
    interaction_density_score: float | None
    timing_reliability_score: float | None
    audio_quality_score: float | None
    speech_ratio: float | None
    silence_ratio: float | None
    overlap_ratio: float | None
    duration_mismatch_seconds: float | None
    track_correlation: float | None
    energy_envelope_correlation: float | None
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
                      WHEN %s IN ('completed', 'failed', 'canceled') THEN now()
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

    def upsert_sample_track(self, sample_id: UUID, track: SampleTrackInput) -> SampleTrackRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO sample_tracks (
                  sample_id, side, speaker_index, storage_uri, access_uri,
                  duration_seconds, sample_rate, channels, sample_count, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (sample_id, side) DO UPDATE
                SET speaker_index = EXCLUDED.speaker_index,
                    storage_uri = EXCLUDED.storage_uri,
                    access_uri = EXCLUDED.access_uri,
                    duration_seconds = EXCLUDED.duration_seconds,
                    sample_rate = EXCLUDED.sample_rate,
                    channels = EXCLUDED.channels,
                    sample_count = EXCLUDED.sample_count,
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
                  calibrated_quality_score, interaction_density_score, timing_reliability_score,
                  audio_quality_score, speech_ratio, silence_ratio, overlap_ratio,
                  duration_mismatch_seconds, track_correlation, energy_envelope_correlation,
                  flags, payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    quality_result.sample_id,
                    quality_result.metric_version,
                    quality_result.status,
                    quality_result.total_quality_score,
                    quality_result.raw_quality_score,
                    quality_result.calibrated_quality_score,
                    quality_result.interaction_density_score,
                    quality_result.timing_reliability_score,
                    quality_result.audio_quality_score,
                    quality_result.speech_ratio,
                    quality_result.silence_ratio,
                    quality_result.overlap_ratio,
                    quality_result.duration_mismatch_seconds,
                    quality_result.track_correlation,
                    quality_result.energy_envelope_correlation,
                    list(quality_result.flags),
                    Jsonb(quality_result.payload),
                ),
            ).fetchone()
            connection.execute(
                """
                UPDATE samples
                SET duration_seconds = COALESCE(%s, duration_seconds),
                    quality_score = %s,
                    quality_flags = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    quality_result.payload.get("duration_seconds"),
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
        where_clause = "WHERE " + " AND ".join(filters) if filters else ""
        parameters.extend([sample_filter.limit, sample_filter.offset])
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                  row_to_json(samples.*) AS sample_json,
                  COALESCE(track_rows.tracks_json, '[]'::json) AS tracks_json,
                  row_to_json(latest_quality.*) AS quality_json,
                  row_to_json(latest_asr_run.*) AS asr_run_json,
                  row_to_json(latest_asr_evaluation.*) AS asr_evaluation_json
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
                {where_clause}
                ORDER BY samples.updated_at DESC, samples.external_id
                LIMIT %s OFFSET %s
                """,
                parameters,
            ).fetchall()
        return [dashboard_sample_from_row(row) for row in rows]

    def get_dashboard_sample(self, sample_id: UUID) -> DashboardSample:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                  row_to_json(samples.*) AS sample_json,
                  COALESCE(track_rows.tracks_json, '[]'::json) AS tracks_json,
                  row_to_json(latest_quality.*) AS quality_json,
                  row_to_json(latest_asr_run.*) AS asr_run_json,
                  row_to_json(latest_asr_evaluation.*) AS asr_evaluation_json
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
                WHERE samples.id = %s
                """,
                (sample_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Sample not found: {sample_id}")
        return dashboard_sample_from_row(row)


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
    )


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
