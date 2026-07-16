from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.asr.full_recording_models import (
    FullRecordingAsrBatchItem,
    FullRecordingAsrBatchRecord,
    FullRecordingAsrBatchRequest,
    FullRecordingAsrBatchStatus,
    FullRecordingAsrItemStatus,
    FullRecordingAsrSampleScope,
    FullRecordingAsrTranscriptRecord,
)
from app.local.db.models import TrackSide
from app.shared.asr import AsrModelId, AsrRuntimeStats, AsrTranscriptResult, TimestampedWord

RECENT_ERROR_LIMIT = 10


class FullRecordingAsrRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def create_batch(self, request: FullRecordingAsrBatchRequest) -> FullRecordingAsrBatchRecord:
        models = unique_models(request.models)
        with self.connection() as connection:
            dataset_row = connection.execute(
                "SELECT id FROM datasets WHERE id = %s", (request.dataset_id,)
            ).fetchone()
            if dataset_row is None:
                raise ValueError(f"Dataset not found: {request.dataset_id}")
            batch_row = connection.execute(
                """
                INSERT INTO full_recording_asr_batches (
                  idempotency_key, dataset_id, sample_scope, model_ids, status
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING *
                """,
                (
                    request.idempotency_key,
                    request.dataset_id,
                    request.sample_scope.value,
                    [model.value for model in models],
                    FullRecordingAsrBatchStatus.QUEUED.value,
                ),
            ).fetchone()
            if batch_row is None:
                existing_row = connection.execute(
                    "SELECT * FROM full_recording_asr_batches WHERE idempotency_key = %s",
                    (request.idempotency_key,),
                ).fetchone()
                assert existing_row is not None
                self._validate_idempotent_request(row=existing_row, request=request, models=models)
                batch_id = uuid_from_row(existing_row["id"])
            else:
                assert batch_row is not None
                batch_id = uuid_from_row(batch_row["id"])
                track_filter = self._track_filter(request.sample_scope)
                connection.execute(
                    f"""
                    INSERT INTO full_recording_asr_batch_items (batch_id, sample_track_id, status)
                    SELECT %s, sample_tracks.id, %s
                    FROM sample_tracks
                    JOIN samples ON samples.id = sample_tracks.sample_id
                    WHERE samples.dataset_id = %s
                    {track_filter}
                    ORDER BY samples.external_id, sample_tracks.speaker_index
                    """,
                    (
                        batch_id,
                        FullRecordingAsrItemStatus.PENDING.value,
                        request.dataset_id,
                    ),
                )
                item_count = connection.execute(
                    """
                    SELECT COUNT(*)::integer AS item_count
                    FROM full_recording_asr_batch_items
                    WHERE batch_id = %s
                    """,
                    (batch_id,),
                ).fetchone()
                assert item_count is not None
                if int_from_row(item_count["item_count"]) == 0:
                    raise ValueError(
                        "No sample tracks match the requested full-recording ASR scope."
                    )
        return self.get_batch(batch_id)

    def get_batch(self, batch_id: UUID) -> FullRecordingAsrBatchRecord:
        with self.connection() as connection:
            row = connection.execute(batch_status_query(), (batch_id,)).fetchone()
        if row is None:
            raise ValueError(f"Full-recording ASR batch not found: {batch_id}")
        return batch_record(row)

    def recover_running_items(self, batch_id: UUID) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE full_recording_asr_batch_items
                SET status = %s, error = NULL, updated_at = now(), started_at = NULL
                WHERE batch_id = %s AND status = %s
                """,
                (
                    FullRecordingAsrItemStatus.PENDING.value,
                    batch_id,
                    FullRecordingAsrItemStatus.RUNNING.value,
                ),
            )
        return cursor.rowcount

    def retry_failed_items(self, batch_id: UUID) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE full_recording_asr_batch_items
                SET status = %s, error = NULL, updated_at = now(),
                    started_at = NULL, finished_at = NULL
                WHERE batch_id = %s AND status = %s
                """,
                (
                    FullRecordingAsrItemStatus.PENDING.value,
                    batch_id,
                    FullRecordingAsrItemStatus.FAILED.value,
                ),
            )
            if cursor.rowcount > 0:
                connection.execute(
                    """
                    UPDATE full_recording_asr_batches
                    SET status = %s, updated_at = now(), finished_at = NULL
                    WHERE id = %s
                    """,
                    (FullRecordingAsrBatchStatus.QUEUED.value, batch_id),
                )
        return cursor.rowcount

    def claim_next_item(self, batch_id: UUID) -> FullRecordingAsrBatchItem | None:
        with self.connection() as connection:
            batch_row = connection.execute(
                "SELECT model_ids FROM full_recording_asr_batches WHERE id = %s FOR UPDATE",
                (batch_id,),
            ).fetchone()
            if batch_row is None:
                raise ValueError(f"Full-recording ASR batch not found: {batch_id}")
            models = model_ids_from_row(batch_row["model_ids"])
            row = connection.execute(
                """
                WITH claimed AS (
                  SELECT full_recording_asr_batch_items.id
                  FROM full_recording_asr_batch_items
                  WHERE batch_id = %s AND status = %s
                  ORDER BY created_at, id
                  FOR UPDATE SKIP LOCKED
                  LIMIT 1
                )
                UPDATE full_recording_asr_batch_items AS items
                SET status = %s,
                    attempt_count = attempt_count + 1,
                    error = NULL,
                    started_at = now(),
                    finished_at = NULL,
                    updated_at = now()
                FROM claimed, sample_tracks, samples
                WHERE items.id = claimed.id
                  AND sample_tracks.id = items.sample_track_id
                  AND samples.id = sample_tracks.sample_id
                RETURNING items.*, samples.external_id, sample_tracks.side,
                          sample_tracks.access_uri
                """,
                (
                    batch_id,
                    FullRecordingAsrItemStatus.PENDING.value,
                    FullRecordingAsrItemStatus.RUNNING.value,
                ),
            ).fetchone()
            if row is not None:
                connection.execute(
                    """
                    UPDATE full_recording_asr_batches
                    SET status = %s, started_at = COALESCE(started_at, now()), updated_at = now()
                    WHERE id = %s
                    """,
                    (FullRecordingAsrBatchStatus.RUNNING.value, batch_id),
                )
        if row is None:
            return None
        return FullRecordingAsrBatchItem(
            id=uuid_from_row(row["id"]),
            batch_id=uuid_from_row(row["batch_id"]),
            sample_track_id=uuid_from_row(row["sample_track_id"]),
            sample_external_id=string_from_row(row["external_id"]),
            side=TrackSide(string_from_row(row["side"])),
            access_uri=string_from_row(row["access_uri"]),
            models=models,
            attempt_count=int_from_row(row["attempt_count"]),
        )

    def complete_item(self, item_id: UUID) -> None:
        self._finish_item(
            item_id=item_id,
            status=FullRecordingAsrItemStatus.COMPLETED,
            error=None,
        )

    def fail_item(self, item_id: UUID, error: str) -> None:
        self._finish_item(
            item_id=item_id,
            status=FullRecordingAsrItemStatus.FAILED,
            error=error,
        )

    def finalize_batch(self, batch_id: UUID) -> FullRecordingAsrBatchRecord:
        with self.connection() as connection:
            counts = connection.execute(
                """
                SELECT
                  COUNT(*) FILTER (WHERE status IN ('pending', 'running'))::integer AS unfinished,
                  COUNT(*) FILTER (WHERE status = 'failed')::integer AS failed
                FROM full_recording_asr_batch_items
                WHERE batch_id = %s
                """,
                (batch_id,),
            ).fetchone()
            assert counts is not None
            if int_from_row(counts["unfinished"]) > 0:
                return self.get_batch(batch_id)
            status = (
                FullRecordingAsrBatchStatus.COMPLETED_WITH_ERRORS
                if int_from_row(counts["failed"]) > 0
                else FullRecordingAsrBatchStatus.COMPLETED
            )
            connection.execute(
                """
                UPDATE full_recording_asr_batches
                SET status = %s, updated_at = now(), finished_at = now()
                WHERE id = %s
                """,
                (status.value, batch_id),
            )
        return self.get_batch(batch_id)

    def get_cached_transcripts(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        model_ids: Sequence[AsrModelId],
    ) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
        if not model_ids:
            return ()
        with self.connection() as connection:
            rows = connection.execute(
                transcript_select_query(
                    """
                    WHERE transcripts.sample_track_id = %s
                      AND transcripts.source_audio_sha256 = %s
                      AND transcripts.model_id = ANY(%s)
                    """
                ),
                (
                    sample_track_id,
                    source_audio_sha256,
                    [model_id.value for model_id in model_ids],
                ),
            ).fetchall()
        return tuple(transcript_record(row) for row in rows)

    def upsert_transcript(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        prepared_audio_sha256: str,
        audio_filename: str,
        source_duration_seconds: float,
        prepared_duration_seconds: float,
        transcript: AsrTranscriptResult,
    ) -> FullRecordingAsrTranscriptRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO full_recording_asr_transcripts (
                  sample_track_id, source_audio_sha256, prepared_audio_sha256,
                  audio_filename, model_id, transcript_text, words,
                  source_duration_seconds, prepared_duration_seconds,
                  processing_time_seconds, runtime, error, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (sample_track_id, source_audio_sha256, model_id) DO UPDATE
                SET prepared_audio_sha256 = EXCLUDED.prepared_audio_sha256,
                    audio_filename = EXCLUDED.audio_filename,
                    transcript_text = EXCLUDED.transcript_text,
                    words = EXCLUDED.words,
                    source_duration_seconds = EXCLUDED.source_duration_seconds,
                    prepared_duration_seconds = EXCLUDED.prepared_duration_seconds,
                    processing_time_seconds = EXCLUDED.processing_time_seconds,
                    runtime = EXCLUDED.runtime,
                    error = EXCLUDED.error,
                    updated_at = now()
                RETURNING id
                """,
                (
                    sample_track_id,
                    source_audio_sha256,
                    prepared_audio_sha256,
                    audio_filename,
                    transcript.model_id.value,
                    transcript.text,
                    Jsonb([word.model_dump(mode="json") for word in transcript.words]),
                    source_duration_seconds,
                    prepared_duration_seconds,
                    transcript.processing_time_seconds,
                    Jsonb(
                        transcript.runtime.model_dump(mode="json")
                        if transcript.runtime is not None
                        else {}
                    ),
                    transcript.error,
                ),
            ).fetchone()
            assert row is not None
            transcript_id = uuid_from_row(row["id"])
            selected = connection.execute(
                transcript_select_query("WHERE transcripts.id = %s"),
                (transcript_id,),
            ).fetchone()
        assert selected is not None
        return transcript_record(selected)

    def list_latest_transcripts(
        self, model_ids: Sequence[AsrModelId]
    ) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
        if not model_ids:
            return ()
        with self.connection() as connection:
            rows = connection.execute(
                transcript_select_query(
                    """
                    WHERE transcripts.model_id = ANY(%s)
                      AND NOT EXISTS (
                        SELECT 1
                        FROM full_recording_asr_transcripts AS newer
                        WHERE newer.sample_track_id = transcripts.sample_track_id
                          AND newer.model_id = transcripts.model_id
                          AND (newer.updated_at, newer.id) >
                              (transcripts.updated_at, transcripts.id)
                      )
                    """
                ),
                ([model_id.value for model_id in model_ids],),
            ).fetchall()
        return tuple(transcript_record(row) for row in rows)

    def _finish_item(
        self,
        item_id: UUID,
        status: FullRecordingAsrItemStatus,
        error: str | None,
    ) -> None:
        with self.connection() as connection:
            row = connection.execute(
                """
                UPDATE full_recording_asr_batch_items
                SET status = %s, error = %s, updated_at = now(), finished_at = now()
                WHERE id = %s AND status = %s
                RETURNING batch_id
                """,
                (status.value, error, item_id, FullRecordingAsrItemStatus.RUNNING.value),
            ).fetchone()
        if row is None:
            raise ValueError(f"Running full-recording ASR item not found: {item_id}")

    @staticmethod
    def _track_filter(scope: FullRecordingAsrSampleScope) -> str:
        match scope:
            case FullRecordingAsrSampleScope.QUALITY_ANALYZED:
                return (
                    "AND EXISTS (SELECT 1 FROM quality_results "
                    "WHERE quality_results.sample_id = samples.id)"
                )
            case FullRecordingAsrSampleScope.ALL_DATASET_SAMPLES:
                return ""

    @staticmethod
    def _validate_idempotent_request(
        row: dict[str, object],
        request: FullRecordingAsrBatchRequest,
        models: tuple[AsrModelId, ...],
    ) -> None:
        stored_dataset_id = uuid_from_row(row["dataset_id"])
        stored_scope = FullRecordingAsrSampleScope(string_from_row(row["sample_scope"]))
        stored_models = model_ids_from_row(row["model_ids"])
        if (
            stored_dataset_id != request.dataset_id
            or stored_scope != request.sample_scope
            or stored_models != models
        ):
            raise ValueError(
                "Full-recording ASR idempotency key already exists with a different request."
            )


def unique_models(models: Sequence[AsrModelId]) -> tuple[AsrModelId, ...]:
    return tuple(sorted(set(models), key=lambda model: model.value))


def batch_status_query() -> str:
    return f"""
        SELECT batches.*,
          COUNT(items.id)::integer AS total_track_count,
          COUNT(items.id) FILTER (WHERE items.status = 'pending')::integer
            AS pending_track_count,
          COUNT(items.id) FILTER (WHERE items.status = 'running')::integer
            AS running_track_count,
          COUNT(items.id) FILTER (WHERE items.status = 'completed')::integer
            AS completed_track_count,
          COUNT(items.id) FILTER (WHERE items.status = 'failed')::integer
            AS failed_track_count,
          COALESCE((
            SELECT array_agg(recent.error ORDER BY recent.updated_at DESC)
            FROM (
              SELECT error, updated_at
              FROM full_recording_asr_batch_items
              WHERE batch_id = batches.id AND error IS NOT NULL
              ORDER BY updated_at DESC
              LIMIT {RECENT_ERROR_LIMIT}
            ) AS recent
          ), ARRAY[]::text[]) AS recent_errors
        FROM full_recording_asr_batches AS batches
        LEFT JOIN full_recording_asr_batch_items AS items ON items.batch_id = batches.id
        WHERE batches.id = %s
        GROUP BY batches.id
    """


def transcript_select_query(where_clause: str) -> str:
    return f"""
        SELECT transcripts.*, sample_tracks.sample_id, sample_tracks.side,
               samples.external_id
        FROM full_recording_asr_transcripts AS transcripts
        JOIN sample_tracks ON sample_tracks.id = transcripts.sample_track_id
        JOIN samples ON samples.id = sample_tracks.sample_id
        {where_clause}
        ORDER BY samples.external_id, sample_tracks.speaker_index, transcripts.model_id
    """


def batch_record(row: dict[str, object]) -> FullRecordingAsrBatchRecord:
    recent_errors = row["recent_errors"]
    assert isinstance(recent_errors, list)
    return FullRecordingAsrBatchRecord(
        id=uuid_from_row(row["id"]),
        idempotency_key=string_from_row(row["idempotency_key"]),
        dataset_id=uuid_from_row(row["dataset_id"]),
        sample_scope=FullRecordingAsrSampleScope(string_from_row(row["sample_scope"])),
        models=model_ids_from_row(row["model_ids"]),
        status=FullRecordingAsrBatchStatus(string_from_row(row["status"])),
        total_track_count=int_from_row(row["total_track_count"]),
        pending_track_count=int_from_row(row["pending_track_count"]),
        running_track_count=int_from_row(row["running_track_count"]),
        completed_track_count=int_from_row(row["completed_track_count"]),
        failed_track_count=int_from_row(row["failed_track_count"]),
        recent_errors=tuple(str(error) for error in recent_errors),
        created_at=datetime_from_row(row["created_at"]),
        updated_at=datetime_from_row(row["updated_at"]),
        started_at=optional_datetime_from_row(row["started_at"]),
        finished_at=optional_datetime_from_row(row["finished_at"]),
    )


def transcript_record(row: dict[str, object]) -> FullRecordingAsrTranscriptRecord:
    words_value = json_value(row["words"])
    runtime_value = json_value(row["runtime"])
    if not isinstance(words_value, list):
        raise ValueError("Full-recording ASR words must be a JSON array.")
    if not isinstance(runtime_value, dict):
        raise ValueError("Full-recording ASR runtime must be a JSON object.")
    return FullRecordingAsrTranscriptRecord(
        id=uuid_from_row(row["id"]),
        sample_track_id=uuid_from_row(row["sample_track_id"]),
        sample_id=uuid_from_row(row["sample_id"]),
        sample_external_id=string_from_row(row["external_id"]),
        side=TrackSide(string_from_row(row["side"])),
        source_audio_sha256=string_from_row(row["source_audio_sha256"]),
        prepared_audio_sha256=string_from_row(row["prepared_audio_sha256"]),
        audio_filename=string_from_row(row["audio_filename"]),
        model_id=AsrModelId(string_from_row(row["model_id"])),
        transcript_text=string_from_row(row["transcript_text"]),
        words=tuple(TimestampedWord.model_validate(word) for word in words_value),
        source_duration_seconds=float_from_row(row["source_duration_seconds"]),
        prepared_duration_seconds=float_from_row(row["prepared_duration_seconds"]),
        processing_time_seconds=optional_float_from_row(row["processing_time_seconds"]),
        runtime=AsrRuntimeStats.model_validate(runtime_value) if runtime_value else None,
        error=optional_string_from_row(row["error"]),
        created_at=datetime_from_row(row["created_at"]),
        updated_at=datetime_from_row(row["updated_at"]),
    )


def model_ids_from_row(value: object) -> tuple[AsrModelId, ...]:
    assert isinstance(value, list)
    return tuple(AsrModelId(str(model_id)) for model_id in value)


def json_value(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


def uuid_from_row(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def string_from_row(value: object) -> str:
    assert isinstance(value, str)
    return value


def int_from_row(value: object) -> int:
    assert isinstance(value, int)
    return value


def float_from_row(value: object) -> float:
    assert isinstance(value, int | float)
    return float(value)


def optional_float_from_row(value: object) -> float | None:
    return float_from_row(value) if value is not None else None


def optional_string_from_row(value: object) -> str | None:
    return string_from_row(value) if value is not None else None


def datetime_from_row(value: object) -> datetime:
    assert isinstance(value, datetime)
    return value


def optional_datetime_from_row(value: object) -> datetime | None:
    return datetime_from_row(value) if value is not None else None
