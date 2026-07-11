from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.asr.schemas import AsrModelId, AsrRuntimeStats, AsrTranscriptResult, TranscriptSegment


@dataclass(frozen=True)
class CachedAsrTranscriptRecord:
    id: UUID
    audio_sha256: str
    audio_filename: str
    model_id: AsrModelId
    transcript_text: str
    segments: tuple[TranscriptSegment, ...]
    processing_time_seconds: float | None
    runtime: AsrRuntimeStats | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class AsrTranscriptRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def get_cached_asr_transcripts(
        self,
        audio_sha256: str,
        model_ids: Sequence[AsrModelId],
    ) -> tuple[AsrTranscriptResult, ...]:
        if not model_ids:
            return ()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM cached_asr_transcripts
                WHERE audio_sha256 = %s
                  AND model_id = ANY(%s)
                ORDER BY updated_at DESC
                """,
                (audio_sha256, [model_id.value for model_id in model_ids]),
            ).fetchall()
        records = tuple(cached_asr_transcript_record(row) for row in rows)
        return tuple(cached_asr_transcript_result(record) for record in records)

    def upsert_cached_asr_transcript(
        self,
        audio_sha256: str,
        audio_filename: str,
        transcript: AsrTranscriptResult,
    ) -> AsrTranscriptResult:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO cached_asr_transcripts (
                  audio_sha256, audio_filename, model_id, transcript_text,
                  segments, processing_time_seconds, runtime, error, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (audio_sha256, model_id) DO UPDATE
                SET audio_filename = EXCLUDED.audio_filename,
                    transcript_text = EXCLUDED.transcript_text,
                    segments = EXCLUDED.segments,
                    processing_time_seconds = EXCLUDED.processing_time_seconds,
                    runtime = EXCLUDED.runtime,
                    error = EXCLUDED.error,
                    updated_at = now()
                RETURNING *
                """,
                (
                    audio_sha256,
                    audio_filename,
                    transcript.model_id.value,
                    transcript.text,
                    Jsonb([segment.model_dump(mode="json") for segment in transcript.segments]),
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
        return cached_asr_transcript_result(cached_asr_transcript_record(row))


def cached_asr_transcript_record(row: dict[str, object]) -> CachedAsrTranscriptRecord:
    segments_value = row["segments"]
    segments = json.loads(segments_value) if isinstance(segments_value, str) else segments_value
    if not isinstance(segments, list):
        raise ValueError("Cached ASR transcript segments must be a JSON array.")
    runtime_value = row["runtime"]
    runtime = json.loads(runtime_value) if isinstance(runtime_value, str) else runtime_value
    if not isinstance(runtime, dict):
        raise ValueError("Cached ASR transcript runtime must be a JSON object.")
    return CachedAsrTranscriptRecord(
        id=UUID(str(row["id"])),
        audio_sha256=str(row["audio_sha256"]),
        audio_filename=str(row["audio_filename"]),
        model_id=AsrModelId(str(row["model_id"])),
        transcript_text=str(row["transcript_text"]),
        segments=tuple(TranscriptSegment.model_validate(segment) for segment in segments),
        processing_time_seconds=optional_float(row["processing_time_seconds"]),
        runtime=AsrRuntimeStats.model_validate(runtime) if runtime else None,
        error=optional_string(row["error"]),
        created_at=datetime_from_row(row["created_at"]),
        updated_at=datetime_from_row(row["updated_at"]),
    )


def cached_asr_transcript_result(record: CachedAsrTranscriptRecord) -> AsrTranscriptResult:
    return AsrTranscriptResult(
        model_id=record.model_id,
        text=record.transcript_text,
        segments=record.segments,
        processing_time_seconds=record.processing_time_seconds,
        runtime=record.runtime,
        error=record.error,
    )


def optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def datetime_from_row(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise ValueError(f"Expected datetime row value, got {type(value).__name__}.")
