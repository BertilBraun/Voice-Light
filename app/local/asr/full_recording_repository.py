from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.asr.full_recording_models import (
    FullRecordingAsrTranscriptPair,
    FullRecordingAsrTranscriptRecord,
)
from app.local.db.models import TrackSide
from app.shared.asr import (
    AsrModelId,
    AsrRuntimeStats,
    AsrTranscriptResult,
    LanguageEstimate,
    TimestampedWord,
)


class FullRecordingAsrRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

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

    def get_transcripts_by_ids(
        self,
        transcript_ids: Sequence[UUID],
    ) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
        if not transcript_ids:
            return ()
        with self.connection() as connection:
            rows = connection.execute(
                transcript_select_query("WHERE transcripts.id = ANY(%s)"),
                (list(transcript_ids),),
            ).fetchall()
        records_by_id = {record.id: record for record in (transcript_record(row) for row in rows)}
        missing_ids = tuple(
            transcript_id for transcript_id in transcript_ids if transcript_id not in records_by_id
        )
        if missing_ids:
            raise ValueError(f"Full-recording transcripts not found: {missing_ids}")
        return tuple(records_by_id[transcript_id] for transcript_id in transcript_ids)

    def get_exact_transcript_pair(
        self,
        sample_id: UUID,
        speaker1_track_id: UUID,
        speaker1_source_audio_sha256: str,
        speaker2_track_id: UUID,
        speaker2_source_audio_sha256: str,
        model_id: AsrModelId,
    ) -> FullRecordingAsrTranscriptPair | None:
        speaker1 = self.get_cached_transcripts(
            sample_track_id=speaker1_track_id,
            source_audio_sha256=speaker1_source_audio_sha256,
            model_ids=(model_id,),
        )
        speaker2 = self.get_cached_transcripts(
            sample_track_id=speaker2_track_id,
            source_audio_sha256=speaker2_source_audio_sha256,
            model_ids=(model_id,),
        )
        if len(speaker1) != 1 or len(speaker2) != 1:
            return None
        if speaker1[0].error is not None or speaker2[0].error is not None:
            return None
        if speaker1[0].sample_id != sample_id or speaker2[0].sample_id != sample_id:
            raise ValueError("Full-recording transcript pair belongs to a different sample.")
        if speaker1[0].side is not TrackSide.SPEAKER1:
            raise ValueError("Speaker 1 transcript is linked to the wrong track side.")
        if speaker2[0].side is not TrackSide.SPEAKER2:
            raise ValueError("Speaker 2 transcript is linked to the wrong track side.")
        if speaker1[0].sample_external_id != speaker2[0].sample_external_id:
            raise ValueError("Full-recording transcript pair has inconsistent sample identifiers.")
        return FullRecordingAsrTranscriptPair(
            sample_id=sample_id,
            sample_external_id=speaker1[0].sample_external_id,
            model_id=model_id,
            speaker1=speaker1[0],
            speaker2=speaker2[0],
        )

    def get_current_transcripts(
        self,
        sample_external_id: str,
        side: TrackSide,
        model_ids: Sequence[AsrModelId],
    ) -> tuple[AsrTranscriptResult, ...]:
        if not model_ids:
            return ()
        with self.connection() as connection:
            rows = connection.execute(
                transcript_select_query(
                    """
                    WHERE samples.external_id = %s
                      AND sample_tracks.side = %s
                      AND transcripts.source_audio_sha256 = sample_tracks.audio_sha256
                      AND transcripts.model_id = ANY(%s)
                    """
                ),
                (
                    sample_external_id,
                    side.value,
                    [model_id.value for model_id in model_ids],
                ),
            ).fetchall()
        records = tuple(transcript_record(row) for row in rows)
        result_by_model_id = {
            record.model_id: transcript_result_from_record(record) for record in records
        }
        return tuple(
            result_by_model_id[model_id] for model_id in model_ids if model_id in result_by_model_id
        )

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
                  language_estimate,
                  source_duration_seconds, prepared_duration_seconds,
                  processing_time_seconds, runtime, error, updated_at
                )
                VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()
                )
                ON CONFLICT (sample_track_id, source_audio_sha256, model_id) DO UPDATE
                SET prepared_audio_sha256 = EXCLUDED.prepared_audio_sha256,
                    audio_filename = EXCLUDED.audio_filename,
                    transcript_text = EXCLUDED.transcript_text,
                    words = EXCLUDED.words,
                    language_estimate = EXCLUDED.language_estimate,
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
                    (
                        Jsonb(transcript.language_estimate.model_dump(mode="json"))
                        if transcript.language_estimate is not None
                        else None
                    ),
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


def transcript_record(row: dict[str, object]) -> FullRecordingAsrTranscriptRecord:
    words_value = json_value(row["words"])
    runtime_value = json_value(row["runtime"])
    language_value = json_value(row["language_estimate"])
    if not isinstance(words_value, list):
        raise ValueError("Full-recording ASR words must be a JSON array.")
    if not isinstance(runtime_value, dict):
        raise ValueError("Full-recording ASR runtime must be a JSON object.")
    if language_value is not None and not isinstance(language_value, dict):
        raise ValueError("Full-recording ASR language estimate must be a JSON object.")
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
        language_estimate=LanguageEstimate.model_validate(language_value)
        if language_value is not None
        else None,
        source_duration_seconds=float_from_row(row["source_duration_seconds"]),
        prepared_duration_seconds=float_from_row(row["prepared_duration_seconds"]),
        processing_time_seconds=optional_float_from_row(row["processing_time_seconds"]),
        runtime=AsrRuntimeStats.model_validate(runtime_value) if runtime_value else None,
        error=optional_string_from_row(row["error"]),
        created_at=datetime_from_row(row["created_at"]),
        updated_at=datetime_from_row(row["updated_at"]),
    )


def transcript_result_from_record(
    record: FullRecordingAsrTranscriptRecord,
) -> AsrTranscriptResult:
    return AsrTranscriptResult(
        model_id=record.model_id,
        text=record.transcript_text,
        words=record.words,
        language_estimate=record.language_estimate,
        processing_time_seconds=record.processing_time_seconds,
        runtime=record.runtime,
        error=record.error,
    )


def json_value(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


def uuid_from_row(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def string_from_row(value: object) -> str:
    assert isinstance(value, str)
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
