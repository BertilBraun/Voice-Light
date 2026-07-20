from __future__ import annotations

import json
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.ingestion.language import (
    LanguageProbeWindow,
    TrackLanguageAssessment,
    TrackLanguageStatus,
)


class LanguageAssessmentRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def get_current_assessment(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        assessment_version: str,
    ) -> TrackLanguageAssessment | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM track_language_assessments
                WHERE sample_track_id = %s
                  AND source_audio_sha256 = %s
                  AND assessment_version = %s
                """,
                (sample_track_id, source_audio_sha256, assessment_version),
            ).fetchone()
        return language_assessment_from_row(row) if row is not None else None

    def upsert_assessment(
        self,
        assessment: TrackLanguageAssessment,
    ) -> TrackLanguageAssessment:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO track_language_assessments (
                  sample_track_id, source_audio_sha256, assessment_version, status,
                  language_code, confidence, transcript_word_count, transcript_text,
                  probe_windows, error, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (
                  sample_track_id, source_audio_sha256, assessment_version
                ) DO UPDATE
                SET status = EXCLUDED.status,
                    language_code = EXCLUDED.language_code,
                    confidence = EXCLUDED.confidence,
                    transcript_word_count = EXCLUDED.transcript_word_count,
                    transcript_text = EXCLUDED.transcript_text,
                    probe_windows = EXCLUDED.probe_windows,
                    error = EXCLUDED.error,
                    updated_at = now()
                RETURNING *
                """,
                (
                    assessment.sample_track_id,
                    assessment.source_audio_sha256,
                    assessment.assessment_version,
                    assessment.status.value,
                    assessment.language_code,
                    assessment.confidence,
                    assessment.transcript_word_count,
                    assessment.transcript_text,
                    Jsonb([window.model_dump(mode="json") for window in assessment.probe_windows]),
                    assessment.error,
                ),
            ).fetchone()
        assert row is not None
        return language_assessment_from_row(row)


def language_assessment_from_row(
    row: dict[str, object],
) -> TrackLanguageAssessment:
    windows_value = row["probe_windows"]
    windows = json.loads(windows_value) if isinstance(windows_value, str) else windows_value
    if not isinstance(windows, list):
        raise ValueError("Language assessment probe windows must be a JSON array.")
    return TrackLanguageAssessment(
        sample_track_id=uuid_from_row(row["sample_track_id"]),
        source_audio_sha256=string_from_row(row["source_audio_sha256"]),
        assessment_version=string_from_row(row["assessment_version"]),
        status=TrackLanguageStatus(string_from_row(row["status"])),
        language_code=optional_string_from_row(row["language_code"]),
        confidence=optional_float_from_row(row["confidence"]),
        transcript_word_count=integer_from_row(row["transcript_word_count"]),
        transcript_text=string_from_row(row["transcript_text"]),
        probe_windows=tuple(LanguageProbeWindow.model_validate(window) for window in windows),
        error=optional_string_from_row(row["error"]),
    )


def uuid_from_row(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def string_from_row(value: object) -> str:
    assert isinstance(value, str)
    return value


def optional_string_from_row(value: object) -> str | None:
    return string_from_row(value) if value is not None else None


def optional_float_from_row(value: object) -> float | None:
    if value is None:
        return None
    assert isinstance(value, int | float)
    return float(value)


def integer_from_row(value: object) -> int:
    assert isinstance(value, int)
    return value
