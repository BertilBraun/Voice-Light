from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.db.models import TrackVadRecord
from app.shared.quality import TrackVadResult


class VadRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def get_current(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        vad_version: str,
    ) -> TrackVadRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM track_vad_results
                WHERE sample_track_id = %s
                  AND source_audio_sha256 = %s
                  AND vad_version = %s
                """,
                (sample_track_id, source_audio_sha256, vad_version),
            ).fetchone()
        return vad_record(row) if row is not None else None

    def upsert(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        vad_version: str,
        result: TrackVadResult,
    ) -> TrackVadRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO track_vad_results (
                  sample_track_id, source_audio_sha256, vad_version, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (sample_track_id, source_audio_sha256, vad_version) DO UPDATE
                SET payload = EXCLUDED.payload,
                    updated_at = now()
                RETURNING *
                """,
                (
                    sample_track_id,
                    source_audio_sha256,
                    vad_version,
                    Jsonb(result.model_dump(mode="json")),
                ),
            ).fetchone()
        assert row is not None
        return vad_record(row)


def vad_record(row: dict[str, object]) -> TrackVadRecord:
    payload = row["payload"]
    payload_value = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(payload_value, dict):
        raise ValueError("VAD payload must be a JSON object.")
    return TrackVadRecord(
        sample_track_id=uuid_from_row(row["sample_track_id"]),
        source_audio_sha256=string_from_row(row["source_audio_sha256"]),
        vad_version=string_from_row(row["vad_version"]),
        result=TrackVadResult.model_validate(payload_value),
        created_at=datetime_from_row(row["created_at"]),
        updated_at=datetime_from_row(row["updated_at"]),
    )


def uuid_from_row(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def string_from_row(value: object) -> str:
    assert isinstance(value, str)
    return value


def datetime_from_row(value: object) -> datetime:
    assert isinstance(value, datetime)
    return value
