from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.conversation_regions.models import (
    ConversationRegionAnalysis,
    ConversationRegionDatasetSummary,
    ConversationRegionReason,
    ConversationRegionReasonDuration,
    ConversationRegionResultRecord,
)
from app.shared.quality import ConversationAnnotation, TrackVadResult


@dataclass(frozen=True)
class ConversationRegionEvidence:
    sample_id: UUID
    annotation: ConversationAnnotation
    speaker1_vad: TrackVadResult
    speaker2_vad: TrackVadResult


class ConversationRegionRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def upsert(
        self,
        sample_id: UUID,
        analysis: ConversationRegionAnalysis,
    ) -> ConversationRegionResultRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO conversation_region_results (
                  sample_id, analysis_version, annotation_version,
                  usable_duration_seconds, unusable_duration_seconds,
                  usable_ratio, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (sample_id, analysis_version, annotation_version) DO UPDATE
                SET usable_duration_seconds = EXCLUDED.usable_duration_seconds,
                    unusable_duration_seconds = EXCLUDED.unusable_duration_seconds,
                    usable_ratio = EXCLUDED.usable_ratio,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                RETURNING *
                """,
                (
                    sample_id,
                    analysis.analysis_version,
                    analysis.annotation_version,
                    analysis.usable_duration_seconds,
                    analysis.unusable_duration_seconds,
                    analysis.usable_ratio,
                    Jsonb(analysis.model_dump(mode="json")),
                ),
            ).fetchone()
        assert row is not None
        return conversation_region_result_record(row)

    def get_current(
        self,
        sample_id: UUID,
        analysis_version: str,
        annotation_version: str,
    ) -> ConversationRegionResultRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM conversation_region_results
                WHERE sample_id = %s
                  AND analysis_version = %s
                  AND annotation_version = %s
                """,
                (sample_id, analysis_version, annotation_version),
            ).fetchone()
        return conversation_region_result_record(row) if row is not None else None

    def list_evidence(
        self,
        dataset_id: UUID | None,
        metric_version: str,
        annotation_version: str,
    ) -> tuple[ConversationRegionEvidence, ...]:
        dataset_filter = "AND samples.dataset_id = %s" if dataset_id is not None else ""
        dataset_parameters: tuple[object, ...] = (dataset_id,) if dataset_id is not None else ()
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                  samples.id AS sample_id,
                  latest_quality.payload AS quality_payload,
                  speaker1_vad.payload AS speaker1_vad_payload,
                  speaker2_vad.payload AS speaker2_vad_payload
                FROM samples
                JOIN sample_tracks AS speaker1_track
                  ON speaker1_track.sample_id = samples.id
                 AND speaker1_track.side = 'speaker1'
                JOIN sample_tracks AS speaker2_track
                  ON speaker2_track.sample_id = samples.id
                 AND speaker2_track.side = 'speaker2'
                JOIN LATERAL (
                  SELECT quality_results.payload
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                    AND quality_results.metric_version = %s
                    AND quality_results.status = 'completed'
                    AND quality_results.payload
                      -> 'conversation_annotation'
                      ->> 'annotation_version' = %s
                  ORDER BY quality_results.created_at DESC, quality_results.id DESC
                  LIMIT 1
                ) AS latest_quality ON true
                JOIN LATERAL (
                  SELECT track_vad_results.payload
                  FROM track_vad_results
                  WHERE track_vad_results.sample_track_id = speaker1_track.id
                    AND track_vad_results.source_audio_sha256 = speaker1_track.audio_sha256
                  ORDER BY track_vad_results.updated_at DESC, track_vad_results.id DESC
                  LIMIT 1
                ) AS speaker1_vad ON true
                JOIN LATERAL (
                  SELECT track_vad_results.payload
                  FROM track_vad_results
                  WHERE track_vad_results.sample_track_id = speaker2_track.id
                    AND track_vad_results.source_audio_sha256 = speaker2_track.audio_sha256
                  ORDER BY track_vad_results.updated_at DESC, track_vad_results.id DESC
                  LIMIT 1
                ) AS speaker2_vad ON true
                WHERE true
                {dataset_filter}
                ORDER BY samples.external_id
                """,
                (metric_version, annotation_version, *dataset_parameters),
            ).fetchall()
        return tuple(conversation_region_evidence(row) for row in rows)

    def dataset_summary(
        self,
        dataset_id: UUID | None,
        analysis_version: str,
        metric_version: str,
        annotation_version: str,
        minimum_quality: float | None,
        maximum_overlap_ratio: float | None,
    ) -> ConversationRegionDatasetSummary:
        dataset_filter = "AND samples.dataset_id = %s" if dataset_id is not None else ""
        dataset_parameters: tuple[object, ...] = (dataset_id,) if dataset_id is not None else ()
        quality_filter = (
            "AND latest_quality.total_quality_score >= %s" if minimum_quality is not None else ""
        )
        quality_parameters: tuple[object, ...] = (
            (minimum_quality,) if minimum_quality is not None else ()
        )
        overlap_filter = (
            "AND latest_quality.overlap_ratio <= %s" if maximum_overlap_ratio is not None else ""
        )
        overlap_parameters: tuple[object, ...] = (
            (maximum_overlap_ratio,) if maximum_overlap_ratio is not None else ()
        )
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT region_results.*
                FROM samples
                JOIN conversation_region_results AS region_results
                  ON region_results.sample_id = samples.id
                 AND region_results.analysis_version = %s
                 AND region_results.annotation_version = %s
                JOIN LATERAL (
                  SELECT
                    quality_results.id,
                    quality_results.total_quality_score,
                    quality_results.overlap_ratio
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                    AND quality_results.metric_version = %s
                    AND quality_results.status = 'completed'
                    AND quality_results.payload
                      -> 'conversation_annotation'
                      ->> 'annotation_version' = %s
                  ORDER BY quality_results.created_at DESC, quality_results.id DESC
                  LIMIT 1
                ) AS latest_quality ON true
                WHERE true
                {dataset_filter}
                {quality_filter}
                {overlap_filter}
                ORDER BY samples.external_id
                """,
                (
                    analysis_version,
                    annotation_version,
                    metric_version,
                    annotation_version,
                    *dataset_parameters,
                    *quality_parameters,
                    *overlap_parameters,
                ),
            ).fetchall()
        records = tuple(conversation_region_result_record(row) for row in rows)
        represented_duration_seconds = sum(record.analysis.duration_seconds for record in records)
        usable_duration_seconds = sum(record.analysis.usable_duration_seconds for record in records)
        unusable_duration_seconds = sum(
            record.analysis.unusable_duration_seconds for record in records
        )
        return ConversationRegionDatasetSummary(
            dataset_id=dataset_id,
            analysis_version=analysis_version,
            analyzed_sample_count=len(records),
            represented_duration_seconds=represented_duration_seconds,
            usable_duration_seconds=usable_duration_seconds,
            unusable_duration_seconds=unusable_duration_seconds,
            usable_ratio=(
                usable_duration_seconds / represented_duration_seconds
                if represented_duration_seconds > 0.0
                else 0.0
            ),
            reason_durations=tuple(
                ConversationRegionReasonDuration(
                    reason=reason,
                    duration_seconds=sum(
                        region.end_seconds - region.start_seconds
                        for record in records
                        for region in record.analysis.unusable_regions
                        if reason in region.reasons
                    ),
                )
                for reason in ConversationRegionReason
            ),
        )


def conversation_region_result_record(
    row: dict[str, object],
) -> ConversationRegionResultRecord:
    return ConversationRegionResultRecord(
        sample_id=uuid_from_row(row["sample_id"]),
        analysis_version=string_from_row(row["analysis_version"]),
        annotation_version=string_from_row(row["annotation_version"]),
        analysis=ConversationRegionAnalysis.model_validate(json_object(row["payload"])),
        created_at=datetime_from_row(row["created_at"]),
        updated_at=datetime_from_row(row["updated_at"]),
    )


def conversation_region_evidence(
    row: dict[str, object],
) -> ConversationRegionEvidence:
    quality_payload = json_object(row["quality_payload"])
    annotation_payload = quality_payload.get("conversation_annotation")
    if not isinstance(annotation_payload, dict):
        raise ValueError("Quality payload has no conversation annotation.")
    return ConversationRegionEvidence(
        sample_id=uuid_from_row(row["sample_id"]),
        annotation=ConversationAnnotation.model_validate(annotation_payload),
        speaker1_vad=TrackVadResult.model_validate(json_object(row["speaker1_vad_payload"])),
        speaker2_vad=TrackVadResult.model_validate(json_object(row["speaker2_vad_payload"])),
    )


def json_object(value: object) -> dict[str, object]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def uuid_from_row(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def string_from_row(value: object) -> str:
    assert isinstance(value, str)
    return value


def datetime_from_row(value: object) -> datetime:
    assert isinstance(value, datetime)
    return value
