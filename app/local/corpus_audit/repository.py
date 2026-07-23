from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from app.local.conversation_regions.models import ConversationRegionAnalysis
from app.shared.quality import ConversationAnnotation, QualityResult


@dataclass(frozen=True)
class CorpusAuditEvidence:
    dataset_id: UUID
    dataset_name: str
    sample_id: UUID
    external_id: str
    represented_duration_seconds: float
    quality_score: float
    annotation: ConversationAnnotation
    conversation_regions: ConversationRegionAnalysis | None


class CorpusAuditRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def load_evidence(
        self,
        dataset_ids: tuple[UUID, ...],
        minimum_quality: float,
        metric_version: str,
        annotation_version: str,
        region_analysis_version: str,
    ) -> tuple[CorpusAuditEvidence, ...]:
        if not dataset_ids:
            raise ValueError("At least one dataset is required for a corpus audit.")
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT
                  datasets.id AS dataset_id,
                  datasets.name AS dataset_name,
                  samples.id AS sample_id,
                  samples.external_id,
                  LEAST(
                    COALESCE(
                      samples.duration_seconds,
                      (latest_quality.payload -> 'conversation_annotation'
                        ->> 'analyzed_duration_seconds')::double precision
                    ),
                    (latest_quality.payload -> 'conversation_annotation'
                      ->> 'analyzed_duration_seconds')::double precision
                  ) AS represented_duration_seconds,
                  latest_quality.total_quality_score,
                  latest_quality.payload AS quality_payload,
                  region_results.payload AS region_payload
                FROM samples
                JOIN datasets ON datasets.id = samples.dataset_id
                JOIN LATERAL (
                  SELECT total_quality_score, payload
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                    AND quality_results.metric_version = %s
                    AND quality_results.status = 'completed'
                    AND quality_results.payload -> 'conversation_annotation'
                      ->> 'annotation_version' = %s
                  ORDER BY quality_results.created_at DESC, quality_results.id DESC
                  LIMIT 1
                ) AS latest_quality ON true
                LEFT JOIN conversation_region_results AS region_results
                  ON region_results.sample_id = samples.id
                 AND region_results.analysis_version = %s
                 AND region_results.annotation_version = %s
                WHERE samples.dataset_id = ANY(%s::uuid[])
                  AND samples.is_unusable = FALSE
                  AND latest_quality.total_quality_score > %s
                ORDER BY datasets.name, samples.external_id
                """,
                (
                    metric_version,
                    annotation_version,
                    region_analysis_version,
                    annotation_version,
                    list(dataset_ids),
                    minimum_quality,
                ),
            ).fetchall()
        return tuple(_evidence_from_row(row) for row in rows)


def _evidence_from_row(row: dict[str, object]) -> CorpusAuditEvidence:
    quality = QualityResult.model_validate(row["quality_payload"])
    annotation = quality.conversation_annotation
    assert annotation is not None
    region_payload = row["region_payload"]
    return CorpusAuditEvidence(
        dataset_id=UUID(str(row["dataset_id"])),
        dataset_name=str(row["dataset_name"]),
        sample_id=UUID(str(row["sample_id"])),
        external_id=str(row["external_id"]),
        represented_duration_seconds=float(row["represented_duration_seconds"]),
        quality_score=float(row["total_quality_score"]),
        annotation=annotation,
        conversation_regions=(
            ConversationRegionAnalysis.model_validate(region_payload)
            if region_payload is not None
            else None
        ),
    )
