from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.conversation_regions.models import CONVERSATION_REGION_ANALYSIS_VERSION
from app.local.corpus_review.models import (
    CorpusReviewDecision,
    CorpusReviewItemRecord,
    CorpusReviewPlan,
    CorpusReviewSetRecord,
    CorpusReviewSetRequest,
    CorpusReviewStatus,
    PlannedCorpusReviewItem,
)
from app.local.db.models import TrackSide
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.local.training_samples.service import (
    FRAME_SECONDS,
    INPUT_DURATION_SECONDS,
    TRAINING_LABEL_VERSION,
)
from app.shared.quality import METRIC_VERSION


class CorpusReviewRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def create_or_get(
        self,
        request: CorpusReviewSetRequest,
        planned_items: tuple[PlannedCorpusReviewItem, ...],
    ) -> CorpusReviewPlan:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            existing = connection.execute(
                "SELECT * FROM corpus_review_sets WHERE name = %s",
                (request.name,),
            ).fetchone()
            if existing is not None:
                review_set = review_set_record(existing)
                if review_set.config != request:
                    raise ValueError(
                        f"Corpus review set {request.name!r} already exists with another config."
                    )
                return CorpusReviewPlan(
                    review_set=review_set,
                    items=self._list_items(connection, review_set.id),
                )
            set_row = connection.execute(
                """
                INSERT INTO corpus_review_sets (name, seed, items_per_dataset, config)
                VALUES (%s, %s, %s, %s)
                RETURNING *
                """,
                (
                    request.name,
                    request.seed,
                    request.items_per_dataset,
                    Jsonb(request.model_dump(mode="json")),
                ),
            ).fetchone()
            assert set_row is not None
            review_set = review_set_record(set_row)
            for item in planned_items:
                inserted = connection.execute(
                    """
                    INSERT INTO corpus_review_items (
                      review_set_id, dataset_id, sample_id, user_side, start_seconds,
                      quality_score, quality_metric_version, annotation_version,
                      quality_result_id, conversation_region_result_id,
                      speaker1_audio_sha256, speaker2_audio_sha256,
                      region_analysis_version, training_label_version,
                      input_duration_seconds, frame_seconds
                    )
                    SELECT %s, %s, %s, %s, %s, %s, %s, %s,
                           quality.id, region_results.id,
                           speaker1.audio_sha256, speaker2.audio_sha256,
                           %s, %s, %s, %s
                    FROM sample_tracks AS speaker1
                    JOIN sample_tracks AS speaker2
                      ON speaker2.sample_id = speaker1.sample_id
                     AND speaker2.side = 'speaker2'
                    JOIN LATERAL (
                      SELECT quality_results.id
                      FROM quality_results
                      WHERE quality_results.sample_id = speaker1.sample_id
                        AND quality_results.metric_version = %s
                        AND quality_results.status = 'completed'
                        AND quality_results.payload -> 'conversation_annotation'
                          ->> 'annotation_version' = %s
                      ORDER BY quality_results.created_at DESC, quality_results.id DESC
                      LIMIT 1
                    ) AS quality ON TRUE
                    JOIN conversation_region_results AS region_results
                      ON region_results.sample_id = speaker1.sample_id
                     AND region_results.analysis_version = %s
                     AND region_results.annotation_version = %s
                    WHERE speaker1.sample_id = %s
                      AND speaker1.side = 'speaker1'
                      AND speaker1.audio_sha256 IS NOT NULL
                      AND speaker2.audio_sha256 IS NOT NULL
                    RETURNING id
                    """,
                    (
                        review_set.id,
                        item.dataset_id,
                        item.sample_id,
                        item.user_side.value,
                        item.start_seconds,
                        item.quality_score,
                        METRIC_VERSION,
                        ANNOTATION_VERSION,
                        CONVERSATION_REGION_ANALYSIS_VERSION,
                        TRAINING_LABEL_VERSION,
                        INPUT_DURATION_SECONDS,
                        FRAME_SECONDS,
                        METRIC_VERSION,
                        ANNOTATION_VERSION,
                        CONVERSATION_REGION_ANALYSIS_VERSION,
                        ANNOTATION_VERSION,
                        item.sample_id,
                    ),
                ).fetchone()
                if inserted is None:
                    raise ValueError(
                        f"Review item provenance is incomplete for sample {item.sample_id}."
                    )
            return CorpusReviewPlan(
                review_set=review_set,
                items=self._list_items(connection, review_set.id),
            )

    def get(self, name: str) -> CorpusReviewPlan:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT * FROM corpus_review_sets WHERE name = %s",
                (name,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Corpus review set not found: {name}")
            review_set = review_set_record(row)
            return CorpusReviewPlan(
                review_set=review_set,
                items=self._list_items(connection, review_set.id),
            )

    def update_decision(
        self,
        item_id: UUID,
        decision: CorpusReviewDecision,
    ) -> CorpusReviewItemRecord:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            row = connection.execute(
                """
                UPDATE corpus_review_items
                SET audio_status = %s,
                    annotation_status = %s,
                    label_status = %s,
                    overall_status = %s,
                    notes = %s,
                    updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (
                    decision.audio_status.value,
                    decision.annotation_status.value,
                    decision.label_status.value,
                    decision.overall_status.value,
                    decision.notes,
                    item_id,
                ),
            ).fetchone()
            if row is None:
                raise ValueError(f"Corpus review item not found: {item_id}")
            return self._item_record(connection, row)

    def _list_items(
        self,
        connection: psycopg.Connection[dict[str, object]],
        review_set_id: UUID,
    ) -> tuple[CorpusReviewItemRecord, ...]:
        rows = connection.execute(
            """
            SELECT corpus_review_items.*,
                   datasets.name AS dataset_name,
                   samples.external_id,
                   (
                     samples.is_unusable = FALSE
                     AND corpus_review_items.speaker1_audio_sha256 = speaker1.audio_sha256
                     AND corpus_review_items.speaker2_audio_sha256 = speaker2.audio_sha256
                     AND corpus_review_items.quality_result_id = (
                       SELECT quality_results.id
                       FROM quality_results
                       WHERE quality_results.sample_id = samples.id
                         AND quality_results.status = 'completed'
                         AND quality_results.metric_version =
                           corpus_review_items.quality_metric_version
                         AND quality_results.payload -> 'conversation_annotation'
                           ->> 'annotation_version' = corpus_review_items.annotation_version
                       ORDER BY quality_results.created_at DESC, quality_results.id DESC
                       LIMIT 1
                     )
                     AND corpus_review_items.quality_score = (
                       SELECT quality_results.total_quality_score
                       FROM quality_results
                       WHERE quality_results.id = corpus_review_items.quality_result_id
                     )
                     AND EXISTS (
                       SELECT 1
                       FROM conversation_region_results AS region_results
                       WHERE region_results.id =
                         corpus_review_items.conversation_region_result_id
                         AND region_results.sample_id = samples.id
                         AND region_results.analysis_version =
                           corpus_review_items.region_analysis_version
                         AND region_results.annotation_version =
                           corpus_review_items.annotation_version
                     )
                   ) AS provenance_current
            FROM corpus_review_items
            JOIN datasets ON datasets.id = corpus_review_items.dataset_id
            JOIN samples ON samples.id = corpus_review_items.sample_id
            JOIN sample_tracks AS speaker1
              ON speaker1.sample_id = samples.id AND speaker1.side = 'speaker1'
            JOIN sample_tracks AS speaker2
              ON speaker2.sample_id = samples.id AND speaker2.side = 'speaker2'
            WHERE corpus_review_items.review_set_id = %s
            ORDER BY datasets.name, samples.external_id, corpus_review_items.start_seconds
            """,
            (review_set_id,),
        ).fetchall()
        return tuple(item_record(row) for row in rows)

    def _item_record(
        self,
        connection: psycopg.Connection[dict[str, object]],
        item_row: dict[str, object],
    ) -> CorpusReviewItemRecord:
        row = connection.execute(
            """
            SELECT corpus_review_items.*,
                   datasets.name AS dataset_name,
                   samples.external_id,
                   (
                     samples.is_unusable = FALSE
                     AND corpus_review_items.speaker1_audio_sha256 = speaker1.audio_sha256
                     AND corpus_review_items.speaker2_audio_sha256 = speaker2.audio_sha256
                     AND corpus_review_items.quality_result_id = (
                       SELECT quality_results.id
                       FROM quality_results
                       WHERE quality_results.sample_id = samples.id
                         AND quality_results.status = 'completed'
                         AND quality_results.metric_version =
                           corpus_review_items.quality_metric_version
                         AND quality_results.payload -> 'conversation_annotation'
                           ->> 'annotation_version' = corpus_review_items.annotation_version
                       ORDER BY quality_results.created_at DESC, quality_results.id DESC
                       LIMIT 1
                     )
                     AND corpus_review_items.quality_score = (
                       SELECT quality_results.total_quality_score
                       FROM quality_results
                       WHERE quality_results.id = corpus_review_items.quality_result_id
                     )
                     AND EXISTS (
                       SELECT 1
                       FROM conversation_region_results AS region_results
                       WHERE region_results.id =
                         corpus_review_items.conversation_region_result_id
                         AND region_results.sample_id = samples.id
                         AND region_results.analysis_version =
                           corpus_review_items.region_analysis_version
                         AND region_results.annotation_version =
                           corpus_review_items.annotation_version
                     )
                   ) AS provenance_current
            FROM corpus_review_items
            JOIN datasets ON datasets.id = corpus_review_items.dataset_id
            JOIN samples ON samples.id = corpus_review_items.sample_id
            JOIN sample_tracks AS speaker1
              ON speaker1.sample_id = samples.id AND speaker1.side = 'speaker1'
            JOIN sample_tracks AS speaker2
              ON speaker2.sample_id = samples.id AND speaker2.side = 'speaker2'
            WHERE corpus_review_items.id = %s
            """,
            (item_row["id"],),
        ).fetchone()
        assert row is not None
        return item_record(row)


def review_set_record(row: dict[str, object]) -> CorpusReviewSetRecord:
    config_value = row["config"]
    config = json.loads(config_value) if isinstance(config_value, str) else config_value
    return CorpusReviewSetRecord(
        id=UUID(str(row["id"])),
        name=str(row["name"]),
        seed=str(row["seed"]),
        items_per_dataset=integer_value(row["items_per_dataset"]),
        config=CorpusReviewSetRequest.model_validate(config),
        created_at=datetime_value(row["created_at"]),
        updated_at=datetime_value(row["updated_at"]),
    )


def item_record(row: dict[str, object]) -> CorpusReviewItemRecord:
    return CorpusReviewItemRecord(
        id=UUID(str(row["id"])),
        review_set_id=UUID(str(row["review_set_id"])),
        dataset_id=UUID(str(row["dataset_id"])),
        dataset_name=str(row["dataset_name"]),
        sample_id=UUID(str(row["sample_id"])),
        external_id=str(row["external_id"]),
        quality_score=float_value(row["quality_score"]),
        quality_result_id=UUID(str(row["quality_result_id"])),
        conversation_region_result_id=UUID(str(row["conversation_region_result_id"])),
        speaker1_audio_sha256=str(row["speaker1_audio_sha256"]),
        speaker2_audio_sha256=str(row["speaker2_audio_sha256"]),
        quality_metric_version=str(row["quality_metric_version"]),
        annotation_version=str(row["annotation_version"]),
        region_analysis_version=str(row["region_analysis_version"]),
        training_label_version=str(row["training_label_version"]),
        input_duration_seconds=float_value(row["input_duration_seconds"]),
        frame_seconds=float_value(row["frame_seconds"]),
        provenance_current=boolean_value(row["provenance_current"]),
        user_side=TrackSide(str(row["user_side"])),
        start_seconds=float_value(row["start_seconds"]),
        audio_status=CorpusReviewStatus(str(row["audio_status"])),
        annotation_status=CorpusReviewStatus(str(row["annotation_status"])),
        label_status=CorpusReviewStatus(str(row["label_status"])),
        overall_status=CorpusReviewStatus(str(row["overall_status"])),
        notes=str(row["notes"]),
        created_at=datetime_value(row["created_at"]),
        updated_at=datetime_value(row["updated_at"]),
    )


def datetime_value(value: object) -> datetime:
    assert isinstance(value, datetime)
    return value


def integer_value(value: object) -> int:
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def float_value(value: object) -> float:
    assert isinstance(value, int | float) and not isinstance(value, bool)
    return float(value)


def boolean_value(value: object) -> bool:
    assert isinstance(value, bool)
    return value
