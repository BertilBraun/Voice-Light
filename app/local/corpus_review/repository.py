from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

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
                connection.execute(
                    """
                    INSERT INTO corpus_review_items (
                      review_set_id, dataset_id, sample_id, user_side, start_seconds,
                      quality_score, quality_metric_version, annotation_version
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                    ),
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
                   samples.external_id
            FROM corpus_review_items
            JOIN datasets ON datasets.id = corpus_review_items.dataset_id
            JOIN samples ON samples.id = corpus_review_items.sample_id
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
                   samples.external_id
            FROM corpus_review_items
            JOIN datasets ON datasets.id = corpus_review_items.dataset_id
            JOIN samples ON samples.id = corpus_review_items.sample_id
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
