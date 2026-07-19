from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from app.local.misalignment_lab.models import (
    MisalignmentJudgmentRequest,
    MisalignmentStoredJudgment,
)


class MisalignmentLabRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def list_judgments(self) -> tuple[MisalignmentStoredJudgment, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                  misalignment_lab_reviews.*,
                  samples.external_id
                FROM misalignment_lab_reviews
                JOIN samples ON samples.id = misalignment_lab_reviews.sample_id
                ORDER BY misalignment_lab_reviews.created_at, candidate_id
                """
            ).fetchall()
        return tuple(MisalignmentStoredJudgment.model_validate(row) for row in rows)

    def save_judgment(
        self,
        request: MisalignmentJudgmentRequest,
    ) -> MisalignmentStoredJudgment:
        with self.connection() as connection:
            row = connection.execute(
                """
                WITH stored AS (
                  INSERT INTO misalignment_lab_reviews (
                    candidate_id,
                    sample_id,
                    window_start_seconds,
                    window_end_seconds,
                    judgment,
                    queue_seed,
                    updated_at
                  )
                  VALUES (%s, %s, %s, %s, %s, %s, now())
                  ON CONFLICT (candidate_id) DO UPDATE
                  SET judgment = EXCLUDED.judgment,
                      queue_seed = EXCLUDED.queue_seed,
                      updated_at = now()
                  RETURNING *
                )
                SELECT stored.*, samples.external_id
                FROM stored
                JOIN samples ON samples.id = stored.sample_id
                """,
                (
                    request.candidate_id,
                    request.sample_id,
                    request.window_start_seconds,
                    request.window_end_seconds,
                    request.judgment.value,
                    request.queue_seed,
                ),
            ).fetchone()
        if row is None:
            raise ValueError(f"Sample not found: {request.sample_id}")
        return MisalignmentStoredJudgment.model_validate(row)
