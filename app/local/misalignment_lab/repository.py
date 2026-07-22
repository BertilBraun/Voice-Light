from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from app.local.misalignment_lab.models import (
    MisalignmentGlobalCountercheckRequest,
    MisalignmentGlobalCountercheckStored,
    MisalignmentJudgmentRequest,
    MisalignmentRepairJudgmentRequest,
    MisalignmentRepairStoredJudgment,
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

    def list_repair_judgments(self) -> tuple[MisalignmentRepairStoredJudgment, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                  misalignment_lab_repair_reviews.*,
                  samples.external_id
                FROM misalignment_lab_repair_reviews
                JOIN samples ON samples.id = misalignment_lab_repair_reviews.sample_id
                ORDER BY misalignment_lab_repair_reviews.created_at, sample_id
                """
            ).fetchall()
        return tuple(MisalignmentRepairStoredJudgment.model_validate(row) for row in rows)

    def save_repair_judgment(
        self,
        request: MisalignmentRepairJudgmentRequest,
    ) -> MisalignmentRepairStoredJudgment:
        with self.connection() as connection:
            row = connection.execute(
                """
                WITH stored AS (
                  INSERT INTO misalignment_lab_repair_reviews (
                    sample_id,
                    candidate_id,
                    predicted_shift_seconds,
                    estimator_version,
                    repair_scope,
                    first_part_shift_seconds,
                    change_point_seconds,
                    change_interval_start_seconds,
                    change_interval_end_seconds,
                    transition_confirmed,
                    judgment,
                    updated_at
                  )
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                  ON CONFLICT (sample_id) DO UPDATE
                  SET candidate_id = EXCLUDED.candidate_id,
                      predicted_shift_seconds = EXCLUDED.predicted_shift_seconds,
                      estimator_version = EXCLUDED.estimator_version,
                      repair_scope = EXCLUDED.repair_scope,
                      first_part_shift_seconds = EXCLUDED.first_part_shift_seconds,
                      change_point_seconds = EXCLUDED.change_point_seconds,
                      change_interval_start_seconds = EXCLUDED.change_interval_start_seconds,
                      change_interval_end_seconds = EXCLUDED.change_interval_end_seconds,
                      transition_confirmed = EXCLUDED.transition_confirmed,
                      judgment = EXCLUDED.judgment,
                      updated_at = now()
                  RETURNING *
                )
                SELECT stored.*, samples.external_id
                FROM stored
                JOIN samples ON samples.id = stored.sample_id
                """,
                (
                    request.sample_id,
                    request.candidate_id,
                    request.predicted_shift_seconds,
                    request.estimator_version,
                    request.repair_scope.value,
                    request.first_part_shift_seconds,
                    request.change_point_seconds,
                    request.change_interval_start_seconds,
                    request.change_interval_end_seconds,
                    request.transition_confirmed,
                    request.judgment.value,
                ),
            ).fetchone()
        if row is None:
            raise ValueError(f"Sample not found: {request.sample_id}")
        return MisalignmentRepairStoredJudgment.model_validate(row)

    def list_global_counterchecks(
        self,
    ) -> tuple[MisalignmentGlobalCountercheckStored, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                  misalignment_lab_global_counterchecks.*,
                  samples.external_id
                FROM misalignment_lab_global_counterchecks
                JOIN samples ON samples.id = misalignment_lab_global_counterchecks.sample_id
                ORDER BY misalignment_lab_global_counterchecks.created_at, sample_id
                """
            ).fetchall()
        return tuple(MisalignmentGlobalCountercheckStored.model_validate(row) for row in rows)

    def save_global_countercheck(
        self,
        request: MisalignmentGlobalCountercheckRequest,
        beginning_start_seconds: float,
        beginning_end_seconds: float,
        ending_start_seconds: float,
        ending_end_seconds: float,
    ) -> MisalignmentGlobalCountercheckStored:
        with self.connection() as connection:
            row = connection.execute(
                """
                WITH stored AS (
                  INSERT INTO misalignment_lab_global_counterchecks (
                    sample_id,
                    beginning_candidate_id,
                    ending_candidate_id,
                    beginning_start_seconds,
                    beginning_end_seconds,
                    ending_start_seconds,
                    ending_end_seconds,
                    predicted_shift_seconds,
                    estimator_version,
                    judgment,
                    updated_at
                  )
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                  ON CONFLICT (sample_id) DO UPDATE
                  SET beginning_candidate_id = EXCLUDED.beginning_candidate_id,
                      ending_candidate_id = EXCLUDED.ending_candidate_id,
                      beginning_start_seconds = EXCLUDED.beginning_start_seconds,
                      beginning_end_seconds = EXCLUDED.beginning_end_seconds,
                      ending_start_seconds = EXCLUDED.ending_start_seconds,
                      ending_end_seconds = EXCLUDED.ending_end_seconds,
                      predicted_shift_seconds = EXCLUDED.predicted_shift_seconds,
                      estimator_version = EXCLUDED.estimator_version,
                      judgment = EXCLUDED.judgment,
                      updated_at = now()
                  RETURNING *
                )
                SELECT stored.*, samples.external_id
                FROM stored
                JOIN samples ON samples.id = stored.sample_id
                """,
                (
                    request.sample_id,
                    request.beginning_candidate_id,
                    request.ending_candidate_id,
                    beginning_start_seconds,
                    beginning_end_seconds,
                    ending_start_seconds,
                    ending_end_seconds,
                    request.predicted_shift_seconds,
                    request.estimator_version,
                    request.judgment.value,
                ),
            ).fetchone()
        if row is None:
            raise ValueError(f"Sample not found: {request.sample_id}")
        return MisalignmentGlobalCountercheckStored.model_validate(row)
