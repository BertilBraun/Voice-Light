from __future__ import annotations

import json
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.timeline_repair.models import (
    TimelineRepairDerivedArtifacts,
    TimelineRepairPlanCreate,
    TimelineRepairPlanRecord,
    TimelineRepairSource,
)


class TimelineRepairRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def list_eligible_sources(self) -> tuple[TimelineRepairSource, ...]:
        with self.connection() as connection:
            rows = connection.execute(eligible_source_query()).fetchall()
        return tuple(TimelineRepairSource.model_validate(row) for row in rows)

    def sample_requires_repair(self, sample_id: UUID) -> bool:
        with self.connection() as connection:
            rows = connection.execute(
                eligible_source_query("AND eligible.sample_id = %s"),
                (sample_id,),
            ).fetchall()
        return len(rows) == 1

    def create_plan(self, request: TimelineRepairPlanCreate) -> TimelineRepairPlanRecord:
        with self.connection() as connection:
            eligible_rows = connection.execute(
                eligible_source_query("AND eligible.sample_id = %s"),
                (request.source.sample_id,),
            ).fetchall()
            if len(eligible_rows) != 1:
                raise ValueError(f"Sample is not eligible for repair: {request.source.sample_id}")
            current_source = TimelineRepairSource.model_validate(eligible_rows[0])
            if current_source != request.source:
                raise ValueError(
                    "Repair source provenance is stale or does not match the database."
                )
            row = connection.execute(
                """
                INSERT INTO virtual_timeline_repair_plans (
                  sample_id, plan_version, plan_fingerprint, repair_scope,
                  first_part_shift_seconds, second_part_shift_seconds,
                  change_point_seconds, transition_location_source,
                  exclusion_start_seconds, exclusion_end_seconds,
                  speaker1_audio_sha256, speaker2_audio_sha256, quality_result_id,
                  speaker1_parakeet_transcript_id, speaker2_parakeet_transcript_id,
                  speaker1_canary_transcript_id, speaker2_canary_transcript_id
                )
                VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (sample_id, plan_version) DO NOTHING
                RETURNING *
                """,
                plan_insert_parameters(request),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """
                    SELECT plans.*, samples.external_id, samples.duration_seconds
                    FROM virtual_timeline_repair_plans AS plans
                    JOIN samples ON samples.id = plans.sample_id
                    WHERE plans.sample_id = %s AND plans.plan_version = %s
                    """,
                    (request.source.sample_id, request.plan_version),
                ).fetchone()
                assert row is not None
                existing = timeline_repair_plan_record(row)
                if existing.plan_fingerprint != request.fingerprint():
                    raise ValueError("Plan version already exists with different repair inputs.")
                return existing
            row["external_id"] = request.source.external_id
            row["duration_seconds"] = request.source.duration_seconds
        return timeline_repair_plan_record(row)

    def save_derived_artifacts(
        self,
        plan_id: UUID,
        artifacts: TimelineRepairDerivedArtifacts,
    ) -> TimelineRepairPlanRecord:
        with self.connection() as connection:
            row = connection.execute(
                """
                UPDATE virtual_timeline_repair_plans AS plans
                SET derived_annotation_version = %s,
                    derived_annotation = %s,
                    conversation_regions_version = %s,
                    conversation_regions = %s
                FROM samples
                WHERE plans.id = %s
                  AND samples.id = plans.sample_id
                  AND plans.derived_annotation IS NULL
                  AND plans.conversation_regions IS NULL
                RETURNING plans.*, samples.external_id, samples.duration_seconds
                """,
                (
                    artifacts.annotation_version,
                    Jsonb(artifacts.annotation.model_dump(mode="json")),
                    artifacts.conversation_regions_version,
                    Jsonb(artifacts.conversation_regions.model_dump(mode="json")),
                    plan_id,
                ),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"Repair plan is missing or already has derived artifacts: {plan_id}"
                )
        return timeline_repair_plan_record(row)

    def list_plans(self) -> tuple[TimelineRepairPlanRecord, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT plans.*, samples.external_id, samples.duration_seconds
                FROM virtual_timeline_repair_plans AS plans
                JOIN samples ON samples.id = plans.sample_id
                ORDER BY samples.external_id, plans.created_at, plans.id
                """
            ).fetchall()
        return tuple(timeline_repair_plan_record(row) for row in rows)

    def get_complete_plan(
        self,
        sample_id: UUID,
        plan_version: str,
    ) -> TimelineRepairPlanRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT plans.*, samples.external_id, samples.duration_seconds
                FROM virtual_timeline_repair_plans AS plans
                JOIN samples ON samples.id = plans.sample_id
                JOIN sample_tracks AS speaker1
                  ON speaker1.sample_id = samples.id AND speaker1.speaker_index = 1
                JOIN sample_tracks AS speaker2
                  ON speaker2.sample_id = samples.id AND speaker2.speaker_index = 2
                JOIN LATERAL (
                  SELECT quality_results.id
                  FROM quality_results
                  WHERE quality_results.sample_id = samples.id
                    AND quality_results.status = 'completed'
                  ORDER BY quality_results.created_at DESC, quality_results.id DESC
                  LIMIT 1
                ) AS current_quality ON TRUE
                WHERE plans.sample_id = %s
                  AND plans.plan_version = %s
                  AND plans.derived_annotation IS NOT NULL
                  AND plans.conversation_regions IS NOT NULL
                  AND plans.speaker1_audio_sha256 = speaker1.audio_sha256
                  AND (
                    (
                      plans.materialized_at IS NULL
                      AND plans.speaker2_audio_sha256 = speaker2.audio_sha256
                    )
                    OR
                    (
                      plans.materialized_at IS NOT NULL
                      AND plans.materialized_speaker2_audio_sha256 = speaker2.audio_sha256
                    )
                  )
                  AND plans.quality_result_id = current_quality.id
                """,
                (sample_id, plan_version),
            ).fetchone()
        return timeline_repair_plan_record(row) if row is not None else None


def eligible_source_query(eligible_filter: str = "") -> str:
    return f"""
        WITH latest_quality AS (
          SELECT DISTINCT ON (quality_results.sample_id) quality_results.*
          FROM quality_results
          WHERE quality_results.status = 'completed'
            AND quality_results.speaker1_parakeet_full_asr_transcript_id IS NOT NULL
            AND quality_results.speaker2_parakeet_full_asr_transcript_id IS NOT NULL
            AND quality_results.speaker1_canary_full_asr_transcript_id IS NOT NULL
            AND quality_results.speaker2_canary_full_asr_transcript_id IS NOT NULL
          ORDER BY quality_results.sample_id,
                   quality_results.created_at DESC,
                   quality_results.id DESC
        ), eligible AS (
          SELECT samples.id AS sample_id, samples.external_id, samples.duration_seconds,
                 'global_offset' AS repair_scope,
                 countercheck.predicted_shift_seconds AS first_part_shift_seconds,
                 countercheck.predicted_shift_seconds AS second_part_shift_seconds,
                 NULL::double precision AS change_point_seconds,
                 NULL::text AS transition_location_source,
                 NULL::double precision AS exclusion_start_seconds,
                 NULL::double precision AS exclusion_end_seconds,
                 speaker1.id AS speaker1_track_id,
                 speaker2.id AS speaker2_track_id,
                 speaker1.audio_sha256 AS speaker1_audio_sha256,
                 speaker2.audio_sha256 AS speaker2_audio_sha256,
                 quality.id AS quality_result_id,
                 quality.speaker1_parakeet_full_asr_transcript_id
                   AS speaker1_parakeet_transcript_id,
                 quality.speaker2_parakeet_full_asr_transcript_id
                   AS speaker2_parakeet_transcript_id,
                 quality.speaker1_canary_full_asr_transcript_id AS speaker1_canary_transcript_id,
                 quality.speaker2_canary_full_asr_transcript_id AS speaker2_canary_transcript_id
          FROM samples
          JOIN sample_tracks AS speaker1
            ON speaker1.sample_id = samples.id AND speaker1.speaker_index = 1
          JOIN sample_tracks AS speaker2
            ON speaker2.sample_id = samples.id AND speaker2.speaker_index = 2
          JOIN latest_quality AS quality ON quality.sample_id = samples.id
          JOIN misalignment_lab_global_counterchecks AS countercheck
            ON countercheck.sample_id = samples.id
           AND countercheck.judgment = 'global_offset_confirmed'
          WHERE samples.is_unusable = FALSE
            AND NOT EXISTS (
              SELECT 1
              FROM virtual_timeline_repair_plans AS completed_plan
              WHERE completed_plan.sample_id = samples.id
                AND completed_plan.materialized_at IS NOT NULL
            )
          UNION ALL
          SELECT samples.id, samples.external_id, samples.duration_seconds,
                 'after_change_point', repair.first_part_shift_seconds,
                 repair.predicted_shift_seconds, repair.change_point_seconds, 'manual',
                 greatest(0.0, repair.change_point_seconds - 10.0),
                 least(samples.duration_seconds, repair.change_point_seconds + 10.0),
                 speaker1.id, speaker2.id,
                 speaker1.audio_sha256, speaker2.audio_sha256, quality.id,
                 quality.speaker1_parakeet_full_asr_transcript_id,
                 quality.speaker2_parakeet_full_asr_transcript_id,
                 quality.speaker1_canary_full_asr_transcript_id,
                 quality.speaker2_canary_full_asr_transcript_id
          FROM samples
          JOIN sample_tracks AS speaker1
            ON speaker1.sample_id = samples.id AND speaker1.speaker_index = 1
          JOIN sample_tracks AS speaker2
            ON speaker2.sample_id = samples.id AND speaker2.speaker_index = 2
          JOIN latest_quality AS quality ON quality.sample_id = samples.id
          JOIN misalignment_lab_repair_reviews AS repair ON repair.sample_id = samples.id
          WHERE samples.is_unusable = FALSE
            AND NOT EXISTS (
              SELECT 1
              FROM virtual_timeline_repair_plans AS completed_plan
              WHERE completed_plan.sample_id = samples.id
                AND completed_plan.materialized_at IS NOT NULL
            )
            AND repair.repair_scope = 'after_change_point'
            AND repair.judgment = 'plausible'
            AND repair.transition_confirmed = TRUE
            AND repair.change_point_seconds IS NOT NULL
        )
        SELECT eligible.*
        FROM eligible
        JOIN full_recording_asr_transcripts AS speaker1_parakeet
          ON speaker1_parakeet.id = eligible.speaker1_parakeet_transcript_id
         AND speaker1_parakeet.sample_track_id = eligible.speaker1_track_id
         AND speaker1_parakeet.source_audio_sha256 = eligible.speaker1_audio_sha256
        JOIN full_recording_asr_transcripts AS speaker2_parakeet
          ON speaker2_parakeet.id = eligible.speaker2_parakeet_transcript_id
         AND speaker2_parakeet.sample_track_id = eligible.speaker2_track_id
         AND speaker2_parakeet.source_audio_sha256 = eligible.speaker2_audio_sha256
        JOIN full_recording_asr_transcripts AS speaker1_canary
          ON speaker1_canary.id = eligible.speaker1_canary_transcript_id
         AND speaker1_canary.sample_track_id = eligible.speaker1_track_id
         AND speaker1_canary.source_audio_sha256 = eligible.speaker1_audio_sha256
        JOIN full_recording_asr_transcripts AS speaker2_canary
          ON speaker2_canary.id = eligible.speaker2_canary_transcript_id
         AND speaker2_canary.sample_track_id = eligible.speaker2_track_id
         AND speaker2_canary.source_audio_sha256 = eligible.speaker2_audio_sha256
        WHERE eligible.duration_seconds > 0.0 {eligible_filter}
        ORDER BY eligible.external_id
    """


def plan_insert_parameters(request: TimelineRepairPlanCreate) -> tuple[object, ...]:
    source = request.source
    return (
        source.sample_id,
        request.plan_version,
        request.fingerprint(),
        source.repair_scope.value,
        source.first_part_shift_seconds,
        source.second_part_shift_seconds,
        source.change_point_seconds,
        source.transition_location_source.value if source.transition_location_source else None,
        source.exclusion_start_seconds,
        source.exclusion_end_seconds,
        source.speaker1_audio_sha256,
        source.speaker2_audio_sha256,
        source.quality_result_id,
        source.speaker1_parakeet_transcript_id,
        source.speaker2_parakeet_transcript_id,
        source.speaker1_canary_transcript_id,
        source.speaker2_canary_transcript_id,
    )


def timeline_repair_plan_record(row: dict[str, object]) -> TimelineRepairPlanRecord:
    values = dict(row)
    for field_name in ("derived_annotation", "conversation_regions"):
        value = values[field_name]
        if isinstance(value, str):
            values[field_name] = json.loads(value)
    return TimelineRepairPlanRecord.model_validate(values)
