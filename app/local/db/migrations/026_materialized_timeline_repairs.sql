ALTER TABLE virtual_timeline_repair_plans
  ADD COLUMN materialization_version text,
  ADD COLUMN materialized_speaker2_audio_sha256 text,
  ADD COLUMN materialized_prepared_audio_sha256 text,
  ADD COLUMN materialized_at timestamptz;

ALTER TABLE virtual_timeline_repair_plans
  ADD CONSTRAINT virtual_timeline_materialization_complete CHECK (
    (
      materialization_version IS NULL
      AND materialized_speaker2_audio_sha256 IS NULL
      AND materialized_prepared_audio_sha256 IS NULL
      AND materialized_at IS NULL
    )
    OR
    (
      length(materialization_version) BETWEEN 1 AND 200
      AND materialized_speaker2_audio_sha256 ~ '^[0-9a-f]{64}$'
      AND materialized_prepared_audio_sha256 ~ '^[0-9a-f]{64}$'
      AND materialized_at IS NOT NULL
    )
  );
