UPDATE samples
SET is_unusable = TRUE,
    updated_at = now()
WHERE EXISTS (
  SELECT 1
  FROM misalignment_lab_global_counterchecks AS countercheck
  WHERE countercheck.sample_id = samples.id
    AND countercheck.judgment = 'needs_transition'
);

CREATE TABLE virtual_timeline_repair_plans (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id uuid NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  plan_version text NOT NULL CHECK (length(plan_version) BETWEEN 1 AND 200),
  plan_fingerprint text NOT NULL UNIQUE CHECK (plan_fingerprint ~ '^[0-9a-f]{64}$'),
  repair_scope text NOT NULL CHECK (repair_scope IN ('global_offset', 'after_change_point')),
  first_part_shift_seconds double precision NOT NULL
    CHECK (first_part_shift_seconds BETWEEN -12.0 AND 12.0),
  second_part_shift_seconds double precision NOT NULL
    CHECK (second_part_shift_seconds BETWEEN -12.0 AND 12.0),
  change_point_seconds double precision,
  transition_location_source text CHECK (transition_location_source IN ('manual', 'automatic')),
  exclusion_start_seconds double precision,
  exclusion_end_seconds double precision,
  speaker1_audio_sha256 text NOT NULL CHECK (speaker1_audio_sha256 ~ '^[0-9a-f]{64}$'),
  speaker2_audio_sha256 text NOT NULL CHECK (speaker2_audio_sha256 ~ '^[0-9a-f]{64}$'),
  quality_result_id uuid NOT NULL REFERENCES quality_results(id),
  speaker1_parakeet_transcript_id uuid NOT NULL REFERENCES full_recording_asr_transcripts(id),
  speaker2_parakeet_transcript_id uuid NOT NULL REFERENCES full_recording_asr_transcripts(id),
  speaker1_canary_transcript_id uuid NOT NULL REFERENCES full_recording_asr_transcripts(id),
  speaker2_canary_transcript_id uuid NOT NULL REFERENCES full_recording_asr_transcripts(id),
  derived_annotation_version text,
  derived_annotation jsonb,
  conversation_regions_version text,
  conversation_regions jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sample_id, plan_version),
  CHECK (
    (
      repair_scope = 'global_offset'
      AND first_part_shift_seconds = second_part_shift_seconds
      AND change_point_seconds IS NULL
      AND transition_location_source IS NULL
      AND exclusion_start_seconds IS NULL
      AND exclusion_end_seconds IS NULL
    )
    OR
    (
      repair_scope = 'after_change_point'
      AND change_point_seconds >= 0.0
      AND transition_location_source IS NOT NULL
      AND exclusion_start_seconds >= 0.0
      AND exclusion_end_seconds > exclusion_start_seconds
      AND change_point_seconds BETWEEN exclusion_start_seconds AND exclusion_end_seconds
    )
  ),
  CHECK (
    (derived_annotation_version IS NULL AND derived_annotation IS NULL)
    OR (derived_annotation_version IS NOT NULL AND derived_annotation IS NOT NULL)
  ),
  CHECK (
    (conversation_regions_version IS NULL AND conversation_regions IS NULL)
    OR (conversation_regions_version IS NOT NULL AND conversation_regions IS NOT NULL)
  )
);

CREATE INDEX idx_virtual_timeline_repair_plans_sample
  ON virtual_timeline_repair_plans (sample_id, created_at DESC);
