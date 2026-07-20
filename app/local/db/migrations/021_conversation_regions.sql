CREATE TABLE IF NOT EXISTS conversation_region_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id uuid NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  analysis_version text NOT NULL,
  annotation_version text NOT NULL,
  usable_duration_seconds double precision NOT NULL CHECK (usable_duration_seconds >= 0),
  unusable_duration_seconds double precision NOT NULL CHECK (unusable_duration_seconds >= 0),
  usable_ratio double precision NOT NULL CHECK (usable_ratio >= 0 AND usable_ratio <= 1),
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sample_id, analysis_version, annotation_version)
);

CREATE INDEX IF NOT EXISTS idx_conversation_region_results_sample
  ON conversation_region_results(sample_id, updated_at DESC);
