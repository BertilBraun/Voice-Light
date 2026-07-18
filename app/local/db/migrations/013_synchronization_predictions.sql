CREATE TABLE IF NOT EXISTS synchronization_predictions (
  sample_id uuid PRIMARY KEY REFERENCES samples(id) ON DELETE CASCADE,
  estimator_version text NOT NULL,
  speaker2_shift_seconds double precision NOT NULL
    CHECK (speaker2_shift_seconds >= -12.0 AND speaker2_shift_seconds <= 12.0),
  confidence_score double precision NOT NULL
    CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
  offset_pattern text NOT NULL
    CHECK (offset_pattern IN ('constant', 'variable', 'uncertain')),
  drift_warning text,
  duration_mismatch_seconds double precision,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synchronization_predictions_review_order
  ON synchronization_predictions(confidence_score, updated_at);
