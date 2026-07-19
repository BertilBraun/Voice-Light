CREATE TABLE IF NOT EXISTS misalignment_lab_repair_reviews (
  sample_id uuid PRIMARY KEY REFERENCES samples(id) ON DELETE CASCADE,
  candidate_id uuid NOT NULL,
  predicted_shift_seconds double precision NOT NULL
    CHECK (predicted_shift_seconds BETWEEN -12.0 AND 12.0),
  estimator_version text NOT NULL CHECK (length(estimator_version) BETWEEN 1 AND 200),
  judgment text NOT NULL
    CHECK (judgment IN ('plausible', 'not_plausible', 'unsure')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS misalignment_lab_repair_reviews_judgment_index
  ON misalignment_lab_repair_reviews (judgment);
