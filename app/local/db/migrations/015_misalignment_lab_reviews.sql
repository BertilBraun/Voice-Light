CREATE TABLE IF NOT EXISTS misalignment_lab_reviews (
  candidate_id uuid PRIMARY KEY,
  sample_id uuid NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  window_start_seconds double precision NOT NULL CHECK (window_start_seconds >= 0.0),
  window_end_seconds double precision NOT NULL
    CHECK (window_end_seconds > window_start_seconds),
  judgment text NOT NULL
    CHECK (judgment IN ('plausibly_aligned', 'likely_misaligned', 'unsure')),
  queue_seed text NOT NULL CHECK (length(queue_seed) BETWEEN 1 AND 100),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS misalignment_lab_reviews_sample_id_index
  ON misalignment_lab_reviews (sample_id);

CREATE INDEX IF NOT EXISTS misalignment_lab_reviews_judgment_index
  ON misalignment_lab_reviews (judgment);
