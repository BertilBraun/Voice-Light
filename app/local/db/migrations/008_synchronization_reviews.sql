CREATE TABLE IF NOT EXISTS synchronization_reviews (
  sample_id uuid PRIMARY KEY REFERENCES samples(id) ON DELETE CASCADE,
  speaker2_shift_seconds double precision NOT NULL
    CHECK (speaker2_shift_seconds >= -12.0 AND speaker2_shift_seconds <= 12.0),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
