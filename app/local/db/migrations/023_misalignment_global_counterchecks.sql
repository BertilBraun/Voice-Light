CREATE TABLE misalignment_lab_global_counterchecks (
  sample_id uuid PRIMARY KEY REFERENCES samples(id) ON DELETE CASCADE,
  beginning_candidate_id uuid NOT NULL,
  ending_candidate_id uuid NOT NULL,
  beginning_start_seconds double precision NOT NULL CHECK (beginning_start_seconds >= 0.0),
  beginning_end_seconds double precision NOT NULL CHECK (beginning_end_seconds > beginning_start_seconds),
  ending_start_seconds double precision NOT NULL CHECK (ending_start_seconds >= 0.0),
  ending_end_seconds double precision NOT NULL CHECK (ending_end_seconds > ending_start_seconds),
  predicted_shift_seconds double precision NOT NULL CHECK (predicted_shift_seconds BETWEEN -12.0 AND 12.0),
  estimator_version text NOT NULL,
  judgment text NOT NULL CHECK (
    judgment IN (
      'needs_transition',
      'global_offset_confirmed',
      'not_repairable',
      'unsure'
    )
  ),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
