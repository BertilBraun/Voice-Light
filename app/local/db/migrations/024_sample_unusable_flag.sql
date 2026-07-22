ALTER TABLE samples
  ADD COLUMN is_unusable boolean NOT NULL DEFAULT FALSE;

UPDATE samples
SET is_unusable = TRUE,
    updated_at = now()
WHERE (
  EXISTS (
    SELECT 1
    FROM misalignment_lab_reviews AS review
    WHERE review.sample_id = samples.id
      AND review.judgment = 'likely_misaligned'
  )
  AND NOT EXISTS (
    SELECT 1
    FROM misalignment_lab_repair_reviews AS repair
    WHERE repair.sample_id = samples.id
      AND repair.repair_scope = 'after_change_point'
      AND repair.judgment = 'plausible'
      AND repair.transition_confirmed = TRUE
  )
  AND NOT EXISTS (
    SELECT 1
    FROM misalignment_lab_global_counterchecks AS countercheck
    WHERE countercheck.sample_id = samples.id
      AND countercheck.judgment IN ('global_offset_confirmed', 'needs_transition')
  )
)
OR EXISTS (
  SELECT 1
  FROM misalignment_lab_global_counterchecks AS countercheck
  WHERE countercheck.sample_id = samples.id
    AND countercheck.judgment = 'not_repairable'
);

CREATE INDEX idx_samples_usable_dataset
  ON samples (dataset_id, updated_at DESC)
  WHERE is_unusable = FALSE;
