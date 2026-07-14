ALTER TABLE quality_results
  ADD COLUMN IF NOT EXISTS interaction_count integer;

CREATE INDEX IF NOT EXISTS idx_quality_results_interaction_count
  ON quality_results(interaction_count);
