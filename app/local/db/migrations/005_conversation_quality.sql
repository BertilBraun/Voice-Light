ALTER TABLE quality_results
  ADD COLUMN IF NOT EXISTS conversation_quality_score double precision,
  ADD COLUMN IF NOT EXISTS speech_segment_count integer,
  ADD COLUMN IF NOT EXISTS turn_count integer,
  ADD COLUMN IF NOT EXISTS turn_taking_count integer,
  ADD COLUMN IF NOT EXISTS pause_count integer,
  ADD COLUMN IF NOT EXISTS backchannel_count integer,
  ADD COLUMN IF NOT EXISTS interruption_count integer,
  ADD COLUMN IF NOT EXISTS usable_event_count integer,
  ADD COLUMN IF NOT EXISTS conversation_events_per_hour double precision;

CREATE INDEX IF NOT EXISTS idx_quality_results_conversation_counts
  ON quality_results(turn_taking_count, backchannel_count, interruption_count);
