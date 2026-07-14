ALTER TABLE quality_results
  ADD COLUMN IF NOT EXISTS annotation_duration_seconds double precision,
  ADD COLUMN IF NOT EXISTS represented_duration_seconds double precision,
  ADD COLUMN IF NOT EXISTS estimated_speech_segment_count integer,
  ADD COLUMN IF NOT EXISTS estimated_interaction_count integer,
  ADD COLUMN IF NOT EXISTS estimated_turn_count integer,
  ADD COLUMN IF NOT EXISTS estimated_turn_taking_count integer,
  ADD COLUMN IF NOT EXISTS estimated_pause_count integer,
  ADD COLUMN IF NOT EXISTS estimated_backchannel_count integer,
  ADD COLUMN IF NOT EXISTS estimated_interruption_count integer,
  ADD COLUMN IF NOT EXISTS estimated_usable_event_count integer;
