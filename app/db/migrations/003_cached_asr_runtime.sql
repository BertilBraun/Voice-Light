ALTER TABLE cached_asr_transcripts
  ADD COLUMN IF NOT EXISTS runtime jsonb NOT NULL DEFAULT '{}'::jsonb;
