ALTER TABLE cached_asr_transcripts
  ADD COLUMN IF NOT EXISTS language_estimate jsonb;

ALTER TABLE full_recording_asr_transcripts
  ADD COLUMN IF NOT EXISTS language_estimate jsonb;
