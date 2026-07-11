DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'cached_asr_transcripts'
      AND column_name = 'segments'
  ) AND NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'cached_asr_transcripts'
      AND column_name = 'words'
  ) THEN
    ALTER TABLE cached_asr_transcripts RENAME COLUMN segments TO words;
  END IF;
END $$;
