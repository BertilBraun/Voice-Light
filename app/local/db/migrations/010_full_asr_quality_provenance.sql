ALTER TABLE quality_results
  ADD COLUMN IF NOT EXISTS speaker1_full_asr_transcript_id uuid
    REFERENCES full_recording_asr_transcripts(id),
  ADD COLUMN IF NOT EXISTS speaker2_full_asr_transcript_id uuid
    REFERENCES full_recording_asr_transcripts(id),
  ADD COLUMN IF NOT EXISTS speaker2_shift_seconds double precision,
  ADD COLUMN IF NOT EXISTS synchronization_alignment_origin text;

ALTER TABLE quality_results
  ADD CONSTRAINT quality_results_full_asr_provenance_pair
  CHECK (
    (
      speaker1_full_asr_transcript_id IS NULL
      AND speaker2_full_asr_transcript_id IS NULL
      AND speaker2_shift_seconds IS NULL
      AND synchronization_alignment_origin IS NULL
    )
    OR
    (
      speaker1_full_asr_transcript_id IS NOT NULL
      AND speaker2_full_asr_transcript_id IS NOT NULL
      AND speaker2_shift_seconds IS NOT NULL
      AND synchronization_alignment_origin IS NOT NULL
    )
  );

CREATE INDEX IF NOT EXISTS idx_quality_results_full_asr_transcripts
  ON quality_results(speaker1_full_asr_transcript_id, speaker2_full_asr_transcript_id);
