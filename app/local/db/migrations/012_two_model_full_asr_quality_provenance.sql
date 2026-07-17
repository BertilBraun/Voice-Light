ALTER TABLE quality_results
  DROP CONSTRAINT IF EXISTS quality_results_full_asr_provenance_pair;

ALTER TABLE quality_results
  RENAME COLUMN speaker1_full_asr_transcript_id
    TO speaker1_parakeet_full_asr_transcript_id;

ALTER TABLE quality_results
  RENAME COLUMN speaker2_full_asr_transcript_id
    TO speaker2_parakeet_full_asr_transcript_id;

ALTER TABLE quality_results
  ADD COLUMN speaker1_canary_full_asr_transcript_id uuid
    REFERENCES full_recording_asr_transcripts(id),
  ADD COLUMN speaker2_canary_full_asr_transcript_id uuid
    REFERENCES full_recording_asr_transcripts(id);

ALTER TABLE quality_results
  ADD CONSTRAINT quality_results_full_asr_provenance_bundle
  CHECK (
    (
      speaker1_parakeet_full_asr_transcript_id IS NULL
      AND speaker2_parakeet_full_asr_transcript_id IS NULL
      AND speaker1_canary_full_asr_transcript_id IS NULL
      AND speaker2_canary_full_asr_transcript_id IS NULL
      AND speaker2_shift_seconds IS NULL
      AND synchronization_alignment_origin IS NULL
    )
    OR
    (
      speaker1_parakeet_full_asr_transcript_id IS NOT NULL
      AND speaker2_parakeet_full_asr_transcript_id IS NOT NULL
      AND (
        (
          speaker1_canary_full_asr_transcript_id IS NULL
          AND speaker2_canary_full_asr_transcript_id IS NULL
        )
        OR
        (
          speaker1_canary_full_asr_transcript_id IS NOT NULL
          AND speaker2_canary_full_asr_transcript_id IS NOT NULL
        )
      )
      AND speaker2_shift_seconds IS NOT NULL
      AND synchronization_alignment_origin IS NOT NULL
    )
  );

DROP INDEX IF EXISTS idx_quality_results_full_asr_transcripts;

CREATE INDEX idx_quality_results_full_asr_transcripts
  ON quality_results(
    speaker1_parakeet_full_asr_transcript_id,
    speaker2_parakeet_full_asr_transcript_id,
    speaker1_canary_full_asr_transcript_id,
    speaker2_canary_full_asr_transcript_id
  );
