ALTER TABLE quality_results
  DROP CONSTRAINT quality_results_full_asr_provenance_bundle;

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
          metric_version <> 'quality-conversation-full-parakeet-canary-v6'
          AND speaker1_canary_full_asr_transcript_id IS NULL
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
