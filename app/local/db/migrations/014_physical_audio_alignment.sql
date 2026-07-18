ALTER TABLE sample_tracks
  ADD COLUMN IF NOT EXISTS audio_sha256 text;

ALTER TABLE sample_tracks
  ADD CONSTRAINT sample_tracks_audio_sha256_format
  CHECK (
    audio_sha256 IS NULL
    OR audio_sha256 ~ '^[0-9a-f]{64}$'
  ) NOT VALID;

ALTER TABLE sample_tracks
  ADD CONSTRAINT sample_tracks_audio_sha256_required
  CHECK (audio_sha256 IS NOT NULL) NOT VALID;

ALTER TABLE quality_results
  DROP CONSTRAINT IF EXISTS quality_results_full_asr_provenance_bundle;

ALTER TABLE quality_results
  DROP COLUMN IF EXISTS speaker2_shift_seconds,
  DROP COLUMN IF EXISTS synchronization_alignment_origin;

ALTER TABLE quality_results
  ADD CONSTRAINT quality_results_full_asr_provenance_bundle
  CHECK (
    (
      speaker1_parakeet_full_asr_transcript_id IS NULL
      AND speaker2_parakeet_full_asr_transcript_id IS NULL
      AND speaker1_canary_full_asr_transcript_id IS NULL
      AND speaker2_canary_full_asr_transcript_id IS NULL
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
    )
  );
