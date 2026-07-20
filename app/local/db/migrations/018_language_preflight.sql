CREATE TABLE IF NOT EXISTS track_language_assessments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_track_id uuid NOT NULL REFERENCES sample_tracks(id) ON DELETE CASCADE,
  source_audio_sha256 text NOT NULL CHECK (source_audio_sha256 ~ '^[0-9a-f]{64}$'),
  assessment_version text NOT NULL,
  status text NOT NULL
    CHECK (status IN ('english', 'non_english', 'inconclusive', 'failed')),
  language_code text,
  confidence double precision CHECK (
    confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)
  ),
  transcript_word_count integer NOT NULL CHECK (transcript_word_count >= 0),
  transcript_text text NOT NULL,
  probe_windows jsonb NOT NULL,
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sample_track_id, source_audio_sha256, assessment_version)
);

CREATE INDEX IF NOT EXISTS idx_track_language_assessments_status
  ON track_language_assessments(status, updated_at DESC);
