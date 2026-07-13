CREATE TABLE IF NOT EXISTS cached_asr_transcripts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  audio_sha256 text NOT NULL,
  audio_filename text NOT NULL,
  model_id text NOT NULL,
  transcript_text text NOT NULL,
  words jsonb NOT NULL DEFAULT '[]'::jsonb,
  processing_time_seconds double precision,
  runtime jsonb NOT NULL DEFAULT '{}'::jsonb,
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (audio_sha256, model_id)
);

CREATE INDEX IF NOT EXISTS idx_cached_asr_transcripts_audio_model
  ON cached_asr_transcripts(audio_sha256, model_id);
