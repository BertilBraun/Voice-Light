CREATE TABLE IF NOT EXISTS full_recording_asr_batches (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  idempotency_key text NOT NULL UNIQUE,
  dataset_id uuid NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  sample_scope text NOT NULL
    CHECK (sample_scope IN ('quality_analyzed', 'all_dataset_samples')),
  model_ids text[] NOT NULL CHECK (cardinality(model_ids) > 0),
  status text NOT NULL
    CHECK (status IN ('queued', 'running', 'completed', 'completed_with_errors')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS full_recording_asr_batch_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  batch_id uuid NOT NULL REFERENCES full_recording_asr_batches(id) ON DELETE CASCADE,
  sample_track_id uuid NOT NULL REFERENCES sample_tracks(id) ON DELETE CASCADE,
  status text NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
  attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  UNIQUE (batch_id, sample_track_id)
);

CREATE INDEX IF NOT EXISTS idx_full_recording_asr_items_claim
  ON full_recording_asr_batch_items(batch_id, status, created_at);

CREATE TABLE IF NOT EXISTS full_recording_asr_transcripts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_track_id uuid NOT NULL REFERENCES sample_tracks(id) ON DELETE CASCADE,
  source_audio_sha256 text NOT NULL CHECK (source_audio_sha256 ~ '^[0-9a-f]{64}$'),
  prepared_audio_sha256 text NOT NULL CHECK (prepared_audio_sha256 ~ '^[0-9a-f]{64}$'),
  audio_filename text NOT NULL,
  model_id text NOT NULL,
  transcript_text text NOT NULL,
  words jsonb NOT NULL DEFAULT '[]'::jsonb,
  source_duration_seconds double precision NOT NULL CHECK (source_duration_seconds > 0.0),
  prepared_duration_seconds double precision NOT NULL CHECK (prepared_duration_seconds > 0.0),
  processing_time_seconds double precision,
  runtime jsonb NOT NULL DEFAULT '{}'::jsonb,
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sample_track_id, source_audio_sha256, model_id)
);

CREATE INDEX IF NOT EXISTS idx_full_recording_asr_transcripts_track_model
  ON full_recording_asr_transcripts(sample_track_id, model_id, updated_at DESC);
