CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS datasets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  storage_kind text NOT NULL,
  root_uri text NOT NULL,
  description text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (name, root_uri)
);

CREATE TABLE IF NOT EXISTS samples (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id uuid NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  external_id text NOT NULL,
  duration_seconds double precision,
  quality_score double precision,
  quality_flags text[] NOT NULL DEFAULT ARRAY[]::text[],
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (dataset_id, external_id)
);

CREATE INDEX IF NOT EXISTS idx_samples_dataset_duration ON samples(dataset_id, duration_seconds);
CREATE INDEX IF NOT EXISTS idx_samples_quality_score ON samples(quality_score);
CREATE INDEX IF NOT EXISTS idx_samples_quality_flags ON samples USING gin(quality_flags);

CREATE TABLE IF NOT EXISTS sample_tracks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id uuid NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  side text NOT NULL,
  speaker_index integer NOT NULL,
  storage_uri text NOT NULL,
  access_uri text NOT NULL,
  duration_seconds double precision,
  sample_rate integer,
  channels integer,
  sample_count bigint,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sample_id, side),
  UNIQUE (sample_id, speaker_index)
);

CREATE TABLE IF NOT EXISTS audio_metadata (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_track_id uuid NOT NULL REFERENCES sample_tracks(id) ON DELETE CASCADE,
  duration_seconds double precision NOT NULL,
  sample_rate integer NOT NULL,
  channels integer NOT NULL,
  sample_count bigint,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sample_track_id)
);

CREATE TABLE IF NOT EXISTS quality_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id uuid NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  metric_version text NOT NULL,
  status text NOT NULL,
  total_quality_score double precision,
  raw_quality_score double precision,
  calibrated_quality_score double precision,
  interaction_density_score double precision,
  timing_reliability_score double precision,
  audio_quality_score double precision,
  speech_ratio double precision,
  silence_ratio double precision,
  overlap_ratio double precision,
  duration_mismatch_seconds double precision,
  track_correlation double precision,
  energy_envelope_correlation double precision,
  flags text[] NOT NULL DEFAULT ARRAY[]::text[],
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_quality_results_sample_created ON quality_results(sample_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_quality_results_score ON quality_results(total_quality_score);
CREATE INDEX IF NOT EXISTS idx_quality_results_ratios ON quality_results(speech_ratio, overlap_ratio, silence_ratio);
CREATE INDEX IF NOT EXISTS idx_quality_results_flags ON quality_results USING gin(flags);

CREATE TABLE IF NOT EXISTS asr_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id uuid NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  model_name text NOT NULL,
  status text NOT NULL,
  runtime_seconds double precision,
  real_time_factor double precision,
  error text,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_asr_runs_sample_model ON asr_runs(sample_id, model_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_asr_runs_status ON asr_runs(status);

CREATE TABLE IF NOT EXISTS asr_words (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  asr_run_id uuid NOT NULL REFERENCES asr_runs(id) ON DELETE CASCADE,
  side text NOT NULL,
  word_index integer NOT NULL,
  text text NOT NULL,
  start_seconds double precision,
  end_seconds double precision,
  confidence double precision,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (asr_run_id, side, word_index)
);

CREATE INDEX IF NOT EXISTS idx_asr_words_run_side_time ON asr_words(asr_run_id, side, start_seconds);

CREATE TABLE IF NOT EXISTS asr_evaluations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  asr_run_id uuid NOT NULL REFERENCES asr_runs(id) ON DELETE CASCADE,
  wer double precision,
  substitutions integer NOT NULL,
  insertions integer NOT NULL,
  deletions integer NOT NULL,
  reference_word_count integer NOT NULL,
  start_median_absolute_error double precision,
  start_mean_absolute_error double precision,
  start_p90_absolute_error double precision,
  end_median_absolute_error double precision,
  end_mean_absolute_error double precision,
  end_p90_absolute_error double precision,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (asr_run_id)
);

CREATE INDEX IF NOT EXISTS idx_asr_evaluations_wer ON asr_evaluations(wer);
CREATE INDEX IF NOT EXISTS idx_asr_evaluations_timestamp ON asr_evaluations(start_p90_absolute_error, end_p90_absolute_error);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id uuid REFERENCES datasets(id) ON DELETE SET NULL,
  status text NOT NULL,
  source_uri text NOT NULL,
  message text NOT NULL DEFAULT '',
  total_samples integer NOT NULL DEFAULT 0,
  processed_samples integer NOT NULL DEFAULT 0,
  failed_samples integer NOT NULL DEFAULT 0,
  error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_dataset_created ON ingestion_jobs(dataset_id, created_at DESC);

CREATE TABLE IF NOT EXISTS annotation_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_id uuid NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  status text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS annotation_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  annotation_run_id uuid NOT NULL REFERENCES annotation_runs(id) ON DELETE CASCADE,
  event_type text NOT NULL,
  side text,
  start_seconds double precision NOT NULL,
  end_seconds double precision NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
