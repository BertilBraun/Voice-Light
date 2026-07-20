ALTER TABLE sample_tracks
  DROP CONSTRAINT IF EXISTS sample_tracks_audio_sha256_required;

CREATE TABLE IF NOT EXISTS track_vad_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_track_id uuid NOT NULL REFERENCES sample_tracks(id) ON DELETE CASCADE,
  source_audio_sha256 text NOT NULL CHECK (source_audio_sha256 ~ '^[0-9a-f]{64}$'),
  vad_version text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sample_track_id, source_audio_sha256, vad_version)
);

CREATE INDEX IF NOT EXISTS idx_track_vad_results_track
  ON track_vad_results(sample_track_id, updated_at DESC);
