ALTER TABLE sample_tracks
  ADD COLUMN IF NOT EXISTS source_size_bytes bigint;

ALTER TABLE sample_tracks
  ADD COLUMN IF NOT EXISTS source_etag text;

ALTER TABLE sample_tracks
  DROP CONSTRAINT IF EXISTS sample_tracks_source_size_positive;

ALTER TABLE sample_tracks
  ADD CONSTRAINT sample_tracks_source_size_positive
  CHECK (source_size_bytes IS NULL OR source_size_bytes > 0);
