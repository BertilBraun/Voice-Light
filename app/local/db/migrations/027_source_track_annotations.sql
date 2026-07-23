CREATE TABLE source_track_annotations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  sample_track_id uuid NOT NULL REFERENCES sample_tracks(id) ON DELETE CASCADE,
  source_name text NOT NULL,
  annotation_version text NOT NULL,
  annotator_id text NOT NULL,
  events jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sample_track_id, source_name, annotation_version, annotator_id)
);

CREATE INDEX source_track_annotations_sample_track_id_idx
  ON source_track_annotations(sample_track_id);
