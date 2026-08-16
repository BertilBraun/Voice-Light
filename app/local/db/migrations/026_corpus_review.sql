CREATE TABLE corpus_review_sets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL UNIQUE,
  seed text NOT NULL,
  items_per_dataset integer NOT NULL CHECK (items_per_dataset > 0),
  config jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE corpus_review_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  review_set_id uuid NOT NULL REFERENCES corpus_review_sets(id) ON DELETE CASCADE,
  dataset_id uuid NOT NULL REFERENCES datasets(id),
  sample_id uuid NOT NULL REFERENCES samples(id),
  user_side text NOT NULL CHECK (user_side IN ('speaker1', 'speaker2')),
  start_seconds double precision NOT NULL CHECK (start_seconds >= 0.0),
  quality_score double precision NOT NULL CHECK (quality_score BETWEEN 0.0 AND 1.0),
  quality_metric_version text NOT NULL,
  annotation_version text NOT NULL,
  audio_status text NOT NULL DEFAULT 'pending'
    CHECK (audio_status IN ('pending', 'pass', 'fail')),
  annotation_status text NOT NULL DEFAULT 'pending'
    CHECK (annotation_status IN ('pending', 'pass', 'fail')),
  label_status text NOT NULL DEFAULT 'pending'
    CHECK (label_status IN ('pending', 'pass', 'fail')),
  overall_status text NOT NULL DEFAULT 'pending'
    CHECK (overall_status IN ('pending', 'pass', 'fail')),
  notes text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (review_set_id, sample_id, user_side, start_seconds)
);

CREATE INDEX idx_corpus_review_items_set
  ON corpus_review_items(review_set_id, dataset_id, created_at);
