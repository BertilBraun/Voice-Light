CREATE TABLE corpus_review_exclusions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  review_item_id uuid NOT NULL UNIQUE
    REFERENCES corpus_review_items(id) ON DELETE RESTRICT,
  sample_id uuid NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  scope text NOT NULL CHECK (scope IN ('recording', 'interval')),
  start_seconds double precision,
  end_seconds double precision,
  reason text NOT NULL CHECK (length(reason) BETWEEN 1 AND 4000),
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK (
    (scope = 'recording' AND start_seconds IS NULL AND end_seconds IS NULL)
    OR
    (scope = 'interval' AND start_seconds >= 0.0 AND end_seconds > start_seconds)
  )
);

CREATE INDEX idx_corpus_review_exclusions_sample
  ON corpus_review_exclusions(sample_id, scope);

ALTER TABLE corpus_review_items
  ADD COLUMN superseded_at timestamptz,
  ADD COLUMN replaces_item_id uuid
    REFERENCES corpus_review_items(id) ON DELETE RESTRICT;

CREATE UNIQUE INDEX idx_corpus_review_items_replaces
  ON corpus_review_items(replaces_item_id)
  WHERE replaces_item_id IS NOT NULL;
