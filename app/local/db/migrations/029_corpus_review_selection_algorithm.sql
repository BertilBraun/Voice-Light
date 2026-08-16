UPDATE corpus_review_sets
SET config = jsonb_set(
      config,
      '{selection_algorithm}',
      '"dataset-id-sha256-v1"'::jsonb,
      true
    ),
    updated_at = now()
WHERE NOT config ? 'selection_algorithm';
