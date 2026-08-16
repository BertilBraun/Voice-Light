ALTER TABLE corpus_review_items
  ADD COLUMN quality_result_id uuid REFERENCES quality_results(id),
  ADD COLUMN conversation_region_result_id uuid REFERENCES conversation_region_results(id),
  ADD COLUMN speaker1_audio_sha256 text,
  ADD COLUMN speaker2_audio_sha256 text,
  ADD COLUMN region_analysis_version text NOT NULL DEFAULT 'conversation-regions-v1',
  ADD COLUMN training_label_version text NOT NULL DEFAULT 'turn-taking-frame-labels-v1',
  ADD COLUMN input_duration_seconds double precision NOT NULL DEFAULT 20.0,
  ADD COLUMN frame_seconds double precision NOT NULL DEFAULT 0.08;

UPDATE corpus_review_items AS review_items
SET quality_result_id = (
      SELECT quality_results.id
      FROM quality_results
      WHERE quality_results.sample_id = review_items.sample_id
        AND quality_results.metric_version = review_items.quality_metric_version
        AND quality_results.status = 'completed'
        AND quality_results.payload -> 'conversation_annotation'
          ->> 'annotation_version' = review_items.annotation_version
      ORDER BY quality_results.created_at DESC, quality_results.id DESC
      LIMIT 1
    ),
    conversation_region_result_id = (
      SELECT region_results.id
      FROM conversation_region_results AS region_results
      WHERE region_results.sample_id = review_items.sample_id
        AND region_results.analysis_version = 'conversation-regions-v1'
        AND region_results.annotation_version = review_items.annotation_version
      LIMIT 1
    ),
    speaker1_audio_sha256 = speaker1.audio_sha256,
    speaker2_audio_sha256 = speaker2.audio_sha256
FROM sample_tracks AS speaker1,
     sample_tracks AS speaker2
WHERE speaker1.sample_id = review_items.sample_id
  AND speaker1.side = 'speaker1'
  AND speaker2.sample_id = review_items.sample_id
  AND speaker2.side = 'speaker2';

ALTER TABLE corpus_review_items
  ALTER COLUMN quality_result_id SET NOT NULL,
  ALTER COLUMN conversation_region_result_id SET NOT NULL,
  ALTER COLUMN speaker1_audio_sha256 SET NOT NULL,
  ALTER COLUMN speaker2_audio_sha256 SET NOT NULL,
  ALTER COLUMN region_analysis_version DROP DEFAULT,
  ALTER COLUMN training_label_version DROP DEFAULT,
  ALTER COLUMN input_duration_seconds DROP DEFAULT,
  ALTER COLUMN frame_seconds DROP DEFAULT,
  ADD CONSTRAINT corpus_review_speaker1_audio_sha256_format
    CHECK (speaker1_audio_sha256 ~ '^[0-9a-f]{64}$'),
  ADD CONSTRAINT corpus_review_speaker2_audio_sha256_format
    CHECK (speaker2_audio_sha256 ~ '^[0-9a-f]{64}$'),
  ADD CONSTRAINT corpus_review_input_duration_positive
    CHECK (input_duration_seconds > 0.0),
  ADD CONSTRAINT corpus_review_frame_seconds_positive
    CHECK (frame_seconds > 0.0);
