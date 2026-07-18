from __future__ import annotations

import json
import re
from dataclasses import dataclass
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.asr.transcript import SpeakerTrack
from app.local.synchronization_review.calibration import ReviewedAlignment
from app.local.synchronization_review.models import (
    AlignmentEstimateOrigin,
    SynchronizationCandidate,
)
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.quality import AudioQualityMetrics, ConversationAnnotation

FILENAME_PATTERN = re.compile(r"^(pmt_\d+)_(speaker1|speaker2)\.flac$")


@dataclass(frozen=True)
class TranscriptPair:
    external_id: str
    model_id: AsrModelId
    speaker1_words: tuple[TimestampedWord, ...]
    speaker2_words: tuple[TimestampedWord, ...]


@dataclass(frozen=True)
class StoredConversationAnnotation:
    sample_id: UUID
    external_id: str
    annotation: ConversationAnnotation
    audio_quality: AudioQualityMetrics | None


class SynchronizationReviewRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connection(self) -> psycopg.Connection[dict[str, object]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def load_transcript_pairs(self) -> tuple[TranscriptPair, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT audio_filename, model_id, words
                FROM cached_asr_transcripts
                WHERE model_id IN (%s, %s)
                  AND error IS NULL
                ORDER BY audio_filename, model_id
                """,
                (AsrModelId.PARAKEET_TDT.value, AsrModelId.CANARY.value),
            ).fetchall()

        transcripts: dict[tuple[str, AsrModelId, SpeakerTrack], tuple[TimestampedWord, ...]] = {}
        for row in rows:
            filename_match = FILENAME_PATTERN.fullmatch(str(row["audio_filename"]))
            if filename_match is None:
                continue
            external_id, side_value = filename_match.groups()
            model_id = AsrModelId(str(row["model_id"]))
            side = SpeakerTrack(side_value)
            transcripts[(external_id, model_id, side)] = _timestamped_words(row["words"])

        pairs: list[TranscriptPair] = []
        external_ids = sorted({key[0] for key in transcripts})
        for external_id in external_ids:
            for model_id in (AsrModelId.PARAKEET_TDT, AsrModelId.CANARY):
                speaker1_words = transcripts.get((external_id, model_id, SpeakerTrack.SPEAKER1))
                speaker2_words = transcripts.get((external_id, model_id, SpeakerTrack.SPEAKER2))
                if speaker1_words is None or speaker2_words is None:
                    continue
                pairs.append(
                    TranscriptPair(
                        external_id=external_id,
                        model_id=model_id,
                        speaker1_words=speaker1_words,
                        speaker2_words=speaker2_words,
                    )
                )
        return tuple(pairs)

    def count_pmt_samples(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT count(*) AS sample_count
                FROM samples
                WHERE external_id ~ '^pmt_[0-9]+$'
                """
            ).fetchone()
        assert row is not None
        return int(str(row["sample_count"]))

    def load_reviewed_alignments(self) -> tuple[ReviewedAlignment, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT samples.external_id, synchronization_reviews.speaker2_shift_seconds
                FROM synchronization_reviews
                JOIN samples ON samples.id = synchronization_reviews.sample_id
                ORDER BY samples.external_id
                """
            ).fetchall()
        return tuple(
            ReviewedAlignment(
                external_id=str(row["external_id"]),
                speaker2_shift_seconds=float(row["speaker2_shift_seconds"]),
            )
            for row in rows
        )

    def save_reviewed_alignment(
        self,
        sample_id: UUID,
        speaker2_shift_seconds: float,
    ) -> ReviewedAlignment:
        with self.connection() as connection:
            row = connection.execute(
                """
                WITH saved_review AS (
                  INSERT INTO synchronization_reviews (
                    sample_id, speaker2_shift_seconds, updated_at
                  )
                  VALUES (%s, %s, now())
                  ON CONFLICT (sample_id) DO UPDATE
                  SET speaker2_shift_seconds = EXCLUDED.speaker2_shift_seconds,
                      updated_at = now()
                  RETURNING sample_id, speaker2_shift_seconds
                )
                SELECT samples.external_id, saved_review.speaker2_shift_seconds
                FROM saved_review
                JOIN samples ON samples.id = saved_review.sample_id
                """,
                (sample_id, speaker2_shift_seconds),
            ).fetchone()
            connection.execute(
                """
                DELETE FROM synchronization_predictions
                WHERE sample_id = %s
                """,
                (sample_id,),
            )
        if row is None:
            raise ValueError(f"Sample not found: {sample_id}")
        return ReviewedAlignment(
            external_id=str(row["external_id"]),
            speaker2_shift_seconds=float(row["speaker2_shift_seconds"]),
        )

    def save_unreviewed_predictions(
        self,
        candidates: tuple[SynchronizationCandidate, ...],
        estimator_version: str,
    ) -> int:
        predictions = tuple(
            candidate
            for candidate in candidates
            if candidate.alignment_estimate_origin is AlignmentEstimateOrigin.PREDICTED
        )
        with self.connection() as connection:
            connection.execute(
                """
                DELETE FROM synchronization_predictions
                USING synchronization_reviews
                WHERE synchronization_predictions.sample_id =
                      synchronization_reviews.sample_id
                """
            )
            saved_count = 0
            for candidate in predictions:
                row = connection.execute(
                    """
                    INSERT INTO synchronization_predictions (
                      sample_id, estimator_version, speaker2_shift_seconds,
                      confidence_score, offset_pattern, drift_warning,
                      duration_mismatch_seconds, payload, updated_at
                    )
                    SELECT %s, %s, %s, %s, %s, %s, %s, %s, now()
                    WHERE NOT EXISTS (
                      SELECT 1
                      FROM synchronization_reviews
                      WHERE synchronization_reviews.sample_id = %s
                    )
                    ON CONFLICT (sample_id) DO UPDATE
                    SET estimator_version = EXCLUDED.estimator_version,
                        speaker2_shift_seconds = EXCLUDED.speaker2_shift_seconds,
                        confidence_score = EXCLUDED.confidence_score,
                        offset_pattern = EXCLUDED.offset_pattern,
                        drift_warning = EXCLUDED.drift_warning,
                        duration_mismatch_seconds = EXCLUDED.duration_mismatch_seconds,
                        payload = EXCLUDED.payload,
                        updated_at = now()
                    RETURNING sample_id
                    """,
                    (
                        candidate.sample_id,
                        estimator_version,
                        candidate.estimated_b_shift_seconds,
                        candidate.offset_confidence_score,
                        candidate.offset_pattern.value,
                        candidate.drift_warning,
                        candidate.duration_mismatch_seconds,
                        Jsonb(candidate.model_dump(mode="json")),
                        candidate.sample_id,
                    ),
                ).fetchone()
                if row is not None:
                    saved_count += 1
        return saved_count

    def load_unreviewed_predictions(
        self,
        estimator_version: str,
    ) -> tuple[SynchronizationCandidate, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT synchronization_predictions.payload
                FROM synchronization_predictions
                WHERE synchronization_predictions.estimator_version = %s
                  AND NOT EXISTS (
                    SELECT 1
                    FROM synchronization_reviews
                    WHERE synchronization_reviews.sample_id =
                          synchronization_predictions.sample_id
                  )
                ORDER BY synchronization_predictions.sample_id
                """,
                (estimator_version,),
            ).fetchall()
        return tuple(SynchronizationCandidate.model_validate(row["payload"]) for row in rows)

    def load_annotations(self, metric_version: str) -> tuple[StoredConversationAnnotation, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                WITH latest_quality AS (
                  SELECT DISTINCT ON (quality_results.sample_id)
                    quality_results.sample_id,
                    samples.external_id,
                    quality_results.payload -> 'conversation_annotation' AS annotation,
                    quality_results.payload -> 'audio_quality' AS audio_quality
                  FROM quality_results
                  JOIN samples ON samples.id = quality_results.sample_id
                  WHERE quality_results.metric_version = %s
                    AND quality_results.status = 'completed'
                  ORDER BY quality_results.sample_id, quality_results.created_at DESC
                )
                SELECT sample_id, external_id, annotation
                     , audio_quality
                FROM latest_quality
                WHERE jsonb_typeof(annotation) = 'object'
                ORDER BY external_id
                """,
                (metric_version,),
            ).fetchall()
        return tuple(
            StoredConversationAnnotation(
                sample_id=UUID(str(row["sample_id"])),
                external_id=str(row["external_id"]),
                annotation=ConversationAnnotation.model_validate(row["annotation"]),
                audio_quality=_optional_audio_quality(row["audio_quality"]),
            )
            for row in rows
        )


def _timestamped_words(value: object) -> tuple[TimestampedWord, ...]:
    parsed_value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed_value, list):
        raise ValueError("Cached ASR transcript words must be a JSON array.")
    return tuple(TimestampedWord.model_validate(word) for word in parsed_value)


def _optional_audio_quality(value: object) -> AudioQualityMetrics | None:
    parsed_value = json.loads(value) if isinstance(value, str) else value
    if parsed_value is None:
        return None
    if not isinstance(parsed_value, dict):
        raise ValueError("Stored audio quality must be a JSON object.")
    return AudioQualityMetrics.model_validate(parsed_value)
