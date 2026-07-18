from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.local.alignment_migration.models import AlignmentSide
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.quality import METRIC_VERSION


@dataclass(frozen=True)
class MigrationDataset:
    id: UUID
    root: Path


@dataclass(frozen=True)
class MigrationTrack:
    id: UUID
    side: AlignmentSide
    storage_uri: str
    access_uri: Path
    audio_sha256: str | None


@dataclass(frozen=True)
class MigrationSample:
    id: UUID
    external_id: str
    tracks: tuple[MigrationTrack, MigrationTrack]


@dataclass(frozen=True)
class StoredAlignmentReview:
    external_id: str
    speaker2_shift_seconds: float


@dataclass(frozen=True)
class FullAsrMigrationRecord:
    id: UUID
    sample_external_id: str
    sample_track_id: UUID
    side: AlignmentSide
    model_id: AsrModelId
    source_audio_sha256: str
    prepared_audio_sha256: str
    source_duration_seconds: float
    prepared_duration_seconds: float
    words: tuple[TimestampedWord, ...]


@dataclass(frozen=True)
class CachedAsrMigrationRecord:
    id: UUID
    audio_filename: str


@dataclass(frozen=True)
class LegacyTableCounts:
    asr_runs: int
    asr_words: int
    asr_evaluations: int
    annotation_runs: int
    annotation_events: int


@dataclass(frozen=True)
class MigrationRepositorySnapshot:
    dataset: MigrationDataset
    samples: tuple[MigrationSample, ...]
    reviews: tuple[StoredAlignmentReview, ...]
    transcripts: tuple[FullAsrMigrationRecord, ...]
    cached_transcripts: tuple[CachedAsrMigrationRecord, ...]
    legacy_counts: LegacyTableCounts
    quality_result_count: int
    current_quality_external_ids: frozenset[str]
    synchronization_prediction_count: int


@dataclass(frozen=True)
class PreparedAudioProvenance:
    sha256: str
    duration_seconds: float


@dataclass(frozen=True)
class TranscriptReconciliation:
    id: UUID
    source_audio_sha256: str
    prepared_audio: PreparedAudioProvenance
    source_duration_seconds: float
    words: tuple[TimestampedWord, ...]


@dataclass(frozen=True)
class TrackReconciliation:
    id: UUID
    audio_sha256: str
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_count: int
    transcripts: tuple[TranscriptReconciliation, TranscriptReconciliation]


@dataclass(frozen=True)
class SampleReconciliation:
    id: UUID
    duration_seconds: float
    tracks: tuple[TrackReconciliation, TrackReconciliation]


@dataclass(frozen=True)
class InvalidSampleDeletion:
    sample_ids: tuple[UUID, ...]
    expected_external_ids: tuple[str, ...]
    cached_transcript_ids: tuple[UUID, ...]


class AlignmentMigrationRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def load_snapshot(self, configured_sessions_root: Path) -> MigrationRepositorySnapshot:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            dataset_rows = connection.execute("SELECT id, root_uri FROM datasets").fetchall()
            matching_datasets = tuple(
                MigrationDataset(
                    id=_uuid(row["id"]),
                    root=Path(_string(row["root_uri"])).resolve(strict=False),
                )
                for row in dataset_rows
                if Path(_string(row["root_uri"])).resolve(strict=False)
                == configured_sessions_root.resolve(strict=False)
            )
            if len(matching_datasets) != 1:
                raise ValueError(
                    "Exactly one dataset must use the configured sessions root; "
                    f"found {len(matching_datasets)}."
                )
            dataset = matching_datasets[0]
            audio_hash_column_exists = connection.execute(
                """
                SELECT EXISTS (
                  SELECT 1
                  FROM information_schema.columns
                  WHERE table_schema = current_schema()
                    AND table_name = 'sample_tracks'
                    AND column_name = 'audio_sha256'
                ) AS exists
                """
            ).fetchone()
            assert audio_hash_column_exists is not None
            audio_hash_selection = (
                "sample_tracks.audio_sha256"
                if bool(audio_hash_column_exists["exists"])
                else "NULL::text AS audio_sha256"
            )
            sample_rows = connection.execute(
                f"""
                SELECT samples.id AS sample_id, samples.external_id,
                       sample_tracks.id AS track_id, sample_tracks.side,
                       sample_tracks.storage_uri, sample_tracks.access_uri,
                       {audio_hash_selection}
                FROM samples
                JOIN sample_tracks ON sample_tracks.sample_id = samples.id
                WHERE samples.dataset_id = %s
                ORDER BY samples.external_id, sample_tracks.speaker_index
                """,
                (dataset.id,),
            ).fetchall()
            samples = _sample_records(sample_rows)
            review_rows = connection.execute(
                """
                SELECT samples.external_id, reviews.speaker2_shift_seconds
                FROM synchronization_reviews AS reviews
                JOIN samples ON samples.id = reviews.sample_id
                WHERE samples.dataset_id = %s
                ORDER BY samples.external_id
                """,
                (dataset.id,),
            ).fetchall()
            transcript_rows = connection.execute(
                """
                SELECT transcripts.id, samples.external_id, tracks.id AS sample_track_id,
                       tracks.side, transcripts.model_id, transcripts.source_audio_sha256,
                       transcripts.prepared_audio_sha256,
                       transcripts.source_duration_seconds,
                       transcripts.prepared_duration_seconds, transcripts.words
                FROM full_recording_asr_transcripts AS transcripts
                JOIN sample_tracks AS tracks ON tracks.id = transcripts.sample_track_id
                JOIN samples ON samples.id = tracks.sample_id
                WHERE samples.dataset_id = %s
                ORDER BY samples.external_id, tracks.speaker_index, transcripts.model_id
                """,
                (dataset.id,),
            ).fetchall()
            cached_rows = connection.execute(
                "SELECT id, audio_filename FROM cached_asr_transcripts ORDER BY audio_filename, id"
            ).fetchall()
            legacy_row = connection.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM asr_runs)::integer AS asr_runs,
                  (SELECT COUNT(*) FROM asr_words)::integer AS asr_words,
                  (SELECT COUNT(*) FROM asr_evaluations)::integer AS asr_evaluations,
                  (SELECT COUNT(*) FROM annotation_runs)::integer AS annotation_runs,
                  (SELECT COUNT(*) FROM annotation_events)::integer AS annotation_events
                """
            ).fetchone()
            assert legacy_row is not None
            state_row = connection.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM quality_results
                   JOIN samples ON samples.id = quality_results.sample_id
                   WHERE samples.dataset_id = %s)::integer AS quality_result_count,
                  (SELECT COUNT(*) FROM synchronization_predictions
                   JOIN samples ON samples.id = synchronization_predictions.sample_id
                   WHERE samples.dataset_id = %s)::integer AS prediction_count
                """,
                (dataset.id, dataset.id),
            ).fetchone()
            assert state_row is not None
            current_quality_rows = connection.execute(
                """
                SELECT DISTINCT samples.external_id
                FROM quality_results
                JOIN samples ON samples.id = quality_results.sample_id
                WHERE samples.dataset_id = %s
                  AND quality_results.metric_version = %s
                  AND quality_results.status IN ('completed', 'invalid')
                """,
                (dataset.id, METRIC_VERSION),
            ).fetchall()
        return MigrationRepositorySnapshot(
            dataset=dataset,
            samples=samples,
            reviews=tuple(
                StoredAlignmentReview(
                    external_id=_string(row["external_id"]),
                    speaker2_shift_seconds=_float(row["speaker2_shift_seconds"]),
                )
                for row in review_rows
            ),
            transcripts=tuple(_transcript_record(row) for row in transcript_rows),
            cached_transcripts=tuple(
                CachedAsrMigrationRecord(
                    id=_uuid(row["id"]),
                    audio_filename=_string(row["audio_filename"]),
                )
                for row in cached_rows
            ),
            legacy_counts=LegacyTableCounts(
                asr_runs=_integer(legacy_row["asr_runs"]),
                asr_words=_integer(legacy_row["asr_words"]),
                asr_evaluations=_integer(legacy_row["asr_evaluations"]),
                annotation_runs=_integer(legacy_row["annotation_runs"]),
                annotation_events=_integer(legacy_row["annotation_events"]),
            ),
            quality_result_count=_integer(state_row["quality_result_count"]),
            current_quality_external_ids=frozenset(
                _string(row["external_id"]) for row in current_quality_rows
            ),
            synchronization_prediction_count=_integer(state_row["prediction_count"]),
        )

    def reconcile_sample(self, reconciliation: SampleReconciliation) -> None:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            locked_rows = connection.execute(
                """
                SELECT transcripts.id
                FROM full_recording_asr_transcripts AS transcripts
                JOIN sample_tracks ON sample_tracks.id = transcripts.sample_track_id
                WHERE sample_tracks.sample_id = %s
                FOR UPDATE
                """,
                (reconciliation.id,),
            ).fetchall()
            expected_ids = {
                transcript.id for track in reconciliation.tracks for transcript in track.transcripts
            }
            if {_uuid(row["id"]) for row in locked_rows} != expected_ids:
                raise ValueError("Full-recording ASR rows changed after migration preflight.")
            for track in reconciliation.tracks:
                for transcript in track.transcripts:
                    cursor = connection.execute(
                        """
                        UPDATE full_recording_asr_transcripts
                        SET source_audio_sha256 = %s,
                            prepared_audio_sha256 = %s,
                            source_duration_seconds = %s,
                            prepared_duration_seconds = %s,
                            words = %s,
                            updated_at = now()
                        WHERE id = %s AND sample_track_id = %s
                        """,
                        (
                            transcript.source_audio_sha256,
                            transcript.prepared_audio.sha256,
                            transcript.source_duration_seconds,
                            transcript.prepared_audio.duration_seconds,
                            Jsonb([word.model_dump(mode="json") for word in transcript.words]),
                            transcript.id,
                            track.id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("Failed to reconcile a full-recording ASR row.")
                track_cursor = connection.execute(
                    """
                    UPDATE sample_tracks
                    SET duration_seconds = %s, sample_rate = %s, channels = %s,
                        sample_count = %s, audio_sha256 = %s, updated_at = now()
                    WHERE id = %s AND sample_id = %s
                    """,
                    (
                        track.duration_seconds,
                        track.sample_rate,
                        track.channels,
                        track.sample_count,
                        track.audio_sha256,
                        track.id,
                        reconciliation.id,
                    ),
                )
                if track_cursor.rowcount != 1:
                    raise ValueError("Failed to reconcile a sample track.")
                metadata_cursor = connection.execute(
                    """
                    UPDATE audio_metadata
                    SET duration_seconds = %s, sample_rate = %s, channels = %s,
                        sample_count = %s,
                        payload = payload || %s,
                        updated_at = now()
                    WHERE sample_track_id = %s
                    """,
                    (
                        track.duration_seconds,
                        track.sample_rate,
                        track.channels,
                        track.sample_count,
                        Jsonb(
                            {
                                "duration_seconds": track.duration_seconds,
                                "sample_rate": track.sample_rate,
                                "channels": track.channels,
                                "sample_count": track.sample_count,
                            }
                        ),
                        track.id,
                    ),
                )
                if metadata_cursor.rowcount != 1:
                    raise ValueError("Failed to reconcile track audio metadata.")
            sample_cursor = connection.execute(
                """
                UPDATE samples
                SET duration_seconds = %s, updated_at = now()
                WHERE id = %s
                """,
                (reconciliation.duration_seconds, reconciliation.id),
            )
            if sample_cursor.rowcount != 1:
                raise ValueError("Failed to reconcile a sample.")

    def delete_invalid_samples(self, deletion: InvalidSampleDeletion) -> None:
        if not deletion.sample_ids:
            return
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            rows = connection.execute(
                """
                SELECT id, external_id
                FROM samples
                WHERE id = ANY(%s)
                FOR UPDATE
                """,
                (list(deletion.sample_ids),),
            ).fetchall()
            actual_ids = {_uuid(row["id"]) for row in rows}
            actual_external_ids = {_string(row["external_id"]) for row in rows}
            if actual_ids != set(deletion.sample_ids):
                raise ValueError("Invalid-sample rows changed after migration preflight.")
            if actual_external_ids != set(deletion.expected_external_ids):
                raise ValueError("Invalid-sample identities changed after migration preflight.")
            connection.execute(
                "DELETE FROM cached_asr_transcripts WHERE id = ANY(%s)",
                (list(deletion.cached_transcript_ids),),
            )
            deleted = connection.execute(
                "DELETE FROM samples WHERE id = ANY(%s)",
                (list(deletion.sample_ids),),
            )
            if deleted.rowcount != len(deletion.sample_ids):
                raise ValueError("Failed to delete the exact invalid-sample set.")

    def clean_derived_state(
        self,
        dataset_id: UUID,
        retained_sample_ids: tuple[UUID, ...],
        cached_transcript_ids: tuple[UUID, ...],
    ) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """
                DELETE FROM quality_results
                USING samples
                WHERE quality_results.sample_id = samples.id
                  AND samples.dataset_id = %s
                """,
                (dataset_id,),
            )
            connection.execute(
                """
                DELETE FROM synchronization_predictions
                USING samples
                WHERE synchronization_predictions.sample_id = samples.id
                  AND samples.dataset_id = %s
                """,
                (dataset_id,),
            )
            connection.execute(
                """
                DELETE FROM synchronization_reviews
                USING samples
                WHERE synchronization_reviews.sample_id = samples.id
                  AND samples.dataset_id = %s
                """,
                (dataset_id,),
            )
            connection.execute(
                """
                UPDATE samples
                SET quality_score = NULL, quality_flags = ARRAY[]::text[], updated_at = now()
                WHERE id = ANY(%s)
                """,
                (list(retained_sample_ids),),
            )
            if cached_transcript_ids:
                connection.execute(
                    "DELETE FROM cached_asr_transcripts WHERE id = ANY(%s)",
                    (list(cached_transcript_ids),),
                )

    def validate_track_hash_constraints(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                "ALTER TABLE sample_tracks VALIDATE CONSTRAINT sample_tracks_audio_sha256_format"
            )
            connection.execute(
                "ALTER TABLE sample_tracks VALIDATE CONSTRAINT sample_tracks_audio_sha256_required"
            )


def _sample_records(rows: list[dict[str, object]]) -> tuple[MigrationSample, ...]:
    records: list[MigrationSample] = []
    current_id: UUID | None = None
    current_external_id = ""
    tracks: list[MigrationTrack] = []
    for row in rows:
        sample_id = _uuid(row["sample_id"])
        if current_id is not None and sample_id != current_id:
            records.append(_sample_record(current_id, current_external_id, tracks))
            tracks = []
        current_id = sample_id
        current_external_id = _string(row["external_id"])
        tracks.append(
            MigrationTrack(
                id=_uuid(row["track_id"]),
                side=AlignmentSide(_string(row["side"])),
                storage_uri=_string(row["storage_uri"]),
                access_uri=Path(_string(row["access_uri"])).resolve(strict=False),
                audio_sha256=(
                    _string(row["audio_sha256"]) if row["audio_sha256"] is not None else None
                ),
            )
        )
    if current_id is not None:
        records.append(_sample_record(current_id, current_external_id, tracks))
    return tuple(records)


def _sample_record(
    sample_id: UUID,
    external_id: str,
    tracks: list[MigrationTrack],
) -> MigrationSample:
    if len(tracks) != 2:
        raise ValueError(f"Sample {external_id} does not have exactly two tracks.")
    ordered = tuple(sorted(tracks, key=lambda track: track.side.value))
    if tuple(track.side for track in ordered) != (
        AlignmentSide.SPEAKER1,
        AlignmentSide.SPEAKER2,
    ):
        raise ValueError(f"Sample {external_id} does not have the required track sides.")
    return MigrationSample(
        id=sample_id,
        external_id=external_id,
        tracks=(ordered[0], ordered[1]),
    )


def _transcript_record(row: dict[str, object]) -> FullAsrMigrationRecord:
    words_value = row["words"]
    if isinstance(words_value, str):
        words_value = json.loads(words_value)
    if not isinstance(words_value, list):
        raise ValueError("Full-recording ASR words must be a JSON array.")
    return FullAsrMigrationRecord(
        id=_uuid(row["id"]),
        sample_external_id=_string(row["external_id"]),
        sample_track_id=_uuid(row["sample_track_id"]),
        side=AlignmentSide(_string(row["side"])),
        model_id=AsrModelId(_string(row["model_id"])),
        source_audio_sha256=_string(row["source_audio_sha256"]),
        prepared_audio_sha256=_string(row["prepared_audio_sha256"]),
        source_duration_seconds=_float(row["source_duration_seconds"]),
        prepared_duration_seconds=_float(row["prepared_duration_seconds"]),
        words=tuple(TimestampedWord.model_validate(word) for word in words_value),
    )


def _uuid(value: object) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected a database text value.")
    return value


def _float(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("Expected a database numeric value.")
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("Expected a database integer value.")
    return value
