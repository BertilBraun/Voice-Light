from __future__ import annotations

import wave
from pathlib import Path
from uuid import UUID

import pytest

from app.local.alignment_migration.models import AlignmentSide
from app.local.alignment_migration.repository import (
    AlignmentMigrationRepository,
    CachedAsrMigrationRecord,
    FullAsrMigrationRecord,
    LegacyTableCounts,
    MigrationDataset,
    MigrationRepositorySnapshot,
    MigrationSample,
    MigrationTrack,
    StoredAlignmentReview,
)
from app.local.alignment_migration.riff import sha256_file
from app.local.alignment_migration.service import (
    EXPECTED_INVALID_TRANSCRIPT_COUNT,
    EXPECTED_RETAINED_TRANSCRIPT_COUNT,
    EXPECTED_SOURCE_SAMPLE_COUNT,
    INVALID_SAMPLE_IDS,
    AlignmentMigrationService,
    cached_asr_external_id,
    classify_cached_asr,
    invalid_directory_stage,
    shift_timestamped_words,
)
from app.local.synchronization_review.calibration import REVIEWED_ALIGNMENTS
from app.shared.asr import AsrModelId, TimestampedWord


class SnapshotOnlyRepository(AlignmentMigrationRepository):
    def __init__(self, snapshot: MigrationRepositorySnapshot) -> None:
        super().__init__(database_url="")
        self.snapshot = snapshot
        self.load_count = 0

    def load_snapshot(self, configured_sessions_root: Path) -> MigrationRepositorySnapshot:
        assert configured_sessions_root == self.snapshot.dataset.root
        self.load_count += 1
        return self.snapshot


def test_dry_run_validates_exact_snapshot_without_mutation(tmp_path: Path) -> None:
    snapshot = migration_snapshot(tmp_path)
    repository = SnapshotOnlyRepository(snapshot)
    service = AlignmentMigrationService(
        repository=repository,
        sessions_root=tmp_path,
    )

    summary = service.dry_run()

    assert repository.load_count == 1
    assert summary.source_sample_count == EXPECTED_SOURCE_SAMPLE_COUNT
    assert summary.retained_sample_count == 165
    assert summary.invalid_sample_count == 12
    assert summary.negative_offset_count == 97
    assert summary.positive_offset_count == 3
    assert summary.zero_offset_count == 65
    assert summary.rewritten_sample_count == 100
    assert summary.full_asr_transcript_count == 684
    assert summary.dataset_cached_asr_count == 684
    assert summary.unrelated_cached_asr_count == 6
    assert not tuple(tmp_path.rglob("*.alignment.json"))
    assert not tuple(tmp_path.rglob("*.alignment.staged"))


@pytest.mark.parametrize(
    ("filename", "expected"),
    (
        ("pmt_123_speaker1.flac", "pmt_123"),
        ("pmt_123_speaker2.flac", "pmt_123"),
        ("pmt_123_speaker1_asr_analysis_token-7.wav", "pmt_123"),
        ("smoke_speaker1.flac", None),
    ),
)
def test_cached_asr_external_id_accepts_only_exact_patterns(
    filename: str,
    expected: str | None,
) -> None:
    assert cached_asr_external_id(filename) == expected


def test_cache_classifier_rejects_unknown_pmt_prefixed_filename() -> None:
    record = CachedAsrMigrationRecord(
        id=UUID(int=1),
        audio_filename="pmt_123_speaker1.wav",
    )

    with pytest.raises(ValueError, match="Unrecognized"):
        classify_cached_asr((record,), {"pmt_123"})


def test_cache_classifier_deletes_by_typed_row_identity() -> None:
    dataset_record = CachedAsrMigrationRecord(
        id=UUID(int=1),
        audio_filename="pmt_123_speaker1.flac",
    )
    smoke_record = CachedAsrMigrationRecord(
        id=UUID(int=2),
        audio_filename="smoke_speaker1.flac",
    )

    classified = classify_cached_asr(
        (dataset_record, smoke_record),
        {"pmt_123"},
    )

    assert classified.dataset_records == (dataset_record,)
    assert classified.unrelated_records == (smoke_record,)


def test_timestamp_shift_preserves_transcript_payload() -> None:
    original = TimestampedWord(
        text="hello",
        start_seconds=0.25,
        end_seconds=0.5,
        confidence=0.75,
    )

    shifted = shift_timestamped_words(
        words=(original,),
        shift_seconds=1.5,
        duration_seconds=3.0,
    )

    assert shifted == (
        TimestampedWord(
            text="hello",
            start_seconds=1.75,
            end_seconds=2.0,
            confidence=0.75,
        ),
    )
    assert original.start_seconds == 0.25


def test_timestamp_shift_never_clips_out_of_bounds_words() -> None:
    word = TimestampedWord(text="late", start_seconds=0.8, end_seconds=1.0)

    with pytest.raises(ValueError, match="outside"):
        shift_timestamped_words((word,), shift_seconds=0.25, duration_seconds=1.0)


def test_invalid_stage_paths_are_contained_by_sessions_root(tmp_path: Path) -> None:
    stage = invalid_directory_stage(tmp_path, "pmt_201")

    assert stage.canonical.parent == tmp_path.resolve()
    assert stage.staged.parent == tmp_path.resolve()
    assert stage.staged.name == ".pmt_201.alignment-invalid.staged"


def test_invalid_stage_paths_reject_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        invalid_directory_stage(tmp_path, "../outside")


def migration_snapshot(root: Path) -> MigrationRepositorySnapshot:
    static_ids = [alignment.external_id for alignment in REVIEWED_ALIGNMENTS]
    generated_ids = [f"pmt_{number}" for number in range(400, 537)]
    retained_ids = static_ids + generated_ids
    assert len(retained_ids) == 165
    all_ids = retained_ids + sorted(INVALID_SAMPLE_IDS)
    samples: list[MigrationSample] = []
    transcripts: list[FullAsrMigrationRecord] = []
    reviews: list[StoredAlignmentReview] = [
        StoredAlignmentReview(
            external_id=alignment.external_id,
            speaker2_shift_seconds=alignment.speaker2_shift_seconds,
        )
        for alignment in REVIEWED_ALIGNMENTS
    ]
    extra_offsets = [-1.0] * 70 + [1.0] * 2 + [0.0] * 65
    reviews.extend(
        StoredAlignmentReview(external_id=external_id, speaker2_shift_seconds=offset)
        for external_id, offset in zip(generated_ids, extra_offsets, strict=True)
    )
    next_uuid = 1
    for external_id in all_ids:
        sample_directory = root / external_id
        sample_directory.mkdir()
        tracks: list[MigrationTrack] = []
        for side in (AlignmentSide.SPEAKER1, AlignmentSide.SPEAKER2):
            audio_path = sample_directory / f"{external_id}_{side.value}.wav"
            write_wave(audio_path)
            track = MigrationTrack(
                id=UUID(int=next_uuid),
                side=side,
                storage_uri=str(audio_path),
                access_uri=audio_path.resolve(),
                audio_sha256=None,
            )
            next_uuid += 1
            tracks.append(track)
            models = (
                (AsrModelId.PARAKEET_TDT,)
                if external_id in INVALID_SAMPLE_IDS
                else (AsrModelId.PARAKEET_TDT, AsrModelId.CANARY)
            )
            for model_id in models:
                transcripts.append(
                    FullAsrMigrationRecord(
                        id=UUID(int=next_uuid),
                        sample_external_id=external_id,
                        sample_track_id=track.id,
                        side=side,
                        model_id=model_id,
                        source_audio_sha256=sha256_file(audio_path),
                        prepared_audio_sha256="a" * 64,
                        source_duration_seconds=1.0,
                        prepared_duration_seconds=1.0,
                        words=(
                            TimestampedWord(
                                text="word",
                                start_seconds=0.1,
                                end_seconds=0.2,
                            ),
                        ),
                    )
                )
                next_uuid += 1
        samples.append(
            MigrationSample(
                id=UUID(int=next_uuid),
                external_id=external_id,
                tracks=(tracks[0], tracks[1]),
            )
        )
        next_uuid += 1
    assert len(transcripts) == (
        EXPECTED_RETAINED_TRANSCRIPT_COUNT + EXPECTED_INVALID_TRANSCRIPT_COUNT
    )
    cached = tuple(
        CachedAsrMigrationRecord(
            id=UUID(int=next_uuid + index),
            audio_filename=(f"{record.sample_external_id}_{record.side.value}.flac"),
        )
        for index, record in enumerate(transcripts)
    )
    next_uuid += len(cached)
    unrelated = tuple(
        CachedAsrMigrationRecord(
            id=UUID(int=next_uuid + index),
            audio_filename=f"smoke_{index}.flac",
        )
        for index in range(6)
    )
    return MigrationRepositorySnapshot(
        dataset=MigrationDataset(id=UUID(int=next_uuid + 10), root=root.resolve()),
        samples=tuple(samples),
        reviews=tuple(reviews),
        transcripts=tuple(transcripts),
        cached_transcripts=(*cached, *unrelated),
        legacy_counts=LegacyTableCounts(
            asr_runs=0,
            asr_words=0,
            asr_evaluations=0,
            annotation_runs=0,
            annotation_events=0,
        ),
        quality_result_count=10,
        current_quality_external_ids=frozenset(),
        synchronization_prediction_count=10,
    )


def write_wave(path: Path) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\0\0" * 8_000)
