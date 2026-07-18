from __future__ import annotations

import wave
from dataclasses import replace
from pathlib import Path
from shutil import rmtree
from uuid import UUID

import pytest

from app.local.alignment_migration import service as migration_service_module
from app.local.alignment_migration.filesystem_transaction import (
    AlignmentTransactionPaths,
    apply_alignment_transaction,
    confirm_database_reconciliation,
)
from app.local.alignment_migration.models import AlignmentSide
from app.local.alignment_migration.repository import (
    AlignmentMigrationRepository,
    CachedAsrMigrationRecord,
    FullAsrMigrationRecord,
    LegacyTableCounts,
    MigrationAudioMetadata,
    MigrationDataset,
    MigrationRepositorySnapshot,
    MigrationSample,
    MigrationTrack,
    StoredAlignmentReview,
    TrackHashConstraintState,
)
from app.local.alignment_migration.riff import sha256_file
from app.local.alignment_migration.service import (
    EXPECTED_INVALID_TRANSCRIPT_COUNT,
    EXPECTED_RETAINED_TRANSCRIPT_COUNT,
    EXPECTED_SOURCE_SAMPLE_COUNT,
    INVALID_SAMPLE_IDS,
    AlignmentMigrationService,
    SampleMigrationPlan,
    _audit_post_migration_sources,
    _validate_word_timestamps,
    cached_asr_external_id,
    classify_cached_asr,
    invalid_directory_stage,
    shift_timestamped_words,
)
from app.local.asr.full_recording_service import PreparedFullRecordingAudio
from app.local.synchronization_review.calibration import REVIEWED_ALIGNMENTS
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.audio.transport import AudioTransportCodec, AudioTransportMetadata
from app.shared.quality import AudioMetadata


class SnapshotOnlyRepository(AlignmentMigrationRepository):
    def __init__(self, snapshot: MigrationRepositorySnapshot) -> None:
        super().__init__(database_url="")
        self.snapshot = snapshot
        self.load_count = 0

    def load_snapshot(self, configured_sessions_root: Path) -> MigrationRepositorySnapshot:
        assert configured_sessions_root == self.snapshot.dataset.root
        self.load_count += 1
        return self.snapshot


class RecordingPreparedAudioFactory:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(
        self,
        source_path: Path,
        source_audio_sha256: str | None = None,
    ) -> PreparedFullRecordingAudio:
        self.calls.append(source_path)
        prepared_path = source_path.with_name(f"{source_path.stem}.prepared-{len(self.calls)}.ogg")
        prepared_path.write_bytes(b"prepared")
        prepared_hash = f"{len(self.calls):064x}"
        duration_seconds = 1.001
        return PreparedFullRecordingAudio(
            path=prepared_path,
            transport_metadata=AudioTransportMetadata(
                original_filename=source_path.name,
                encoded_filename=prepared_path.name,
                sha256=prepared_hash,
                codec=AudioTransportCodec.OGG_OPUS,
                duration_seconds=duration_seconds,
                sample_rate=16_000,
                channels=1,
                size_bytes=8,
            ),
            source_audio_sha256=source_audio_sha256 or sha256_file(source_path),
            prepared_audio_sha256=prepared_hash,
            source_duration_seconds=duration_seconds,
            prepared_duration_seconds=duration_seconds,
        )


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


def test_retry_preflight_allows_684_cache_rows_after_invalid_database_deletion(
    tmp_path: Path,
) -> None:
    initial = migration_snapshot(tmp_path)
    retry = replace(
        initial,
        samples=tuple(
            sample for sample in initial.samples if sample.external_id not in INVALID_SAMPLE_IDS
        ),
        transcripts=tuple(
            record
            for record in initial.transcripts
            if record.sample_external_id not in INVALID_SAMPLE_IDS
        ),
    )
    service = AlignmentMigrationService(
        repository=SnapshotOnlyRepository(retry),
        sessions_root=tmp_path,
    )

    summary = service.dry_run()

    assert summary.retained_sample_count == 165
    assert summary.full_asr_transcript_count == 660
    assert summary.dataset_cached_asr_count == 684


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


def test_timestamp_validation_allows_local_backward_chunk_overlap() -> None:
    record = FullAsrMigrationRecord(
        id=UUID(int=1),
        sample_external_id="pmt_001",
        sample_track_id=UUID(int=2),
        side=AlignmentSide.SPEAKER1,
        model_id=AsrModelId.PARAKEET_TDT,
        source_audio_sha256="a" * 64,
        prepared_audio_sha256="b" * 64,
        source_duration_seconds=1.0,
        prepared_duration_seconds=1.0,
        words=(
            TimestampedWord(text="chunk", start_seconds=0.20, end_seconds=0.30),
            TimestampedWord(text="overlap", start_seconds=0.16, end_seconds=0.27),
        ),
    )

    _validate_word_timestamps(record)

    shifted = shift_timestamped_words(
        record.words,
        shift_seconds=0.5,
        duration_seconds=1.5,
    )
    assert shifted[0].start_seconds == pytest.approx(0.70)
    assert shifted[1].start_seconds == pytest.approx(0.66)


def test_invalid_stage_paths_are_contained_by_sessions_root(tmp_path: Path) -> None:
    stage = invalid_directory_stage(tmp_path, "pmt_201")

    assert stage.canonical.parent == tmp_path.resolve()
    assert stage.staged.parent == tmp_path.resolve()
    assert stage.staged.name == ".pmt_201.alignment-invalid.staged"


def test_invalid_stage_paths_reject_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        invalid_directory_stage(tmp_path, "../outside")


def test_sample_reconciliation_reads_real_riff_channel_count(tmp_path: Path) -> None:
    sample_directory = tmp_path / "pmt_999"
    sample_directory.mkdir()
    speaker1_path = sample_directory / "pmt_999_speaker1.wav"
    speaker2_path = sample_directory / "pmt_999_speaker2.wav"
    write_wave(speaker1_path, channels=2)
    write_wave(speaker2_path, channels=2)
    tracks = (
        migration_track(UUID(int=1), AlignmentSide.SPEAKER1, speaker1_path, channels=2),
        migration_track(UUID(int=2), AlignmentSide.SPEAKER2, speaker2_path, channels=2),
    )
    sample = MigrationSample(
        id=UUID(int=3),
        external_id="pmt_999",
        duration_seconds=1.0,
        tracks=tracks,
    )
    transcripts = tuple(
        FullAsrMigrationRecord(
            id=UUID(int=10 + index),
            sample_external_id=sample.external_id,
            sample_track_id=track.id,
            side=track.side,
            model_id=model_id,
            source_audio_sha256=sha256_file(track.access_uri),
            prepared_audio_sha256=f"{index + 1:064x}",
            source_duration_seconds=1.0,
            prepared_duration_seconds=1.0 + (index * 0.001),
            words=(TimestampedWord(text="word", start_seconds=0.1, end_seconds=0.2),),
        )
        for index, (track, model_id) in enumerate(
            (
                (tracks[0], AsrModelId.PARAKEET_TDT),
                (tracks[0], AsrModelId.CANARY),
                (tracks[1], AsrModelId.PARAKEET_TDT),
                (tracks[1], AsrModelId.CANARY),
            )
        )
    )
    assert len(transcripts) == 4
    sidecar = apply_alignment_transaction(
        paths=AlignmentTransactionPaths.create(
            sample_directory=sample_directory,
            sample_external_id=sample.external_id,
            speaker1_filename=speaker1_path.name,
            speaker2_filename=speaker2_path.name,
        ),
        reviewed_speaker2_shift_seconds=0.0,
    ).sidecar
    service = AlignmentMigrationService(
        repository=AlignmentMigrationRepository(database_url=""),
        sessions_root=tmp_path,
    )

    reconciliation = service._sample_reconciliation(
        plan=SampleMigrationPlan(
            sample=sample,
            shift_seconds=0.0,
            transcripts=(transcripts[0], transcripts[1], transcripts[2], transcripts[3]),
        ),
        sidecar=sidecar,
    )

    assert tuple(track.channels for track in reconciliation.tracks) == (2, 2)
    assert tuple(track.sample_count for track in reconciliation.tracks) == (8_000, 8_000)
    reconciled_prepared = {
        transcript.id: (
            transcript.prepared_audio.sha256,
            transcript.prepared_audio.duration_seconds,
        )
        for track in reconciliation.tracks
        for transcript in track.transcripts
    }
    assert reconciled_prepared == {
        record.id: (
            record.prepared_audio_sha256,
            record.prepared_duration_seconds,
        )
        for record in transcripts
    }
    assert {
        transcript.id: transcript.words
        for track in reconciliation.tracks
        for transcript in track.transcripts
    } == {record.id: record.words for record in transcripts}


def test_changed_tracks_reuse_one_prepared_provenance_per_model_pair(
    tmp_path: Path,
) -> None:
    sample_directory = tmp_path / "pmt_998"
    sample_directory.mkdir()
    speaker1_path = sample_directory / "pmt_998_speaker1.wav"
    speaker2_path = sample_directory / "pmt_998_speaker2.wav"
    write_wave(speaker1_path)
    write_wave(speaker2_path)
    tracks = (
        migration_track(UUID(int=1), AlignmentSide.SPEAKER1, speaker1_path, channels=1),
        migration_track(UUID(int=2), AlignmentSide.SPEAKER2, speaker2_path, channels=1),
    )
    original_hashes = {track.id: sha256_file(track.access_uri) for track in tracks}
    sample = MigrationSample(
        id=UUID(int=3),
        external_id="pmt_998",
        duration_seconds=1.0,
        tracks=tracks,
    )
    transcripts = tuple(
        FullAsrMigrationRecord(
            id=UUID(int=20 + index),
            sample_external_id=sample.external_id,
            sample_track_id=track.id,
            side=track.side,
            model_id=model_id,
            source_audio_sha256=original_hashes[track.id],
            prepared_audio_sha256=f"{index + 10:064x}",
            source_duration_seconds=1.0,
            prepared_duration_seconds=1.0 + (index * 0.001),
            words=(TimestampedWord(text="word", start_seconds=0.1, end_seconds=0.2),),
        )
        for index, (track, model_id) in enumerate(
            (
                (tracks[0], AsrModelId.PARAKEET_TDT),
                (tracks[0], AsrModelId.CANARY),
                (tracks[1], AsrModelId.PARAKEET_TDT),
                (tracks[1], AsrModelId.CANARY),
            )
        )
    )
    assert len(transcripts) == 4
    sidecar = apply_alignment_transaction(
        paths=AlignmentTransactionPaths.create(
            sample_directory=sample_directory,
            sample_external_id=sample.external_id,
            speaker1_filename=speaker1_path.name,
            speaker2_filename=speaker2_path.name,
        ),
        reviewed_speaker2_shift_seconds=-0.001,
    ).sidecar
    prepared_factory = RecordingPreparedAudioFactory()
    service = AlignmentMigrationService(
        repository=AlignmentMigrationRepository(database_url=""),
        sessions_root=tmp_path,
        prepared_audio_factory=prepared_factory,
    )

    reconciliation = service._sample_reconciliation(
        plan=SampleMigrationPlan(
            sample=sample,
            shift_seconds=-0.001,
            transcripts=(transcripts[0], transcripts[1], transcripts[2], transcripts[3]),
        ),
        sidecar=sidecar,
    )

    assert prepared_factory.calls == [speaker1_path, speaker2_path]
    prepared_by_track = tuple(
        {
            (
                transcript.prepared_audio.sha256,
                transcript.prepared_audio.duration_seconds,
            )
            for transcript in track.transcripts
        }
        for track in reconciliation.tracks
    )
    assert all(len(values) == 1 for values in prepared_by_track)
    assert prepared_by_track[0] != prepared_by_track[1]
    assert not tuple(tmp_path.rglob("*.prepared-*.ogg"))


def test_post_migration_source_audit_checks_exact_filesystem_and_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = post_migration_snapshot(tmp_path)
    repository = SnapshotOnlyRepository(snapshot)
    service = AlignmentMigrationService(
        repository=repository,
        sessions_root=tmp_path,
    )
    plan = service.preflight()

    _audit_post_migration_sources(plan=plan, sessions_root=tmp_path)

    first = plan.retained[0]
    bad_track = replace(first.sample.tracks[0], channels=2)
    bad_sample = replace(
        first.sample,
        tracks=(bad_track, first.sample.tracks[1]),
    )
    bad_plan = replace(
        plan,
        retained=(replace(first, sample=bad_sample), *plan.retained[1:]),
    )
    with pytest.raises(ValueError, match="sample-track metadata"):
        _audit_post_migration_sources(plan=bad_plan, sessions_root=tmp_path)

    unvalidated_plan = replace(
        plan,
        snapshot=replace(
            plan.snapshot,
            track_hash_constraints=TrackHashConstraintState(
                format_validated=True,
                required_validated=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="constraints"):
        _audit_post_migration_sources(
            plan=unvalidated_plan,
            sessions_root=tmp_path,
        )

    retained_ids = {sample.external_id for sample in snapshot.samples}
    initially_complete = set(snapshot.current_quality_external_ids)

    class FakeQualityRepository:
        load_count = 0

        def completed_quality_sample_ids(
            self,
            dataset_id: UUID,
            metric_version: str,
        ) -> set[str]:
            self.load_count += 1
            return initially_complete if self.load_count == 1 else retained_ids

    fake_quality_repository = FakeQualityRepository()
    monkeypatch.setattr(
        migration_service_module,
        "Repository",
        lambda database_url: fake_quality_repository,
    )
    monkeypatch.setattr(
        migration_service_module,
        "FullRecordingAsrRepository",
        lambda database_url: "existing-asr",
    )
    monkeypatch.setattr(
        migration_service_module,
        "register_sample",
        lambda repository, dataset_id, storage, discovered: discovered,
    )
    monkeypatch.setattr(
        migration_service_module,
        "ready_sample",
        lambda full_asr_repository, registered: registered,
    )
    monkeypatch.setattr(
        migration_service_module,
        "process_ready_sample",
        lambda storage, ready: ready,
    )
    monkeypatch.setattr(
        migration_service_module,
        "persist_processed_ready_sample",
        lambda repository, processed: None,
    )

    summary = service.regenerate_quality()

    assert summary.already_current_count == 1
    assert summary.regenerated_count == 164


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
                duration_seconds=1.0,
                sample_rate=8_000,
                channels=1,
                sample_count=8_000,
                audio_sha256=None,
                audio_metadata=MigrationAudioMetadata(
                    duration_seconds=1.0,
                    sample_rate=8_000,
                    channels=1,
                    sample_count=8_000,
                    payload=AudioMetadata(
                        duration_seconds=1.0,
                        sample_rate=8_000,
                        channels=1,
                        sample_count=8_000,
                    ),
                ),
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
                duration_seconds=1.0,
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
        track_hash_constraints=TrackHashConstraintState(
            format_validated=False,
            required_validated=False,
        ),
    )


def post_migration_snapshot(root: Path) -> MigrationRepositorySnapshot:
    initial = migration_snapshot(root)
    offsets = {review.external_id: review.speaker2_shift_seconds for review in initial.reviews}
    records_by_sample = {
        sample.external_id: tuple(
            record
            for record in initial.transcripts
            if record.sample_external_id == sample.external_id
        )
        for sample in initial.samples
    }
    samples: list[MigrationSample] = []
    transcripts: list[FullAsrMigrationRecord] = []
    for sample in initial.samples:
        if sample.external_id in INVALID_SAMPLE_IDS:
            rmtree(root / sample.external_id)
            continue
        paths = AlignmentTransactionPaths.create(
            sample_directory=sample.tracks[0].access_uri.parent,
            sample_external_id=sample.external_id,
            speaker1_filename=sample.tracks[0].access_uri.name,
            speaker2_filename=sample.tracks[1].access_uri.name,
        )
        result = apply_alignment_transaction(
            paths=paths,
            reviewed_speaker2_shift_seconds=offsets[sample.external_id],
        )
        confirm_database_reconciliation(paths=paths, expected_sidecar=result.sidecar)
        (paths.final_sidecar.parent / f"{sample.external_id}.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        rewritten_tracks: list[MigrationTrack] = []
        sample_duration = 0.0
        for track, rewrite in (
            (sample.tracks[0], result.sidecar.speaker1),
            (sample.tracks[1], result.sidecar.speaker2),
        ):
            duration = rewrite.aligned_frame_count / result.sidecar.sample_rate
            payload = AudioMetadata(
                duration_seconds=duration,
                sample_rate=result.sidecar.sample_rate,
                channels=1,
                sample_count=rewrite.aligned_frame_count,
            )
            rewritten_tracks.append(
                replace(
                    track,
                    duration_seconds=duration,
                    sample_rate=result.sidecar.sample_rate,
                    channels=1,
                    sample_count=rewrite.aligned_frame_count,
                    audio_sha256=rewrite.aligned_sha256,
                    audio_metadata=MigrationAudioMetadata(
                        duration_seconds=duration,
                        sample_rate=result.sidecar.sample_rate,
                        channels=1,
                        sample_count=rewrite.aligned_frame_count,
                        payload=payload,
                    ),
                )
            )
            shift_seconds = rewrite.prepended_silence_frame_count / result.sidecar.sample_rate
            for record in records_by_sample[sample.external_id]:
                if record.sample_track_id != track.id:
                    continue
                transcripts.append(
                    replace(
                        record,
                        source_audio_sha256=rewrite.aligned_sha256,
                        source_duration_seconds=duration,
                        prepared_audio_sha256=(
                            f"{record.id.int:064x}"
                            if rewrite.original_sha256 == rewrite.aligned_sha256
                            else record.prepared_audio_sha256
                        ),
                        prepared_duration_seconds=(
                            record.prepared_duration_seconds
                            + (
                                0.001
                                if rewrite.original_sha256 == rewrite.aligned_sha256
                                and record.model_id is AsrModelId.CANARY
                                else 0.0
                            )
                        ),
                        words=shift_timestamped_words(
                            record.words,
                            shift_seconds=shift_seconds,
                            duration_seconds=duration,
                        ),
                    )
                )
            sample_duration = max(sample_duration, duration)
        samples.append(
            replace(
                sample,
                duration_seconds=sample_duration,
                tracks=(rewritten_tracks[0], rewritten_tracks[1]),
            )
        )
    unrelated_cache = tuple(
        record
        for record in initial.cached_transcripts
        if cached_asr_external_id(record.audio_filename) is None
    )
    first_external_id = samples[0].external_id
    return replace(
        initial,
        samples=tuple(samples),
        reviews=(),
        transcripts=tuple(transcripts),
        cached_transcripts=unrelated_cache,
        quality_result_count=1,
        current_quality_external_ids=frozenset({first_external_id}),
        synchronization_prediction_count=0,
        track_hash_constraints=TrackHashConstraintState(
            format_validated=True,
            required_validated=True,
        ),
    )


def migration_track(
    identifier: UUID,
    side: AlignmentSide,
    path: Path,
    channels: int,
) -> MigrationTrack:
    payload = AudioMetadata(
        duration_seconds=1.0,
        sample_rate=8_000,
        channels=channels,
        sample_count=8_000,
    )
    return MigrationTrack(
        id=identifier,
        side=side,
        storage_uri=str(path),
        access_uri=path.resolve(),
        duration_seconds=1.0,
        sample_rate=8_000,
        channels=channels,
        sample_count=8_000,
        audio_sha256=None,
        audio_metadata=MigrationAudioMetadata(
            duration_seconds=1.0,
            sample_rate=8_000,
            channels=channels,
            sample_count=8_000,
            payload=payload,
        ),
    )


def write_wave(path: Path, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\0\0" * 8_000 * channels)
