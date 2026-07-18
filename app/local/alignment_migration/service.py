from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.local.alignment_migration.filesystem_transaction import (
    AlignmentTransactionPaths,
    apply_alignment_transaction,
    confirm_database_reconciliation,
)
from app.local.alignment_migration.models import (
    AlignmentApplicationStatus,
    AlignmentSide,
    AlignmentSidecar,
    MigrationApplySummary,
    MigrationAuditSummary,
    MigrationPreflightSummary,
    MigrationQualitySummary,
)
from app.local.alignment_migration.repository import (
    AlignmentMigrationRepository,
    CachedAsrMigrationRecord,
    FullAsrMigrationRecord,
    InvalidSampleDeletion,
    MigrationRepositorySnapshot,
    MigrationSample,
    PreparedAudioProvenance,
    SampleReconciliation,
    TrackReconciliation,
    TranscriptReconciliation,
)
from app.local.alignment_migration.riff import inspect_pcm_wave, sha256_file
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.asr.full_recording_service import (
    PreparedFullRecordingAudio,
    prepare_full_recording_audio,
)
from app.local.db.repository import Repository
from app.local.ingestion.discovery import DiscoveredSample
from app.local.ingestion.service import (
    persist_processed_ready_sample,
    process_ready_sample,
    ready_sample,
    register_sample,
)
from app.local.synchronization_review.calibration import REVIEWED_ALIGNMENTS
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.quality import METRIC_VERSION, AudioMetadata
from app.shared.storage.local import LocalStorageBackend

INVALID_SAMPLE_IDS = frozenset(
    {
        "pmt_201",
        "pmt_204",
        "pmt_211",
        "pmt_234",
        "pmt_252",
        "pmt_265",
        "pmt_269",
        "pmt_274",
        "pmt_277",
        "pmt_280",
        "pmt_287",
        "pmt_323",
    }
)
EXPECTED_SOURCE_SAMPLE_COUNT = 177
EXPECTED_RETAINED_SAMPLE_COUNT = 165
EXPECTED_NEGATIVE_OFFSET_COUNT = 97
EXPECTED_POSITIVE_OFFSET_COUNT = 3
EXPECTED_ZERO_OFFSET_COUNT = 65
EXPECTED_REWRITTEN_SAMPLE_COUNT = 100
EXPECTED_RETAINED_TRANSCRIPT_COUNT = 660
EXPECTED_INVALID_TRANSCRIPT_COUNT = 24
EXPECTED_UNRELATED_CACHE_COUNT = 6
MAXIMUM_OFFSET_SECONDS = 12.0

_PRIMARY_CACHE_PATTERN = re.compile(r"^(pmt_\d+)_speaker[12]\.flac$")
_ANALYSIS_CACHE_PATTERN = re.compile(r"^(pmt_\d+)_speaker[12]_asr_analysis_[A-Za-z0-9_-]+\.wav$")


class PreparedAudioFactory(Protocol):
    def __call__(
        self,
        source_path: Path,
        source_audio_sha256: str | None = None,
    ) -> PreparedFullRecordingAudio: ...


@dataclass(frozen=True)
class CachedAsrClassification:
    dataset_records: tuple[CachedAsrMigrationRecord, ...]
    unrelated_records: tuple[CachedAsrMigrationRecord, ...]


@dataclass(frozen=True)
class SampleMigrationPlan:
    sample: MigrationSample
    shift_seconds: float
    transcripts: tuple[
        FullAsrMigrationRecord,
        FullAsrMigrationRecord,
        FullAsrMigrationRecord,
        FullAsrMigrationRecord,
    ]


@dataclass(frozen=True)
class AlignmentMigrationPlan:
    snapshot: MigrationRepositorySnapshot
    retained: tuple[SampleMigrationPlan, ...]
    invalid_samples: tuple[MigrationSample, ...]
    cached_asr: CachedAsrClassification

    def summary(self) -> MigrationPreflightSummary:
        negative = sum(sample.shift_seconds < 0.0 for sample in self.retained)
        positive = sum(sample.shift_seconds > 0.0 for sample in self.retained)
        zero = sum(sample.shift_seconds == 0.0 for sample in self.retained)
        return MigrationPreflightSummary(
            dataset_id=str(self.snapshot.dataset.id),
            source_sample_count=EXPECTED_SOURCE_SAMPLE_COUNT,
            retained_sample_count=len(self.retained),
            invalid_sample_count=len(INVALID_SAMPLE_IDS),
            negative_offset_count=negative,
            positive_offset_count=positive,
            zero_offset_count=zero,
            rewritten_sample_count=negative + positive,
            full_asr_transcript_count=len(self.snapshot.transcripts),
            dataset_cached_asr_count=len(self.cached_asr.dataset_records),
            unrelated_cached_asr_count=len(self.cached_asr.unrelated_records),
        )


@dataclass(frozen=True)
class InvalidDirectoryStage:
    external_id: str
    canonical: Path
    staged: Path


class AlignmentMigrationService:
    def __init__(
        self,
        repository: AlignmentMigrationRepository,
        sessions_root: Path,
        prepared_audio_factory: PreparedAudioFactory = prepare_full_recording_audio,
    ) -> None:
        self.repository = repository
        self.sessions_root = sessions_root.resolve(strict=False)
        self.prepared_audio_factory = prepared_audio_factory

    def dry_run(self) -> MigrationPreflightSummary:
        return self.preflight().summary()

    def preflight(self) -> AlignmentMigrationPlan:
        snapshot = self.repository.load_snapshot(self.sessions_root)
        sample_by_external_id = {sample.external_id: sample for sample in snapshot.samples}
        actual_sample_ids = set(sample_by_external_id)
        invalid_present = actual_sample_ids & INVALID_SAMPLE_IDS
        if invalid_present not in (set(), set(INVALID_SAMPLE_IDS)):
            raise ValueError("The invalid sample set is only partially present in PostgreSQL.")
        expected_count = (
            EXPECTED_SOURCE_SAMPLE_COUNT if invalid_present else EXPECTED_RETAINED_SAMPLE_COUNT
        )
        if len(snapshot.samples) != expected_count:
            raise ValueError(
                f"Expected {expected_count} PostgreSQL samples, found {len(snapshot.samples)}."
            )
        retained_ids = actual_sample_ids - INVALID_SAMPLE_IDS
        if len(retained_ids) != EXPECTED_RETAINED_SAMPLE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_RETAINED_SAMPLE_COUNT} retained samples, "
                f"found {len(retained_ids)}."
            )
        _validate_legacy_tables(snapshot)
        offsets = _reconciled_offsets(snapshot=snapshot, retained_ids=retained_ids)
        transcripts_by_sample = _validate_transcript_coverage(
            snapshot=snapshot,
            retained_ids=retained_ids,
            invalid_present=invalid_present,
        )
        cache = classify_cached_asr(
            records=snapshot.cached_transcripts,
            known_dataset_ids=actual_sample_ids | INVALID_SAMPLE_IDS,
        )
        expected_dataset_cache_counts = (
            {EXPECTED_RETAINED_TRANSCRIPT_COUNT + EXPECTED_INVALID_TRANSCRIPT_COUNT}
            if invalid_present
            else {
                0,
                EXPECTED_RETAINED_TRANSCRIPT_COUNT,
                EXPECTED_RETAINED_TRANSCRIPT_COUNT + EXPECTED_INVALID_TRANSCRIPT_COUNT,
            }
        )
        if len(cache.dataset_records) not in expected_dataset_cache_counts:
            raise ValueError(
                "Dataset cached ASR coverage is not an authorized initial or retry state."
            )
        if len(cache.unrelated_records) != EXPECTED_UNRELATED_CACHE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_UNRELATED_CACHE_COUNT} unrelated cached ASR rows, "
                f"found {len(cache.unrelated_records)}."
            )
        retained_plans: list[SampleMigrationPlan] = []
        for external_id in sorted(retained_ids):
            sample = sample_by_external_id[external_id]
            _validate_sample_files(sample=sample, sessions_root=self.sessions_root)
            retained_plans.append(
                SampleMigrationPlan(
                    sample=sample,
                    shift_seconds=offsets[external_id],
                    transcripts=_four_transcripts(transcripts_by_sample[external_id]),
                )
            )
        invalid_samples = tuple(
            sample_by_external_id[external_id] for external_id in sorted(invalid_present)
        )
        for sample in invalid_samples:
            _validate_invalid_sample_files(sample=sample, sessions_root=self.sessions_root)
        plan = AlignmentMigrationPlan(
            snapshot=snapshot,
            retained=tuple(retained_plans),
            invalid_samples=invalid_samples,
            cached_asr=cache,
        )
        _validate_expected_summary(plan.summary())
        return plan

    def apply(self) -> MigrationApplySummary:
        self._recover_invalid_directory_state()
        plan = self.preflight()
        newly_applied_count = 0
        already_applied_count = 0
        for sample_plan in plan.retained:
            paths = _transaction_paths(sample_plan.sample)
            result = apply_alignment_transaction(
                paths=paths,
                reviewed_speaker2_shift_seconds=sample_plan.shift_seconds,
            )
            if not _database_is_reconciled(sample_plan, result.sidecar):
                reconciliation = self._sample_reconciliation(
                    plan=sample_plan,
                    sidecar=result.sidecar,
                )
                self.repository.reconcile_sample(reconciliation)
            confirm_database_reconciliation(paths=paths, expected_sidecar=result.sidecar)
            if result.status is AlignmentApplicationStatus.ALREADY_APPLIED:
                already_applied_count += 1
            else:
                newly_applied_count += 1
        self._delete_invalid_samples(plan)
        retained_ids = tuple(sample.sample.id for sample in plan.retained)
        cached_ids = tuple(record.id for record in plan.cached_asr.dataset_records)
        if (
            cached_ids
            or plan.snapshot.reviews
            or plan.snapshot.synchronization_prediction_count
            or plan.invalid_samples
        ):
            self.repository.clean_derived_state(
                dataset_id=plan.snapshot.dataset.id,
                retained_sample_ids=retained_ids,
                cached_transcript_ids=cached_ids,
            )
        self.repository.validate_track_hash_constraints()
        self.audit()
        return MigrationApplySummary(
            retained_sample_count=len(plan.retained),
            newly_applied_sample_count=newly_applied_count,
            already_applied_sample_count=already_applied_count,
            invalid_sample_count=len(INVALID_SAMPLE_IDS),
            deleted_cached_asr_count=len(cached_ids),
        )

    def audit(self) -> MigrationAuditSummary:
        plan = self.preflight()
        _audit_post_migration_sources(plan=plan, sessions_root=self.sessions_root)
        if plan.cached_asr.dataset_records:
            raise ValueError("Post-migration audit found dataset cached ASR rows.")
        retained_ids = {sample.sample.external_id for sample in plan.retained}
        quality_is_empty = (
            plan.snapshot.quality_result_count == 0
            and not plan.snapshot.current_quality_external_ids
        )
        quality_is_complete = (
            plan.snapshot.quality_result_count == EXPECTED_RETAINED_SAMPLE_COUNT
            and plan.snapshot.current_quality_external_ids == retained_ids
        )
        if not quality_is_empty and not quality_is_complete:
            raise ValueError("Post-migration quality coverage is neither empty nor exact.")
        if plan.snapshot.reviews:
            raise ValueError("Post-migration audit found stale synchronization reviews.")
        if plan.snapshot.synchronization_prediction_count:
            raise ValueError("Post-migration audit found synchronization predictions.")
        return MigrationAuditSummary(
            retained_sample_count=len(plan.retained),
            sidecar_count=len(plan.retained),
            full_asr_transcript_count=len(plan.snapshot.transcripts),
            quality_result_count=plan.snapshot.quality_result_count,
            dataset_cached_asr_count=len(plan.cached_asr.dataset_records),
        )

    def regenerate_quality(self) -> MigrationQualitySummary:
        plan = self.preflight()
        _audit_post_migration_sources(plan=plan, sessions_root=self.sessions_root)
        if plan.cached_asr.dataset_records:
            raise ValueError("Quality regeneration requires the stale ASR cache to be deleted.")
        repository = Repository(self.repository.database_url)
        full_asr_repository = FullRecordingAsrRepository(self.repository.database_url)
        storage = LocalStorageBackend()
        completed = repository.completed_quality_sample_ids(
            dataset_id=plan.snapshot.dataset.id,
            metric_version=METRIC_VERSION,
        )
        retained_ids = {sample.sample.external_id for sample in plan.retained}
        if not completed <= retained_ids:
            raise ValueError("Current quality rows reference samples outside the retained set.")
        if (
            plan.snapshot.current_quality_external_ids != completed
            or plan.snapshot.quality_result_count != len(completed)
        ):
            raise ValueError("Partial quality retry state is not exact and current.")
        regenerated_count = 0
        for sample_plan in plan.retained:
            if sample_plan.sample.external_id in completed:
                continue
            discovered = DiscoveredSample(
                external_id=sample_plan.sample.external_id,
                speaker1_path=str(sample_plan.sample.tracks[0].access_uri),
                speaker2_path=str(sample_plan.sample.tracks[1].access_uri),
            )
            registered = register_sample(
                repository=repository,
                dataset_id=plan.snapshot.dataset.id,
                storage=storage,
                discovered=discovered,
            )
            ready = ready_sample(
                full_asr_repository=full_asr_repository,
                registered=registered,
            )
            processed = process_ready_sample(storage=storage, ready=ready)
            persist_processed_ready_sample(repository=repository, processed=processed)
            regenerated_count += 1
        final_completed = repository.completed_quality_sample_ids(
            dataset_id=plan.snapshot.dataset.id,
            metric_version=METRIC_VERSION,
        )
        if final_completed != retained_ids:
            raise ValueError("Quality regeneration did not produce exact retained coverage.")
        return MigrationQualitySummary(
            retained_sample_count=len(retained_ids),
            already_current_count=len(completed),
            regenerated_count=regenerated_count,
        )

    def _sample_reconciliation(
        self,
        plan: SampleMigrationPlan,
        sidecar: AlignmentSidecar,
    ) -> SampleReconciliation:
        records_by_track = {
            track.id: tuple(
                record for record in plan.transcripts if record.sample_track_id == track.id
            )
            for track in plan.sample.tracks
        }
        track_reconciliations: list[TrackReconciliation] = []
        for track, rewrite in (
            (plan.sample.tracks[0], sidecar.speaker1),
            (plan.sample.tracks[1], sidecar.speaker2),
        ):
            records = records_by_track[track.id]
            if len(records) != 2:
                raise ValueError("A retained track does not have exactly two ASR rows.")
            metadata = inspect_pcm_wave(track.access_uri)
            source_duration = metadata.frame_count / metadata.sample_rate
            source_hash = sha256_file(track.access_uri)
            if source_hash != rewrite.aligned_sha256:
                raise ValueError("Canonical audio does not match its alignment sidecar.")
            already_reconciled = all(
                record.source_audio_sha256 == rewrite.aligned_sha256 for record in records
            )
            physically_changed = rewrite.original_sha256 != rewrite.aligned_sha256
            shared_prepared: PreparedAudioProvenance | None = None
            prepared_path: Path | None = None
            if physically_changed and already_reconciled:
                prepared_values = {
                    (
                        record.prepared_audio_sha256,
                        record.prepared_duration_seconds,
                    )
                    for record in records
                }
                if len(prepared_values) != 1:
                    raise ValueError("Reconciled ASR rows disagree on prepared-audio provenance.")
                prepared_hash, prepared_duration = prepared_values.pop()
                shared_prepared = PreparedAudioProvenance(
                    sha256=prepared_hash,
                    duration_seconds=prepared_duration,
                )
            elif physically_changed:
                generated = self.prepared_audio_factory(
                    source_path=track.access_uri,
                    source_audio_sha256=source_hash,
                )
                prepared_path = generated.path
                shared_prepared = PreparedAudioProvenance(
                    sha256=generated.prepared_audio_sha256,
                    duration_seconds=generated.prepared_duration_seconds,
                )
            try:
                shift_seconds = (
                    0.0
                    if already_reconciled or not physically_changed
                    else rewrite.prepended_silence_frame_count / sidecar.sample_rate
                )
                transcript_reconciliations = tuple(
                    TranscriptReconciliation(
                        id=record.id,
                        source_audio_sha256=source_hash,
                        prepared_audio=shared_prepared
                        or PreparedAudioProvenance(
                            sha256=record.prepared_audio_sha256,
                            duration_seconds=record.prepared_duration_seconds,
                        ),
                        source_duration_seconds=source_duration,
                        words=shift_timestamped_words(
                            words=record.words,
                            shift_seconds=shift_seconds,
                            duration_seconds=source_duration,
                        ),
                    )
                    for record in records
                )
                assert len(transcript_reconciliations) == 2
                track_reconciliations.append(
                    TrackReconciliation(
                        id=track.id,
                        audio_sha256=source_hash,
                        duration_seconds=source_duration,
                        sample_rate=metadata.sample_rate,
                        channels=metadata.channel_count,
                        sample_count=metadata.frame_count,
                        transcripts=(
                            transcript_reconciliations[0],
                            transcript_reconciliations[1],
                        ),
                    )
                )
            finally:
                if prepared_path is not None:
                    prepared_path.unlink(missing_ok=True)
        if len(track_reconciliations) != 2:
            raise ValueError("Sample reconciliation requires exactly two tracks.")
        return SampleReconciliation(
            id=plan.sample.id,
            duration_seconds=max(track.duration_seconds for track in track_reconciliations),
            tracks=(track_reconciliations[0], track_reconciliations[1]),
        )

    def _delete_invalid_samples(self, plan: AlignmentMigrationPlan) -> None:
        stages = tuple(
            invalid_directory_stage(self.sessions_root, sample.external_id)
            for sample in plan.invalid_samples
        )
        staged_now: list[InvalidDirectoryStage] = []
        try:
            for stage in stages:
                if not stage.canonical.is_dir() or stage.staged.exists():
                    raise ValueError("Invalid sample directory state changed after preflight.")
                os.replace(stage.canonical, stage.staged)
                staged_now.append(stage)
            invalid_ids = {sample.external_id for sample in plan.invalid_samples}
            invalid_cache_ids = tuple(
                record.id
                for record in plan.cached_asr.dataset_records
                if cached_asr_external_id(record.audio_filename) in invalid_ids
            )
            self.repository.delete_invalid_samples(
                InvalidSampleDeletion(
                    sample_ids=tuple(sample.id for sample in plan.invalid_samples),
                    expected_external_ids=tuple(
                        sample.external_id for sample in plan.invalid_samples
                    ),
                    cached_transcript_ids=invalid_cache_ids,
                )
            )
        except Exception:
            for stage in reversed(staged_now):
                if stage.staged.exists() and not stage.canonical.exists():
                    os.replace(stage.staged, stage.canonical)
            raise
        for stage in stages:
            _delete_contained_stage(self.sessions_root, stage.staged)

    def _recover_invalid_directory_state(self) -> None:
        snapshot = self.repository.load_snapshot(self.sessions_root)
        database_invalid_ids = {
            sample.external_id
            for sample in snapshot.samples
            if sample.external_id in INVALID_SAMPLE_IDS
        }
        stages = tuple(
            invalid_directory_stage(self.sessions_root, external_id)
            for external_id in sorted(INVALID_SAMPLE_IDS)
        )
        staged_ids = {stage.external_id for stage in stages if stage.staged.exists()}
        if not staged_ids:
            return
        if staged_ids != INVALID_SAMPLE_IDS:
            raise ValueError("Only part of the invalid sample set is staged.")
        if database_invalid_ids == INVALID_SAMPLE_IDS:
            for stage in stages:
                if stage.canonical.exists():
                    raise ValueError("Canonical and staged invalid directories both exist.")
                os.replace(stage.staged, stage.canonical)
            return
        if not database_invalid_ids:
            for stage in stages:
                _delete_contained_stage(self.sessions_root, stage.staged)
            return
        raise ValueError("Invalid sample database deletion is only partially complete.")


def classify_cached_asr(
    records: tuple[CachedAsrMigrationRecord, ...],
    known_dataset_ids: set[str] | frozenset[str],
) -> CachedAsrClassification:
    dataset: list[CachedAsrMigrationRecord] = []
    unrelated: list[CachedAsrMigrationRecord] = []
    for record in records:
        external_id = cached_asr_external_id(record.audio_filename)
        if external_id is None:
            if record.audio_filename.startswith("pmt_"):
                raise ValueError(
                    f"Unrecognized dataset cached ASR filename: {record.audio_filename}"
                )
            unrelated.append(record)
            continue
        if external_id not in known_dataset_ids:
            raise ValueError(
                f"Cached ASR filename references an unknown dataset sample: {record.audio_filename}"
            )
        dataset.append(record)
    return CachedAsrClassification(
        dataset_records=tuple(dataset),
        unrelated_records=tuple(unrelated),
    )


def cached_asr_external_id(audio_filename: str) -> str | None:
    primary = _PRIMARY_CACHE_PATTERN.fullmatch(audio_filename)
    if primary is not None:
        return primary.group(1)
    analysis = _ANALYSIS_CACHE_PATTERN.fullmatch(audio_filename)
    return analysis.group(1) if analysis is not None else None


def shift_timestamped_words(
    words: tuple[TimestampedWord, ...],
    shift_seconds: float,
    duration_seconds: float,
) -> tuple[TimestampedWord, ...]:
    shifted: list[TimestampedWord] = []
    for word in words:
        if word.start_seconds is None or word.end_seconds is None:
            raise ValueError("Migration requires complete ASR word timestamps.")
        start = word.start_seconds + shift_seconds
        end = word.end_seconds + shift_seconds
        if start < 0.0 or end < start or end > duration_seconds + 1e-6:
            raise ValueError("Shifted ASR timestamp falls outside the aligned audio.")
        shifted.append(
            word.model_copy(
                update={
                    "start_seconds": start,
                    "end_seconds": end,
                }
            )
        )
    return tuple(shifted)


def invalid_directory_stage(sessions_root: Path, external_id: str) -> InvalidDirectoryStage:
    root = sessions_root.resolve(strict=False)
    canonical = (root / external_id).resolve(strict=False)
    staged = (root / f".{external_id}.alignment-invalid.staged").resolve(strict=False)
    if canonical.parent != root or staged.parent != root:
        raise ValueError("Invalid sample path escapes the configured sessions root.")
    return InvalidDirectoryStage(
        external_id=external_id,
        canonical=canonical,
        staged=staged,
    )


def _delete_contained_stage(sessions_root: Path, staged: Path) -> None:
    root = sessions_root.resolve(strict=False)
    resolved = staged.resolve(strict=False)
    if resolved.parent != root or not resolved.name.endswith(".alignment-invalid.staged"):
        raise ValueError("Refusing to delete a path outside the invalid staging namespace.")
    if resolved.exists():
        shutil.rmtree(resolved)


def _reconciled_offsets(
    snapshot: MigrationRepositorySnapshot,
    retained_ids: set[str],
) -> dict[str, float]:
    offsets = {
        alignment.external_id: alignment.speaker2_shift_seconds for alignment in REVIEWED_ALIGNMENTS
    }
    for review in snapshot.reviews:
        previous = offsets.get(review.external_id)
        if previous is not None and previous != review.speaker2_shift_seconds:
            raise ValueError(f"Stored and static reviews disagree for {review.external_id}.")
        offsets[review.external_id] = review.speaker2_shift_seconds
    for sample in snapshot.samples:
        if sample.external_id not in retained_ids:
            continue
        sidecar_path = sample.tracks[0].access_uri.parent / f"{sample.external_id}.alignment.json"
        if sidecar_path.exists():
            sidecar = AlignmentSidecar.model_validate_json(sidecar_path.read_text(encoding="utf-8"))
            _validate_existing_sidecar(sample=sample, sidecar=sidecar)
            previous = offsets.get(sample.external_id)
            if previous is not None and previous != sidecar.reviewed_speaker2_shift_seconds:
                raise ValueError(f"Review and sidecar disagree for {sample.external_id}.")
            offsets[sample.external_id] = sidecar.reviewed_speaker2_shift_seconds
    missing = retained_ids - offsets.keys()
    unexpected = offsets.keys() - retained_ids - INVALID_SAMPLE_IDS
    if missing:
        raise ValueError(f"Retained samples without reviewed offsets: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Reviewed offsets reference unknown samples: {sorted(unexpected)}")
    selected = {external_id: offsets[external_id] for external_id in retained_ids}
    for external_id, offset in selected.items():
        if not math.isfinite(offset) or abs(offset) > MAXIMUM_OFFSET_SECONDS:
            raise ValueError(f"Invalid reviewed offset for {external_id}: {offset}")
    return selected


def _validate_transcript_coverage(
    snapshot: MigrationRepositorySnapshot,
    retained_ids: set[str],
    invalid_present: set[str],
) -> dict[str, tuple[FullAsrMigrationRecord, ...]]:
    expected_count = EXPECTED_RETAINED_TRANSCRIPT_COUNT + (
        EXPECTED_INVALID_TRANSCRIPT_COUNT if invalid_present else 0
    )
    if len(snapshot.transcripts) != expected_count:
        raise ValueError(
            f"Expected {expected_count} full-recording ASR rows, found {len(snapshot.transcripts)}."
        )
    grouped: dict[str, list[FullAsrMigrationRecord]] = {}
    for record in snapshot.transcripts:
        grouped.setdefault(record.sample_external_id, []).append(record)
        _validate_word_timestamps(record)
    for external_id in retained_ids:
        records = grouped.get(external_id, [])
        expected = {
            (AlignmentSide.SPEAKER1, AsrModelId.PARAKEET_TDT),
            (AlignmentSide.SPEAKER1, AsrModelId.CANARY),
            (AlignmentSide.SPEAKER2, AsrModelId.PARAKEET_TDT),
            (AlignmentSide.SPEAKER2, AsrModelId.CANARY),
        }
        if len(records) != 4 or {(record.side, record.model_id) for record in records} != expected:
            raise ValueError(f"Retained ASR coverage is incomplete for {external_id}.")
    for external_id in invalid_present:
        records = grouped.get(external_id, [])
        expected = {
            (AlignmentSide.SPEAKER1, AsrModelId.PARAKEET_TDT),
            (AlignmentSide.SPEAKER2, AsrModelId.PARAKEET_TDT),
        }
        if len(records) != 2 or {(record.side, record.model_id) for record in records} != expected:
            raise ValueError(f"Invalid-sample ASR coverage is unexpected for {external_id}.")
    return {external_id: tuple(records) for external_id, records in grouped.items()}


def _validate_word_timestamps(record: FullAsrMigrationRecord) -> None:
    for word in record.words:
        if word.start_seconds is None or word.end_seconds is None:
            raise ValueError("Migration requires complete full-recording ASR timestamps.")
        if (
            word.start_seconds < 0.0
            or word.end_seconds < word.start_seconds
            or word.end_seconds > record.source_duration_seconds + 1e-6
        ):
            raise ValueError("Full-recording ASR timestamps are invalid.")


def _validate_sample_files(sample: MigrationSample, sessions_root: Path) -> None:
    for track in sample.tracks:
        _validate_contained_track(track.access_uri, sessions_root, sample.external_id)
        inspect_pcm_wave(track.access_uri)
    if sample.tracks[0].access_uri.parent != sample.tracks[1].access_uri.parent:
        raise ValueError(f"Sample {sample.external_id} tracks are in different directories.")
    sample_directory = sample.tracks[0].access_uri.parent
    known_artifacts = (
        {f"{track.access_uri.name}.alignment.staged" for track in sample.tracks}
        | {f"{track.access_uri.name}.alignment.bak" for track in sample.tracks}
        | {
            f"{sample.external_id}.alignment.json",
            f"{sample.external_id}.alignment.json.pending",
        }
    )
    for path in sample_directory.iterdir():
        if ".alignment." in path.name and path.name not in known_artifacts:
            raise ValueError(f"Unknown alignment transaction artifact: {path}")


def _validate_invalid_sample_files(sample: MigrationSample, sessions_root: Path) -> None:
    directory = (sessions_root / sample.external_id).resolve(strict=False)
    if not directory.is_dir():
        raise ValueError(f"Invalid sample directory is missing: {directory}")
    _validate_sample_files(sample=sample, sessions_root=sessions_root)


def _validate_contained_track(path: Path, sessions_root: Path, external_id: str) -> None:
    resolved = path.resolve(strict=False)
    expected_parent = (sessions_root / external_id).resolve(strict=False)
    if resolved.parent != expected_parent or not resolved.is_file():
        raise ValueError(f"Track path is missing or outside its sample directory: {path}")


def _validate_existing_sidecar(
    sample: MigrationSample,
    sidecar: AlignmentSidecar,
) -> None:
    if sidecar.sample_external_id != sample.external_id:
        raise ValueError("Alignment sidecar belongs to a different sample.")
    expected_frame_count = round(sidecar.reviewed_speaker2_shift_seconds * sidecar.sample_rate)
    if sidecar.applied_speaker2_shift_frame_count != expected_frame_count:
        raise ValueError("Alignment sidecar offset frames are inconsistent.")
    for track, rewrite, side in (
        (sample.tracks[0], sidecar.speaker1, AlignmentSide.SPEAKER1),
        (sample.tracks[1], sidecar.speaker2, AlignmentSide.SPEAKER2),
    ):
        if rewrite.side is not side or rewrite.filename != track.access_uri.name:
            raise ValueError("Alignment sidecar track identity is inconsistent.")
        if sha256_file(track.access_uri) != rewrite.aligned_sha256:
            raise ValueError("Alignment sidecar does not match canonical audio.")
        metadata = inspect_pcm_wave(track.access_uri)
        if (
            metadata.sample_rate != sidecar.sample_rate
            or metadata.frame_count != rewrite.aligned_frame_count
        ):
            raise ValueError("Alignment sidecar audio metadata is inconsistent.")


def _transaction_paths(sample: MigrationSample) -> AlignmentTransactionPaths:
    speaker1, speaker2 = sample.tracks
    return AlignmentTransactionPaths.create(
        sample_directory=speaker1.access_uri.parent,
        sample_external_id=sample.external_id,
        speaker1_filename=speaker1.access_uri.name,
        speaker2_filename=speaker2.access_uri.name,
    )


def _database_is_reconciled(
    plan: SampleMigrationPlan,
    sidecar: AlignmentSidecar,
) -> bool:
    rewrite_by_side = {
        AlignmentSide.SPEAKER1: sidecar.speaker1,
        AlignmentSide.SPEAKER2: sidecar.speaker2,
    }
    expected_sample_duration = max(
        rewrite.aligned_frame_count / sidecar.sample_rate for rewrite in rewrite_by_side.values()
    )
    if plan.sample.duration_seconds != expected_sample_duration:
        return False
    for track in plan.sample.tracks:
        rewrite = rewrite_by_side[track.side]
        expected_duration = rewrite.aligned_frame_count / sidecar.sample_rate
        if (
            track.audio_sha256 != rewrite.aligned_sha256
            or track.duration_seconds != expected_duration
            or track.sample_rate != sidecar.sample_rate
            or track.sample_count != rewrite.aligned_frame_count
        ):
            return False
        records = tuple(record for record in plan.transcripts if record.sample_track_id == track.id)
        if len(records) != 2:
            return False
        if any(
            record.source_audio_sha256 != rewrite.aligned_sha256
            or abs(record.source_duration_seconds - expected_duration) > 1e-6
            for record in records
        ):
            return False
    return True


def _audit_post_migration_sources(
    plan: AlignmentMigrationPlan,
    sessions_root: Path,
) -> None:
    if plan.invalid_samples:
        raise ValueError("Post-migration source audit found invalid PostgreSQL samples.")
    constraints = plan.snapshot.track_hash_constraints
    if not constraints.format_validated or not constraints.required_validated:
        raise ValueError("Post-migration sample-track hash constraints are not validated.")
    root = sessions_root.resolve(strict=False)
    expected_external_ids = {sample.sample.external_id for sample in plan.retained}
    root_entries = tuple(root.iterdir())
    if (
        any(not entry.is_dir() for entry in root_entries)
        or {entry.name for entry in root_entries} != expected_external_ids
    ):
        raise ValueError("Sessions root does not contain the exact retained directory set.")
    for external_id in INVALID_SAMPLE_IDS:
        stage = invalid_directory_stage(root, external_id)
        if stage.canonical.exists() or stage.staged.exists():
            raise ValueError("Post-migration source audit found an invalid sample directory.")
    canonical_wav_count = 0
    sidecar_count = 0
    legacy_json_count = 0
    for sample_plan in plan.retained:
        sample = sample_plan.sample
        paths = _transaction_paths(sample)
        legacy_json = paths.final_sidecar.parent / f"{sample.external_id}.json"
        expected_files = {
            sample.tracks[0].access_uri.resolve(strict=False),
            sample.tracks[1].access_uri.resolve(strict=False),
            paths.final_sidecar.resolve(strict=False),
            legacy_json.resolve(strict=False),
        }
        actual_entries = tuple(paths.final_sidecar.parent.iterdir())
        if (
            any(not entry.is_file() for entry in actual_entries)
            or {entry.resolve(strict=False) for entry in actual_entries} != expected_files
        ):
            raise ValueError(
                f"Sample {sample.external_id} does not have the exact post-migration files."
            )
        if any(track.access_uri.suffix.lower() != ".wav" for track in sample.tracks):
            raise ValueError("Post-migration canonical tracks must be WAV files.")
        if not paths.final_sidecar.is_file() or not legacy_json.is_file():
            raise ValueError("Post-migration sidecar or legacy sample JSON is missing.")
        if any(
            path.exists()
            for path in (
                paths.pending_sidecar,
                paths.speaker1.staged,
                paths.speaker1.backup,
                paths.speaker2.staged,
                paths.speaker2.backup,
            )
        ):
            raise ValueError("Post-migration source audit found transaction artifacts.")
        sidecar = AlignmentSidecar.model_validate_json(
            paths.final_sidecar.read_text(encoding="utf-8")
        )
        if sidecar.reviewed_speaker2_shift_seconds != sample_plan.shift_seconds:
            raise ValueError("Post-migration sidecar has the wrong reviewed offset.")
        _validate_existing_sidecar(sample=sample, sidecar=sidecar)
        expected_sample_duration = 0.0
        for track, rewrite in (
            (sample.tracks[0], sidecar.speaker1),
            (sample.tracks[1], sidecar.speaker2),
        ):
            if Path(track.storage_uri).resolve(strict=False) != track.access_uri.resolve(
                strict=False
            ):
                raise ValueError("Post-migration storage and access paths disagree.")
            metadata = inspect_pcm_wave(track.access_uri)
            duration_seconds = metadata.frame_count / metadata.sample_rate
            expected_payload = AudioMetadata(
                duration_seconds=duration_seconds,
                sample_rate=metadata.sample_rate,
                channels=metadata.channel_count,
                sample_count=metadata.frame_count,
            )
            if (
                track.audio_sha256 != rewrite.aligned_sha256
                or track.duration_seconds != duration_seconds
                or track.sample_rate != metadata.sample_rate
                or track.channels != metadata.channel_count
                or track.sample_count != metadata.frame_count
            ):
                raise ValueError("Post-migration sample-track metadata is not canonical.")
            stored_metadata = track.audio_metadata
            if (
                stored_metadata.duration_seconds != duration_seconds
                or stored_metadata.sample_rate != metadata.sample_rate
                or stored_metadata.channels != metadata.channel_count
                or stored_metadata.sample_count != metadata.frame_count
                or stored_metadata.payload != expected_payload
            ):
                raise ValueError("Post-migration audio_metadata is not canonical.")
            records = tuple(
                record for record in sample_plan.transcripts if record.sample_track_id == track.id
            )
            if len(records) != 2 or {record.model_id for record in records} != {
                AsrModelId.PARAKEET_TDT,
                AsrModelId.CANARY,
            }:
                raise ValueError("Post-migration track ASR model coverage is not exact.")
            if any(
                record.source_audio_sha256 != rewrite.aligned_sha256
                or record.source_duration_seconds != duration_seconds
                for record in records
            ):
                raise ValueError("Post-migration ASR source provenance is not canonical.")
            if rewrite.original_sha256 != rewrite.aligned_sha256 and (
                len(
                    {
                        (
                            record.prepared_audio_sha256,
                            record.prepared_duration_seconds,
                        )
                        for record in records
                    }
                )
                != 1
            ):
                raise ValueError("Post-migration ASR prepared provenance disagrees by model.")
            for record in records:
                _validate_word_timestamps(record)
            expected_sample_duration = max(expected_sample_duration, duration_seconds)
            canonical_wav_count += 1
        if sample.duration_seconds != expected_sample_duration:
            raise ValueError("Post-migration sample duration is not canonical.")
        sidecar_count += 1
        legacy_json_count += 1
    if (
        canonical_wav_count != 330
        or sidecar_count != EXPECTED_RETAINED_SAMPLE_COUNT
        or legacy_json_count != EXPECTED_RETAINED_SAMPLE_COUNT
    ):
        raise ValueError("Post-migration filesystem cardinality is not exact.")


def _four_transcripts(
    records: tuple[FullAsrMigrationRecord, ...],
) -> tuple[
    FullAsrMigrationRecord,
    FullAsrMigrationRecord,
    FullAsrMigrationRecord,
    FullAsrMigrationRecord,
]:
    if len(records) != 4:
        raise ValueError("Retained samples require exactly four ASR rows.")
    ordered = tuple(sorted(records, key=lambda record: (record.side.value, record.model_id.value)))
    return (ordered[0], ordered[1], ordered[2], ordered[3])


def _validate_legacy_tables(snapshot: MigrationRepositorySnapshot) -> None:
    counts = snapshot.legacy_counts
    if any(
        (
            counts.asr_runs,
            counts.asr_words,
            counts.asr_evaluations,
            counts.annotation_runs,
            counts.annotation_events,
        )
    ):
        raise ValueError("Legacy ASR or annotation tables are not empty.")


def _validate_expected_summary(summary: MigrationPreflightSummary) -> None:
    expected = (
        EXPECTED_NEGATIVE_OFFSET_COUNT,
        EXPECTED_POSITIVE_OFFSET_COUNT,
        EXPECTED_ZERO_OFFSET_COUNT,
        EXPECTED_REWRITTEN_SAMPLE_COUNT,
    )
    actual = (
        summary.negative_offset_count,
        summary.positive_offset_count,
        summary.zero_offset_count,
        summary.rewritten_sample_count,
    )
    if actual != expected:
        raise ValueError(f"Reviewed offset distribution is {actual}, expected {expected}.")
