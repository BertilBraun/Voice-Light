from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from app.local.alignment_migration.models import (
    AlignmentApplicationStatus,
    AlignmentSide,
    AlignmentSidecar,
    AlignmentTrackRewrite,
)
from app.local.alignment_migration.riff import (
    PcmWaveMetadata,
    inspect_pcm_wave,
    rewrite_pcm_wave,
    sha256_file,
)


@dataclass(frozen=True)
class AlignmentTrackPaths:
    canonical: Path
    staged: Path
    backup: Path


@dataclass(frozen=True)
class AlignmentTransactionPaths:
    sample_external_id: str
    speaker1: AlignmentTrackPaths
    speaker2: AlignmentTrackPaths
    final_sidecar: Path
    pending_sidecar: Path

    @classmethod
    def create(
        cls,
        sample_directory: Path,
        sample_external_id: str,
        speaker1_filename: str,
        speaker2_filename: str,
    ) -> AlignmentTransactionPaths:
        final_sidecar = sample_directory / f"{sample_external_id}.alignment.json"
        return cls(
            sample_external_id=sample_external_id,
            speaker1=_track_paths(sample_directory / speaker1_filename),
            speaker2=_track_paths(sample_directory / speaker2_filename),
            final_sidecar=final_sidecar,
            pending_sidecar=final_sidecar.with_name(f"{final_sidecar.name}.pending"),
        )


@dataclass(frozen=True)
class AlignmentApplicationResult:
    status: AlignmentApplicationStatus
    sidecar: AlignmentSidecar


def apply_alignment_transaction(
    paths: AlignmentTransactionPaths,
    reviewed_speaker2_shift_seconds: float,
    applied_at: datetime | None = None,
) -> AlignmentApplicationResult:
    if not math.isfinite(reviewed_speaker2_shift_seconds):
        raise ValueError("Reviewed synchronization offset must be finite.")
    if paths.final_sidecar.exists():
        sidecar = _load_sidecar(paths.final_sidecar)
        _validate_completed_application(
            paths=paths,
            sidecar=sidecar,
            requested_shift_seconds=reviewed_speaker2_shift_seconds,
        )
        return AlignmentApplicationResult(
            status=AlignmentApplicationStatus.ALREADY_APPLIED,
            sidecar=sidecar,
        )
    if _has_transaction_artifacts(paths):
        if not paths.pending_sidecar.exists():
            raise ValueError(
                "Alignment transaction artifacts exist without a pending sidecar; "
                "manual inspection is required."
            )
        recovered = recover_alignment_transaction(paths)
        _validate_requested_application(
            paths=paths,
            sidecar=recovered.sidecar,
            requested_shift_seconds=reviewed_speaker2_shift_seconds,
        )
        return recovered

    speaker1_metadata = inspect_pcm_wave(paths.speaker1.canonical)
    speaker2_metadata = inspect_pcm_wave(paths.speaker2.canonical)
    if speaker1_metadata.sample_rate != speaker2_metadata.sample_rate:
        raise ValueError("Alignment requires matching WAV sample rates.")
    shift_frame_count = round(reviewed_speaker2_shift_seconds * speaker1_metadata.sample_rate)
    speaker1_prepend = max(0, -shift_frame_count)
    speaker2_prepend = max(0, shift_frame_count)
    if shift_frame_count == 0:
        sidecar = _unchanged_sidecar(
            paths=paths,
            speaker1_metadata=speaker1_metadata,
            speaker2_metadata=speaker2_metadata,
            reviewed_speaker2_shift_seconds=reviewed_speaker2_shift_seconds,
            applied_at=applied_at or datetime.now(tz=UTC),
        )
        _write_pending_sidecar(paths.pending_sidecar, sidecar)
        _promote_sidecar(paths)
        return AlignmentApplicationResult(
            status=AlignmentApplicationStatus.APPLIED,
            sidecar=sidecar,
        )

    target_frame_count = max(
        speaker1_metadata.frame_count + speaker1_prepend,
        speaker2_metadata.frame_count + speaker2_prepend,
    )
    try:
        speaker1_rewritten = rewrite_pcm_wave(
            source_path=paths.speaker1.canonical,
            output_path=paths.speaker1.staged,
            prepended_silence_frame_count=speaker1_prepend,
            target_frame_count=target_frame_count,
        )
        speaker2_rewritten = rewrite_pcm_wave(
            source_path=paths.speaker2.canonical,
            output_path=paths.speaker2.staged,
            prepended_silence_frame_count=speaker2_prepend,
            target_frame_count=target_frame_count,
        )
        sidecar = AlignmentSidecar(
            sample_external_id=paths.sample_external_id,
            reviewed_speaker2_shift_seconds=reviewed_speaker2_shift_seconds,
            sample_rate=speaker1_metadata.sample_rate,
            applied_speaker2_shift_frame_count=shift_frame_count,
            speaker1=AlignmentTrackRewrite(
                side=AlignmentSide.SPEAKER1,
                filename=paths.speaker1.canonical.name,
                original_sha256=sha256_file(paths.speaker1.canonical),
                aligned_sha256=speaker1_rewritten.sha256,
                original_frame_count=speaker1_metadata.frame_count,
                aligned_frame_count=speaker1_rewritten.frame_count,
                prepended_silence_frame_count=(speaker1_rewritten.prepended_silence_frame_count),
                appended_silence_frame_count=(speaker1_rewritten.appended_silence_frame_count),
            ),
            speaker2=AlignmentTrackRewrite(
                side=AlignmentSide.SPEAKER2,
                filename=paths.speaker2.canonical.name,
                original_sha256=sha256_file(paths.speaker2.canonical),
                aligned_sha256=speaker2_rewritten.sha256,
                original_frame_count=speaker2_metadata.frame_count,
                aligned_frame_count=speaker2_rewritten.frame_count,
                prepended_silence_frame_count=(speaker2_rewritten.prepended_silence_frame_count),
                appended_silence_frame_count=(speaker2_rewritten.appended_silence_frame_count),
            ),
            applied_at=applied_at or datetime.now(tz=UTC),
        )
        _write_pending_sidecar(paths.pending_sidecar, sidecar)
    except Exception:
        if not paths.pending_sidecar.exists():
            paths.speaker1.staged.unlink(missing_ok=True)
            paths.speaker2.staged.unlink(missing_ok=True)
        raise
    return _complete_pending_transaction(paths=paths, recovered=False)


def recover_alignment_transaction(
    paths: AlignmentTransactionPaths,
) -> AlignmentApplicationResult:
    if paths.final_sidecar.exists():
        sidecar = _load_sidecar(paths.final_sidecar)
        _validate_completed_application(paths=paths, sidecar=sidecar)
        if paths.pending_sidecar.exists():
            pending = _load_sidecar(paths.pending_sidecar)
            if pending != sidecar:
                raise ValueError("Final and pending alignment sidecars differ.")
            paths.pending_sidecar.unlink()
        return AlignmentApplicationResult(
            status=AlignmentApplicationStatus.ALREADY_APPLIED,
            sidecar=sidecar,
        )
    if not paths.pending_sidecar.is_file():
        raise ValueError("No pending alignment sidecar is available for recovery.")
    return _complete_pending_transaction(paths=paths, recovered=True)


def confirm_database_reconciliation(
    paths: AlignmentTransactionPaths,
    expected_sidecar: AlignmentSidecar,
) -> None:
    if not paths.final_sidecar.is_file():
        raise ValueError("Cannot confirm reconciliation before the final sidecar exists.")
    final_sidecar = _load_sidecar(paths.final_sidecar)
    if final_sidecar != expected_sidecar:
        raise ValueError("Final alignment sidecar differs from the reconciled record.")
    _validate_completed_application(paths=paths, sidecar=final_sidecar)
    backups_to_remove: list[Path] = []
    for track_paths, track_record in _track_pairs(paths, final_sidecar):
        if track_paths.backup.exists():
            if sha256_file(track_paths.backup) != track_record.original_sha256:
                raise ValueError(f"Alignment backup hash mismatch: {track_paths.backup}")
            backups_to_remove.append(track_paths.backup)
    for backup_path in backups_to_remove:
        backup_path.unlink()


def _complete_pending_transaction(
    paths: AlignmentTransactionPaths,
    recovered: bool,
) -> AlignmentApplicationResult:
    sidecar = _load_sidecar(paths.pending_sidecar)
    _validate_sidecar_identity(paths=paths, sidecar=sidecar)
    if sidecar.applied_speaker2_shift_frame_count == 0:
        _validate_zero_offset_sidecar(sidecar)
    else:
        for track_paths, track_record in _track_pairs(paths, sidecar):
            _install_aligned_track(track_paths=track_paths, track_record=track_record)
    _validate_canonical_tracks(paths=paths, sidecar=sidecar)
    _promote_sidecar(paths)
    return AlignmentApplicationResult(
        status=(
            AlignmentApplicationStatus.RECOVERED
            if recovered
            else AlignmentApplicationStatus.APPLIED
        ),
        sidecar=sidecar,
    )


def _install_aligned_track(
    track_paths: AlignmentTrackPaths,
    track_record: AlignmentTrackRewrite,
) -> None:
    backup_hash = _optional_file_hash(track_paths.backup)
    if backup_hash is not None and backup_hash != track_record.original_sha256:
        raise ValueError(f"Alignment backup hash mismatch: {track_paths.backup}")
    staged_hash = _optional_file_hash(track_paths.staged)
    if staged_hash is not None and staged_hash != track_record.aligned_sha256:
        raise ValueError(f"Staged alignment hash mismatch: {track_paths.staged}")
    canonical_hash = _optional_file_hash(track_paths.canonical)
    if canonical_hash == track_record.aligned_sha256:
        track_paths.staged.unlink(missing_ok=True)
        return
    if canonical_hash == track_record.original_sha256:
        if staged_hash != track_record.aligned_sha256:
            raise ValueError(f"Aligned staged file is unavailable: {track_paths.staged}")
        if backup_hash is None:
            os.replace(track_paths.canonical, track_paths.backup)
        os.replace(track_paths.staged, track_paths.canonical)
        return
    if canonical_hash is None:
        if backup_hash != track_record.original_sha256:
            raise ValueError(f"Original alignment backup is unavailable: {track_paths.backup}")
        if staged_hash != track_record.aligned_sha256:
            raise ValueError(f"Aligned staged file is unavailable: {track_paths.staged}")
        os.replace(track_paths.staged, track_paths.canonical)
        return
    raise ValueError(
        f"Canonical audio hash is neither original nor aligned: {track_paths.canonical}"
    )


def _validate_completed_application(
    paths: AlignmentTransactionPaths,
    sidecar: AlignmentSidecar,
    requested_shift_seconds: float | None = None,
) -> None:
    _validate_sidecar_identity(paths=paths, sidecar=sidecar)
    if requested_shift_seconds is not None:
        _validate_requested_application(
            paths=paths,
            sidecar=sidecar,
            requested_shift_seconds=requested_shift_seconds,
        )
    _validate_canonical_tracks(paths=paths, sidecar=sidecar)


def _validate_requested_application(
    paths: AlignmentTransactionPaths,
    sidecar: AlignmentSidecar,
    requested_shift_seconds: float,
) -> None:
    _validate_sidecar_identity(paths=paths, sidecar=sidecar)
    requested_frame_count = round(requested_shift_seconds * sidecar.sample_rate)
    if (
        requested_shift_seconds != sidecar.reviewed_speaker2_shift_seconds
        or requested_frame_count != sidecar.applied_speaker2_shift_frame_count
    ):
        raise ValueError(
            f"Sample {paths.sample_external_id} already has a distinct alignment application."
        )


def _validate_sidecar_identity(
    paths: AlignmentTransactionPaths,
    sidecar: AlignmentSidecar,
) -> None:
    if sidecar.sample_external_id != paths.sample_external_id:
        raise ValueError("Alignment sidecar belongs to a different sample.")
    if sidecar.speaker1.side is not AlignmentSide.SPEAKER1:
        raise ValueError("Alignment sidecar speaker 1 record has the wrong side.")
    if sidecar.speaker2.side is not AlignmentSide.SPEAKER2:
        raise ValueError("Alignment sidecar speaker 2 record has the wrong side.")
    if sidecar.speaker1.filename != paths.speaker1.canonical.name:
        raise ValueError("Alignment sidecar speaker 1 filename is inconsistent.")
    if sidecar.speaker2.filename != paths.speaker2.canonical.name:
        raise ValueError("Alignment sidecar speaker 2 filename is inconsistent.")
    expected_frame_count = round(sidecar.reviewed_speaker2_shift_seconds * sidecar.sample_rate)
    if sidecar.applied_speaker2_shift_frame_count != expected_frame_count:
        raise ValueError("Alignment sidecar offset frames are inconsistent.")
    if sidecar.applied_speaker2_shift_frame_count == 0:
        _validate_zero_offset_sidecar(sidecar)
        return
    expected_speaker1_prepend = max(0, -expected_frame_count)
    expected_speaker2_prepend = max(0, expected_frame_count)
    if (
        sidecar.speaker1.prepended_silence_frame_count != expected_speaker1_prepend
        or sidecar.speaker2.prepended_silence_frame_count != expected_speaker2_prepend
    ):
        raise ValueError("Alignment sidecar leading-silence counts are inconsistent.")
    for track in (sidecar.speaker1, sidecar.speaker2):
        expected_aligned_frames = (
            track.original_frame_count
            + track.prepended_silence_frame_count
            + track.appended_silence_frame_count
        )
        if track.aligned_frame_count != expected_aligned_frames:
            raise ValueError("Alignment sidecar track frame counts are inconsistent.")
    if sidecar.speaker1.aligned_frame_count != sidecar.speaker2.aligned_frame_count:
        raise ValueError("Aligned tracks do not share a final frame count.")


def _validate_zero_offset_sidecar(sidecar: AlignmentSidecar) -> None:
    for track in (sidecar.speaker1, sidecar.speaker2):
        if (
            track.original_sha256 != track.aligned_sha256
            or track.original_frame_count != track.aligned_frame_count
            or track.prepended_silence_frame_count != 0
            or track.appended_silence_frame_count != 0
        ):
            raise ValueError("Zero-offset sidecar must describe unchanged audio.")


def _validate_canonical_tracks(
    paths: AlignmentTransactionPaths,
    sidecar: AlignmentSidecar,
) -> None:
    for track_paths, track_record in _track_pairs(paths, sidecar):
        if not track_paths.canonical.is_file():
            raise ValueError(f"Aligned canonical audio is missing: {track_paths.canonical}")
        if sha256_file(track_paths.canonical) != track_record.aligned_sha256:
            raise ValueError(f"Aligned canonical audio hash mismatch: {track_paths.canonical}")
        metadata = inspect_pcm_wave(track_paths.canonical)
        if metadata.sample_rate != sidecar.sample_rate:
            raise ValueError(f"Aligned canonical sample rate mismatch: {track_paths.canonical}")
        if metadata.frame_count != track_record.aligned_frame_count:
            raise ValueError(f"Aligned canonical frame count mismatch: {track_paths.canonical}")


def _unchanged_sidecar(
    paths: AlignmentTransactionPaths,
    speaker1_metadata: PcmWaveMetadata,
    speaker2_metadata: PcmWaveMetadata,
    reviewed_speaker2_shift_seconds: float,
    applied_at: datetime,
) -> AlignmentSidecar:
    speaker1_hash = sha256_file(paths.speaker1.canonical)
    speaker2_hash = sha256_file(paths.speaker2.canonical)
    return AlignmentSidecar(
        sample_external_id=paths.sample_external_id,
        reviewed_speaker2_shift_seconds=reviewed_speaker2_shift_seconds,
        sample_rate=speaker1_metadata.sample_rate,
        applied_speaker2_shift_frame_count=0,
        speaker1=AlignmentTrackRewrite(
            side=AlignmentSide.SPEAKER1,
            filename=paths.speaker1.canonical.name,
            original_sha256=speaker1_hash,
            aligned_sha256=speaker1_hash,
            original_frame_count=speaker1_metadata.frame_count,
            aligned_frame_count=speaker1_metadata.frame_count,
            prepended_silence_frame_count=0,
            appended_silence_frame_count=0,
        ),
        speaker2=AlignmentTrackRewrite(
            side=AlignmentSide.SPEAKER2,
            filename=paths.speaker2.canonical.name,
            original_sha256=speaker2_hash,
            aligned_sha256=speaker2_hash,
            original_frame_count=speaker2_metadata.frame_count,
            aligned_frame_count=speaker2_metadata.frame_count,
            prepended_silence_frame_count=0,
            appended_silence_frame_count=0,
        ),
        applied_at=applied_at,
    )


def _write_pending_sidecar(path: Path, sidecar: AlignmentSidecar) -> None:
    serialized = f"{sidecar.model_dump_json(indent=2)}\n".encode()
    with path.open("xb") as sidecar_file:
        sidecar_file.write(serialized)
        sidecar_file.flush()
        os.fsync(sidecar_file.fileno())


def _promote_sidecar(paths: AlignmentTransactionPaths) -> None:
    if paths.final_sidecar.exists():
        raise ValueError(f"Final alignment sidecar already exists: {paths.final_sidecar}")
    os.link(paths.pending_sidecar, paths.final_sidecar)
    paths.pending_sidecar.unlink()


def _load_sidecar(path: Path) -> AlignmentSidecar:
    try:
        return AlignmentSidecar.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ValueError(f"Invalid alignment sidecar: {path}") from error


def _track_pairs(
    paths: AlignmentTransactionPaths,
    sidecar: AlignmentSidecar,
) -> tuple[
    tuple[AlignmentTrackPaths, AlignmentTrackRewrite],
    tuple[AlignmentTrackPaths, AlignmentTrackRewrite],
]:
    return (
        (paths.speaker1, sidecar.speaker1),
        (paths.speaker2, sidecar.speaker2),
    )


def _has_transaction_artifacts(paths: AlignmentTransactionPaths) -> bool:
    return any(
        path.exists()
        for path in (
            paths.pending_sidecar,
            paths.speaker1.staged,
            paths.speaker1.backup,
            paths.speaker2.staged,
            paths.speaker2.backup,
        )
    )


def _optional_file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _track_paths(canonical: Path) -> AlignmentTrackPaths:
    return AlignmentTrackPaths(
        canonical=canonical,
        staged=canonical.with_name(f"{canonical.name}.alignment.staged"),
        backup=canonical.with_name(f"{canonical.name}.alignment.bak"),
    )
