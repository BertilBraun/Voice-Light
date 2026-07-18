from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import app.local.alignment_migration.filesystem_transaction as transaction_module
from app.local.alignment_migration.filesystem_transaction import (
    AlignmentApplicationResult,
    AlignmentApplicationStatus,
    AlignmentTrackPaths,
    AlignmentTransactionPaths,
    apply_alignment_transaction,
    confirm_database_reconciliation,
    recover_alignment_transaction,
)
from app.local.alignment_migration.models import AlignmentSidecar, AlignmentTrackRewrite
from app.local.alignment_migration.riff import inspect_pcm_wave, sha256_file

APPLIED_AT = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("shift_seconds", "speaker1_prepend", "speaker2_prepend"),
    (
        (-0.00025, 2, 0),
        (0.00025, 0, 2),
    ),
)
def test_nonzero_offset_physically_rewrites_pair_without_cutting_audio(
    alignment_paths: AlignmentTransactionPaths,
    shift_seconds: float,
    speaker1_prepend: int,
    speaker2_prepend: int,
) -> None:
    original_speaker1 = alignment_paths.speaker1.canonical.read_bytes()
    original_speaker2 = alignment_paths.speaker2.canonical.read_bytes()

    result = apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=shift_seconds,
        applied_at=APPLIED_AT,
    )

    assert result.status is AlignmentApplicationStatus.APPLIED
    assert result.sidecar.speaker1.prepended_silence_frame_count == speaker1_prepend
    assert result.sidecar.speaker2.prepended_silence_frame_count == speaker2_prepend
    assert (
        result.sidecar.speaker1.aligned_frame_count == result.sidecar.speaker2.aligned_frame_count
    )
    assert alignment_paths.final_sidecar.is_file()
    assert not alignment_paths.pending_sidecar.exists()
    assert not alignment_paths.speaker1.staged.exists()
    assert not alignment_paths.speaker2.staged.exists()
    assert alignment_paths.speaker1.backup.read_bytes() == original_speaker1
    assert alignment_paths.speaker2.backup.read_bytes() == original_speaker2
    _assert_sidecar_matches_canonical(alignment_paths, result.sidecar)


def test_zero_offset_writes_sidecar_without_rewriting_audio(
    alignment_paths: AlignmentTransactionPaths,
) -> None:
    original_speaker1 = alignment_paths.speaker1.canonical.read_bytes()
    original_speaker2 = alignment_paths.speaker2.canonical.read_bytes()

    result = apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=0.0,
        applied_at=APPLIED_AT,
    )

    assert result.status is AlignmentApplicationStatus.APPLIED
    assert alignment_paths.speaker1.canonical.read_bytes() == original_speaker1
    assert alignment_paths.speaker2.canonical.read_bytes() == original_speaker2
    assert result.sidecar.speaker1.original_sha256 == result.sidecar.speaker1.aligned_sha256
    assert result.sidecar.speaker2.original_sha256 == result.sidecar.speaker2.aligned_sha256
    assert not alignment_paths.speaker1.backup.exists()
    assert not alignment_paths.speaker2.backup.exists()
    assert not alignment_paths.speaker1.staged.exists()
    assert not alignment_paths.speaker2.staged.exists()


def test_matching_final_sidecar_skips_and_distinct_second_application_is_rejected(
    alignment_paths: AlignmentTransactionPaths,
) -> None:
    first = apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=-0.00025,
        applied_at=APPLIED_AT,
    )

    repeated = apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=-0.00025,
    )

    assert repeated.status is AlignmentApplicationStatus.ALREADY_APPLIED
    assert repeated.sidecar == first.sidecar
    with pytest.raises(ValueError, match="distinct alignment"):
        apply_alignment_transaction(
            paths=alignment_paths,
            reviewed_speaker2_shift_seconds=0.00025,
        )


def test_final_sidecar_hash_mismatch_aborts(
    alignment_paths: AlignmentTransactionPaths,
) -> None:
    apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=0.0,
        applied_at=APPLIED_AT,
    )
    alignment_paths.speaker1.canonical.write_bytes(alignment_paths.speaker2.canonical.read_bytes())

    with pytest.raises(ValueError, match="hash mismatch"):
        apply_alignment_transaction(
            paths=alignment_paths,
            reviewed_speaker2_shift_seconds=0.0,
        )


def test_pending_transaction_recovers_after_first_track_was_installed(
    alignment_paths: AlignmentTransactionPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_install = transaction_module._install_aligned_track
    install_count = 0

    def interrupted_install(
        track_paths: AlignmentTrackPaths,
        track_record: AlignmentTrackRewrite,
    ) -> None:
        nonlocal install_count
        install_count += 1
        if install_count == 2:
            raise RuntimeError("simulated interruption")
        original_install(track_paths, track_record)

    monkeypatch.setattr(transaction_module, "_install_aligned_track", interrupted_install)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        apply_alignment_transaction(
            paths=alignment_paths,
            reviewed_speaker2_shift_seconds=-0.00025,
            applied_at=APPLIED_AT,
        )
    assert alignment_paths.pending_sidecar.is_file()
    assert alignment_paths.speaker1.backup.is_file()
    monkeypatch.setattr(transaction_module, "_install_aligned_track", original_install)

    recovered = recover_alignment_transaction(alignment_paths)

    assert recovered.status is AlignmentApplicationStatus.RECOVERED
    _assert_completed_recovery(alignment_paths, recovered)


def test_rerun_recovers_after_staging_and_pending_sidecar_creation(
    alignment_paths: AlignmentTransactionPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_complete = transaction_module._complete_pending_transaction

    def interrupted_completion(
        paths: AlignmentTransactionPaths,
        recovered: bool,
    ) -> AlignmentApplicationResult:
        del paths, recovered
        raise RuntimeError("simulated interruption after staging")

    monkeypatch.setattr(
        transaction_module,
        "_complete_pending_transaction",
        interrupted_completion,
    )
    with pytest.raises(RuntimeError, match="after staging"):
        apply_alignment_transaction(
            paths=alignment_paths,
            reviewed_speaker2_shift_seconds=-0.00025,
            applied_at=APPLIED_AT,
        )
    assert alignment_paths.pending_sidecar.is_file()
    assert alignment_paths.speaker1.staged.is_file()
    assert alignment_paths.speaker2.staged.is_file()
    assert not alignment_paths.speaker1.backup.exists()
    assert not alignment_paths.speaker2.backup.exists()
    monkeypatch.setattr(
        transaction_module,
        "_complete_pending_transaction",
        original_complete,
    )

    recovered = apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=-0.00025,
    )

    assert recovered.status is AlignmentApplicationStatus.RECOVERED
    _assert_completed_recovery(alignment_paths, recovered)


def test_recovery_after_canonical_was_backed_up_before_staged_replacement(
    alignment_paths: AlignmentTransactionPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = transaction_module.os.replace

    def interrupted_replace(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        if (
            source == alignment_paths.speaker1.canonical
            and destination == alignment_paths.speaker1.backup
        ):
            raise RuntimeError("simulated interruption after backup")

    monkeypatch.setattr(transaction_module.os, "replace", interrupted_replace)
    with pytest.raises(RuntimeError, match="after backup"):
        apply_alignment_transaction(
            paths=alignment_paths,
            reviewed_speaker2_shift_seconds=-0.00025,
            applied_at=APPLIED_AT,
        )
    assert not alignment_paths.speaker1.canonical.exists()
    assert alignment_paths.speaker1.backup.is_file()
    assert alignment_paths.speaker1.staged.is_file()
    monkeypatch.setattr(transaction_module.os, "replace", original_replace)

    recovered = recover_alignment_transaction(alignment_paths)

    assert recovered.status is AlignmentApplicationStatus.RECOVERED
    _assert_completed_recovery(alignment_paths, recovered)


def test_recovery_after_both_tracks_were_replaced_before_sidecar_promotion(
    alignment_paths: AlignmentTransactionPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_promote = transaction_module._promote_sidecar

    def interrupted_promotion(paths: AlignmentTransactionPaths) -> None:
        del paths
        raise RuntimeError("simulated interruption before sidecar promotion")

    monkeypatch.setattr(transaction_module, "_promote_sidecar", interrupted_promotion)
    with pytest.raises(RuntimeError, match="before sidecar promotion"):
        apply_alignment_transaction(
            paths=alignment_paths,
            reviewed_speaker2_shift_seconds=-0.00025,
            applied_at=APPLIED_AT,
        )
    assert alignment_paths.pending_sidecar.is_file()
    assert alignment_paths.speaker1.backup.is_file()
    assert alignment_paths.speaker2.backup.is_file()
    assert not alignment_paths.speaker1.staged.exists()
    assert not alignment_paths.speaker2.staged.exists()
    monkeypatch.setattr(transaction_module, "_promote_sidecar", original_promote)

    recovered = recover_alignment_transaction(alignment_paths)

    assert recovered.status is AlignmentApplicationStatus.RECOVERED
    _assert_completed_recovery(alignment_paths, recovered)


def test_rerun_recovers_after_final_sidecar_link_before_pending_unlink(
    alignment_paths: AlignmentTransactionPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_link = transaction_module.os.link

    def interrupted_link(source: Path, destination: Path) -> None:
        original_link(source, destination)
        raise RuntimeError("simulated interruption after final sidecar link")

    monkeypatch.setattr(transaction_module.os, "link", interrupted_link)
    with pytest.raises(RuntimeError, match="after final sidecar link"):
        apply_alignment_transaction(
            paths=alignment_paths,
            reviewed_speaker2_shift_seconds=-0.00025,
            applied_at=APPLIED_AT,
        )
    assert alignment_paths.final_sidecar.is_file()
    assert alignment_paths.pending_sidecar.is_file()
    monkeypatch.setattr(transaction_module.os, "link", original_link)

    recovered = apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=-0.00025,
    )

    assert recovered.status is AlignmentApplicationStatus.ALREADY_APPLIED
    _assert_completed_recovery(alignment_paths, recovered)


def test_backups_are_removed_only_after_database_reconciliation_confirmation(
    alignment_paths: AlignmentTransactionPaths,
) -> None:
    result = apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=-0.00025,
        applied_at=APPLIED_AT,
    )
    assert alignment_paths.speaker1.backup.is_file()
    assert alignment_paths.speaker2.backup.is_file()

    confirm_database_reconciliation(alignment_paths, result.sidecar)

    assert not alignment_paths.speaker1.backup.exists()
    assert not alignment_paths.speaker2.backup.exists()


def test_sidecar_models_are_frozen(alignment_paths: AlignmentTransactionPaths) -> None:
    result = apply_alignment_transaction(
        paths=alignment_paths,
        reviewed_speaker2_shift_seconds=0.0,
        applied_at=APPLIED_AT,
    )

    with pytest.raises(ValidationError):
        result.sidecar.sample_rate = 16_000


def _assert_sidecar_matches_canonical(
    paths: AlignmentTransactionPaths,
    sidecar: AlignmentSidecar,
) -> None:
    for track_paths, track_record in (
        (paths.speaker1, sidecar.speaker1),
        (paths.speaker2, sidecar.speaker2),
    ):
        assert sha256_file(track_paths.canonical) == track_record.aligned_sha256
        assert (
            inspect_pcm_wave(track_paths.canonical).frame_count == track_record.aligned_frame_count
        )


def _assert_completed_recovery(
    paths: AlignmentTransactionPaths,
    result: AlignmentApplicationResult,
) -> None:
    _assert_sidecar_matches_canonical(paths, result.sidecar)
    assert paths.final_sidecar.is_file()
    assert not paths.pending_sidecar.exists()
    assert not paths.speaker1.staged.exists()
    assert not paths.speaker2.staged.exists()
    assert sha256_file(paths.speaker1.backup) == result.sidecar.speaker1.original_sha256
    assert sha256_file(paths.speaker2.backup) == result.sidecar.speaker2.original_sha256
    with pytest.raises(ValueError, match="distinct alignment"):
        apply_alignment_transaction(
            paths=paths,
            reviewed_speaker2_shift_seconds=0.00025,
        )
