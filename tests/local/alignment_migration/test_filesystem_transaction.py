from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import app.local.alignment_migration.filesystem_transaction as transaction_module
from app.local.alignment_migration.filesystem_transaction import (
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
    _assert_sidecar_matches_canonical(alignment_paths, recovered.sidecar)
    assert alignment_paths.speaker1.backup.is_file()
    assert alignment_paths.speaker2.backup.is_file()


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
