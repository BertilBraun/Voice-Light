from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.discovery import DatasetLayout, discover_samples
from app.storage.local import LocalStorageBackend


@pytest.mark.parametrize("suffix", [".wav", ".mp3", ".m4a", ".ogg", ".flac"])
def test_two_audio_files_layout_supports_audio_suffixes(tmp_path: Path, suffix: str) -> None:
    sample_root = tmp_path / "sessions"
    sample_directory = sample_root / "sample_001"
    sample_directory.mkdir(parents=True)
    speaker1_path = sample_directory / f"sample_001_speaker1{suffix}"
    speaker2_path = sample_directory / f"sample_001_speaker2{suffix.upper()}"
    speaker1_path.write_bytes(b"speaker1")
    speaker2_path.write_bytes(b"speaker2")
    (sample_directory / "notes.txt").write_text("not audio", encoding="utf-8")

    samples = discover_samples(
        LocalStorageBackend(), str(sample_root), DatasetLayout.TWO_AUDIO_FILES
    )

    assert len(samples) == 1
    assert samples[0].external_id == "sample_001"
    assert samples[0].speaker1_path == speaker1_path.as_posix()
    assert samples[0].speaker2_path == speaker2_path.as_posix()


def test_two_audio_files_layout_requires_exactly_two_audio_files(tmp_path: Path) -> None:
    sample_root = tmp_path / "sessions"
    sample_directory = sample_root / "sample_001"
    sample_directory.mkdir(parents=True)
    for name in ("speaker1.wav", "speaker2.wav", "extra.wav"):
        (sample_directory / name).write_bytes(b"audio")

    samples = discover_samples(
        LocalStorageBackend(), str(sample_root), DatasetLayout.TWO_AUDIO_FILES
    )

    assert samples == []


def test_mix_layout_excludes_the_mix_track(tmp_path: Path) -> None:
    sample_root = tmp_path / "sessions"
    sample_directory = sample_root / "sample_001"
    sample_directory.mkdir(parents=True)
    for name in ("call.mix.wav", "call.speaker1.flac", "call.speaker2.mp3"):
        (sample_directory / name).write_bytes(b"audio")

    samples = discover_samples(
        LocalStorageBackend(), str(sample_root), DatasetLayout.MIX_AND_TWO_SPEAKERS
    )

    assert len(samples) == 1
    assert samples[0].speaker1_path.endswith("call.speaker1.flac")
    assert samples[0].speaker2_path.endswith("call.speaker2.mp3")
