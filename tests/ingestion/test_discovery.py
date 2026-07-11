from __future__ import annotations

from pathlib import Path

from app.ingestion.discovery import discover_two_track_samples
from app.storage.local import LocalStorageBackend


def test_discover_two_track_samples_prefers_speaker_names(tmp_path: Path) -> None:
    sample_root = tmp_path / "sessions"
    sample_directory = sample_root / "sample_001"
    sample_directory.mkdir(parents=True)
    speaker1_path = sample_directory / "sample_001_speaker1.wav"
    speaker2_path = sample_directory / "sample_001_speaker2.wav"
    ignored_path = sample_directory / "notes.txt"
    speaker1_path.write_bytes(b"speaker1")
    speaker2_path.write_bytes(b"speaker2")
    ignored_path.write_text("not audio", encoding="utf-8")

    samples = discover_two_track_samples(LocalStorageBackend(), sample_root)

    assert len(samples) == 1
    assert samples[0].external_id == "sample_001"
    assert samples[0].speaker1_path == speaker1_path
    assert samples[0].speaker2_path == speaker2_path
