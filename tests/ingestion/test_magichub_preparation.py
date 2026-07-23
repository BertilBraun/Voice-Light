from __future__ import annotations

import json
import wave
import zipfile
from pathlib import Path

import pytest

from app.local.ingestion.magichub_preparation import (
    apply_preparation,
    build_preparation_plan,
    verify_prepared_layout,
)


def test_build_plan_pairs_tracks_and_uses_generic_identifiers(tmp_path: Path) -> None:
    archive_path = write_archive(tmp_path, include_second_pair=True)

    plan = build_preparation_plan(archive_path)

    assert [sample.sample_id for sample in plan.samples] == ["sample_001", "sample_002"]
    assert plan.samples[0].speaker_1.archive_path.endswith("ID001.wav")
    assert plan.samples[0].speaker_2.archive_path.endswith("ID002.wav")
    assert plan.samples[0].speaker_1.wave_header.duration_seconds == pytest.approx(1.0)


def test_apply_preparation_writes_only_generic_sample_files(tmp_path: Path) -> None:
    archive_path = write_archive(tmp_path, include_second_pair=True)
    target_root = tmp_path / "data" / "dataset_2" / "samples"
    restricted_record_path = tmp_path / "restricted" / "record.json"

    plan = apply_preparation(archive_path, target_root, restricted_record_path)

    verify_prepared_layout(target_root, plan)
    assert sorted(path.name for path in target_root.iterdir()) == ["sample_001", "sample_002"]
    assert sorted(path.name for path in (target_root / "sample_001").iterdir()) == [
        "metadata.json",
        "speaker_1.wav",
        "speaker_2.wav",
    ]
    assert json.loads((target_root / "sample_001" / "metadata.json").read_text()) == {
        "name": "sample_001",
        "speaker1_audioURL": "speaker_1.wav",
        "speaker2_audioURL": "speaker_2.wav",
    }
    record = json.loads(restricted_record_path.read_text())
    assert record["samples"][0]["sample_id"] == "sample_001"


def test_apply_preparation_rejects_nonempty_target(tmp_path: Path) -> None:
    archive_path = write_archive(tmp_path, include_second_pair=False)
    target_root = tmp_path / "data" / "dataset_2" / "samples"
    target_root.mkdir(parents=True)
    (target_root / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        apply_preparation(archive_path, target_root, tmp_path / "restricted.json")


def test_build_plan_rejects_unpaired_audio(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("WAV/Conversation_0_ID001.wav", wave_bytes(16_000))

    with pytest.raises(ValueError, match="exactly two tracks"):
        build_preparation_plan(archive_path)


def write_archive(tmp_path: Path, include_second_pair: bool) -> Path:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("WAV/Conversation_0_ID001.wav", wave_bytes(16_000))
        archive.writestr("WAV/Conversation_0_ID002.wav", wave_bytes(16_000))
        if include_second_pair:
            archive.writestr("WAV/Conversation_1_ID003.wav", wave_bytes(32_000))
            archive.writestr("WAV/Conversation_1_ID004.wav", wave_bytes(32_000))
        archive.writestr("TXT/sensitive.txt", "do not copy")
    return archive_path


def wave_bytes(frame_count: int) -> bytes:
    # wave only accepts a filesystem path or a binary stream; the latter avoids test files.
    from io import BytesIO

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(16_000)
        wave_file.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()
