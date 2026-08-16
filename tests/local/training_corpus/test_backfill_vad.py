from __future__ import annotations

import wave
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.local.db.models import SampleTrackRecord, TrackSide
from app.local.training_corpus.backfill_vad import (
    VAD_VERSION,
    detect_pair_vad,
    required_audio_sha256,
)
from app.shared.quality import SpeakerSide


def test_detect_pair_vad_reconstructs_versioned_two_track_evidence(tmp_path: Path) -> None:
    speaker1_path = tmp_path / "speaker_1.wav"
    speaker2_path = tmp_path / "speaker_2.wav"
    write_activity_wave(speaker1_path, active_start_frame=3_200, active_end_frame=8_000)
    write_activity_wave(speaker2_path, active_start_frame=9_600, active_end_frame=14_400)

    speaker1_vad, speaker2_vad = detect_pair_vad(
        sample_track(speaker1_path, TrackSide.SPEAKER1),
        sample_track(speaker2_path, TrackSide.SPEAKER2),
    )

    assert VAD_VERSION == "energy-pair-v1"
    assert speaker1_vad.side is SpeakerSide.SPEAKER1
    assert speaker2_vad.side is SpeakerSide.SPEAKER2
    assert speaker1_vad.speech_segments
    assert speaker2_vad.speech_segments
    assert speaker1_vad.speech_segments[0].end_seconds <= 0.6
    assert speaker2_vad.speech_segments[0].start_seconds >= 0.5


def test_required_audio_sha256_rejects_unidentified_source(tmp_path: Path) -> None:
    track = sample_track(tmp_path / "missing.wav", TrackSide.SPEAKER1, audio_sha256=None)

    with pytest.raises(ValueError, match="has no source audio SHA-256"):
        required_audio_sha256(track)


def sample_track(
    path: Path,
    side: TrackSide,
    audio_sha256: str | None = "a" * 64,
) -> SampleTrackRecord:
    timestamp = datetime.now(UTC)
    return SampleTrackRecord(
        id=uuid4(),
        sample_id=uuid4(),
        side=side,
        speaker_index=1 if side is TrackSide.SPEAKER1 else 2,
        storage_uri=path.as_posix(),
        access_uri=path.as_posix(),
        duration_seconds=1.0,
        sample_rate=16_000,
        channels=1,
        sample_count=16_000,
        audio_sha256=audio_sha256,
        created_at=timestamp,
        updated_at=timestamp,
    )


def write_activity_wave(
    path: Path,
    active_start_frame: int,
    active_end_frame: int,
) -> None:
    frames = bytearray()
    for frame_index in range(16_000):
        amplitude = 4_000 if active_start_frame <= frame_index < active_end_frame else 0
        frames.extend(amplitude.to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(16_000)
        wave_file.writeframes(frames)
