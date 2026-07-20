from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

import app.local.ingestion.local_audio as local_audio_module
from app.local.db.models import SampleTrackRecord, TrackSide
from app.local.ingestion.local_audio import materialize_sample_track
from app.shared.audio.s3 import CachedAudioFile, S3AudioSource
from app.shared.quality import AudioMetadata


class RecordingAudioCache:
    def __init__(self, cached: CachedAudioFile) -> None:
        self.cached = cached
        self.sources: list[S3AudioSource] = []

    def materialize(self, source: S3AudioSource) -> CachedAudioFile:
        self.sources.append(source)
        return self.cached


def test_local_path_is_returned_without_using_s3_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "speaker.wav"
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr(
        local_audio_module,
        "local_dataset_audio_cache",
        lambda: pytest.fail("local audio must not use the S3 cache"),
    )

    result = materialize_sample_track(track_record(audio_path.as_posix()))

    assert result == audio_path.resolve()


def test_host_data_path_is_remapped_to_container_data_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    audio_path = data_root / "luel" / "sessions" / "pmt_001" / "speaker.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr(local_audio_module, "DATA_ROOT", data_root)

    result = materialize_sample_track(
        track_record("C:\\Projects\\Voice-Light\\data\\luel\\sessions\\pmt_001\\speaker.wav")
    )

    assert result == audio_path.resolve()


def test_s3_track_is_materialized_from_persisted_object_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_path = tmp_path / "cache" / "speaker.flac"
    expected_sha256 = "a" * 64
    source = S3AudioSource(
        uri="s3://meetings/sample/speaker.flac",
        size_bytes=1234,
        etag='"etag"',
        duration_seconds=12.5,
    )
    cache = RecordingAudioCache(
        CachedAudioFile(
            source=source,
            path=cached_path,
            content_sha256=expected_sha256,
            audio_metadata=AudioMetadata(
                duration_seconds=12.5,
                sample_rate=48_000,
                channels=1,
                sample_count=600_000,
            ),
        )
    )
    monkeypatch.setattr(local_audio_module, "local_dataset_audio_cache", lambda: cache)

    result = materialize_sample_track(
        track_record(
            source.uri,
            audio_sha256=expected_sha256,
            source_size_bytes=source.size_bytes,
            source_etag=source.etag,
            duration_seconds=source.duration_seconds,
        )
    )

    assert result == cached_path
    assert cache.sources == [source]


def test_s3_track_rejects_cached_content_that_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = S3AudioSource(
        uri="s3://meetings/sample/speaker.flac",
        size_bytes=1234,
        etag="etag",
        duration_seconds=12.5,
    )
    cache = RecordingAudioCache(
        CachedAudioFile(
            source=source,
            path=tmp_path / "speaker.flac",
            content_sha256="b" * 64,
            audio_metadata=AudioMetadata(
                duration_seconds=12.5,
                sample_rate=48_000,
                channels=1,
                sample_count=600_000,
            ),
        )
    )
    monkeypatch.setattr(local_audio_module, "local_dataset_audio_cache", lambda: cache)

    with pytest.raises(ValueError, match="does not match track provenance"):
        materialize_sample_track(
            track_record(
                source.uri,
                audio_sha256="a" * 64,
                source_size_bytes=source.size_bytes,
                source_etag=source.etag,
                duration_seconds=source.duration_seconds,
            )
        )


def track_record(
    access_uri: str,
    *,
    audio_sha256: str | None = None,
    source_size_bytes: int | None = None,
    source_etag: str | None = None,
    duration_seconds: float | None = 1.0,
) -> SampleTrackRecord:
    now = datetime.now(tz=UTC)
    return SampleTrackRecord(
        id=uuid4(),
        sample_id=uuid4(),
        side=TrackSide.SPEAKER1,
        speaker_index=1,
        storage_uri=access_uri,
        access_uri=access_uri,
        duration_seconds=duration_seconds,
        sample_rate=None,
        channels=None,
        sample_count=None,
        audio_sha256=audio_sha256,
        source_size_bytes=source_size_bytes,
        source_etag=source_etag,
        created_at=now,
        updated_at=now,
    )
