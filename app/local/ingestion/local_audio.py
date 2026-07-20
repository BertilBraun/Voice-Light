from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.local.config import LOCAL_DATASET_AUDIO_CACHE_ROOT
from app.local.db.models import SampleTrackRecord
from app.shared.audio.s3 import S3AudioCache, S3AudioSource, default_s3_downloader


@lru_cache(maxsize=1)
def local_dataset_audio_cache() -> S3AudioCache:
    return S3AudioCache(
        root=LOCAL_DATASET_AUDIO_CACHE_ROOT,
        downloader=default_s3_downloader(),
    )


def materialize_sample_track(track: SampleTrackRecord) -> Path:
    if track.access_uri.startswith("s3://"):
        return materialize_s3_track(track)
    if "://" in track.access_uri:
        raise ValueError(f"Unsupported track access URI: {track.access_uri}")
    path = Path(track.access_uri).resolve()
    if not path.is_file():
        raise ValueError(f"Audio file does not exist: {path}")
    return path


def materialize_s3_track(track: SampleTrackRecord) -> Path:
    if track.source_size_bytes is None:
        raise ValueError(f"S3 track is missing source_size_bytes: {track.id}")
    if track.source_etag is None:
        raise ValueError(f"S3 track is missing source_etag: {track.id}")
    if track.duration_seconds is None:
        raise ValueError(f"S3 track is missing duration_seconds: {track.id}")
    cached = local_dataset_audio_cache().materialize(
        S3AudioSource(
            uri=track.access_uri,
            size_bytes=track.source_size_bytes,
            etag=track.source_etag,
            duration_seconds=track.duration_seconds,
        )
    )
    if track.audio_sha256 is not None and cached.content_sha256 != track.audio_sha256:
        raise ValueError(f"Cached S3 audio content does not match track provenance: {track.id}")
    return cached.path
