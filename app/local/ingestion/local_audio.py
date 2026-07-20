from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PurePosixPath

from app.local.config import DATA_ROOT, LOCAL_DATASET_AUDIO_CACHE_ROOT
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
    direct_path = Path(track.access_uri).resolve()
    if direct_path.is_file():
        return direct_path
    mounted_path = _mounted_data_path(track.access_uri)
    if mounted_path is not None and mounted_path.is_file():
        return mounted_path
    raise ValueError(f"Audio file does not exist: {direct_path}")


def _mounted_data_path(access_uri: str) -> Path | None:
    normalized_parts = PurePosixPath(access_uri.replace("\\", "/")).parts
    data_indexes = [
        index for index, part in enumerate(normalized_parts) if part.casefold() == "data"
    ]
    if not data_indexes:
        return None
    relative_parts = normalized_parts[data_indexes[-1] + 1 :]
    if not relative_parts:
        return None
    data_root = DATA_ROOT.resolve()
    candidate = data_root.joinpath(*relative_parts).resolve()
    if not candidate.is_relative_to(data_root):
        raise ValueError(f"Local audio path escapes the configured data root: {access_uri}")
    return candidate


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
