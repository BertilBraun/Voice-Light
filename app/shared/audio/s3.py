from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import BinaryIO, Protocol
from urllib.parse import urlparse

import boto3
from mypy_boto3_s3 import S3Client
from pydantic import Field

from app.shared.audio.loading import probe_local_audio_metadata
from app.shared.audio.transport import sha256_file
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import AudioMetadata


class S3AudioSource(FrozenBaseModel):
    uri: str = Field(pattern=r"^s3://[^/]+/.+")
    size_bytes: int = Field(gt=0)
    etag: str
    duration_seconds: float = Field(gt=0.0)


class CachedAudioMetadata(FrozenBaseModel):
    source_uri: str
    size_bytes: int
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio: AudioMetadata


@dataclass(frozen=True)
class CachedAudioFile:
    source: S3AudioSource
    path: Path
    content_sha256: str
    audio_metadata: AudioMetadata


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str


class S3Downloader(Protocol):
    def download_fileobj(
        self,
        bucket: str,
        key: str,
        file_object: BinaryIO,
    ) -> None: ...


class LazyS3Downloader:
    def __init__(self) -> None:
        self.client_creation_lock = Lock()
        self.client: S3Client | None = None

    def download_fileobj(
        self,
        bucket: str,
        key: str,
        file_object: BinaryIO,
    ) -> None:
        self.s3_client().download_fileobj(bucket, key, file_object)

    def s3_client(self) -> S3Client:
        with self.client_creation_lock:
            if self.client is None:
                self.client = boto3.client("s3")
            return self.client


class S3AudioCache:
    def __init__(self, root: Path, downloader: S3Downloader) -> None:
        self.root = root.resolve()
        self.downloader = downloader
        self.locks_guard = Lock()
        self.source_locks: dict[str, Lock] = {}

    def materialize(self, source: S3AudioSource) -> CachedAudioFile:
        cache_key = source_cache_key(source.uri)
        target_path = self.cache_path(cache_key=cache_key, source_uri=source.uri)
        metadata_path = target_path.with_suffix(f"{target_path.suffix}.json")
        with self.source_lock(cache_key):
            cached = read_cached_audio(
                source=source,
                target_path=target_path,
                metadata_path=metadata_path,
            )
            if cached is not None:
                return cached
            return self.download(
                source=source,
                target_path=target_path,
                metadata_path=metadata_path,
            )

    def cache_path(self, cache_key: str, source_uri: str) -> Path:
        source_suffix = PurePosixPath(urlparse(source_uri).path).suffix.lower()
        suffix = source_suffix if source_suffix else ".audio"
        return self.root / cache_key[:2] / f"{cache_key}{suffix}"

    def source_lock(self, cache_key: str) -> Lock:
        with self.locks_guard:
            existing = self.source_locks.get(cache_key)
            if existing is not None:
                return existing
            created = Lock()
            self.source_locks[cache_key] = created
            return created

    def download(
        self,
        source: S3AudioSource,
        target_path: Path,
        metadata_path: Path,
    ) -> CachedAudioFile:
        location = parse_s3_uri(source.uri)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = tempfile.NamedTemporaryFile(
            prefix=f"{target_path.stem}.",
            suffix=".download",
            dir=target_path.parent,
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        try:
            with temporary_file:
                self.downloader.download_fileobj(
                    location.bucket,
                    location.key,
                    temporary_file,
                )
            actual_size_bytes = temporary_path.stat().st_size
            if actual_size_bytes != source.size_bytes:
                raise ValueError(
                    f"S3 audio size mismatch for {source.uri}: "
                    f"expected {source.size_bytes}, downloaded {actual_size_bytes}."
                )
            audio_metadata = probe_local_audio_metadata(temporary_path)
            content_sha256 = sha256_file(temporary_path)
            temporary_path.replace(target_path)
            write_cached_metadata(
                metadata_path=metadata_path,
                metadata=CachedAudioMetadata(
                    source_uri=source.uri,
                    size_bytes=source.size_bytes,
                    content_sha256=content_sha256,
                    audio=audio_metadata,
                ),
            )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return CachedAudioFile(
            source=source,
            path=target_path,
            content_sha256=content_sha256,
            audio_metadata=audio_metadata,
        )


def default_s3_downloader() -> LazyS3Downloader:
    return LazyS3Downloader()


def parse_s3_uri(uri: str) -> S3Location:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return S3Location(bucket=parsed.netloc, key=parsed.path.lstrip("/"))


def source_cache_key(uri: str) -> str:
    return hashlib.sha256(uri.encode("utf-8")).hexdigest()


def read_cached_audio(
    source: S3AudioSource,
    target_path: Path,
    metadata_path: Path,
) -> CachedAudioFile | None:
    if not target_path.is_file() or target_path.stat().st_size != source.size_bytes:
        return None
    if metadata_path.is_file():
        metadata = CachedAudioMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.source_uri != source.uri or metadata.size_bytes != source.size_bytes:
            return None
    else:
        metadata = CachedAudioMetadata(
            source_uri=source.uri,
            size_bytes=source.size_bytes,
            content_sha256=sha256_file(target_path),
            audio=probe_local_audio_metadata(target_path),
        )
        write_cached_metadata(metadata_path=metadata_path, metadata=metadata)
    return CachedAudioFile(
        source=source,
        path=target_path,
        content_sha256=metadata.content_sha256,
        audio_metadata=metadata.audio,
    )


def write_cached_metadata(metadata_path: Path, metadata: CachedAudioMetadata) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix=f"{metadata_path.stem}.",
        suffix=".json",
        dir=metadata_path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    try:
        with temporary_file:
            temporary_file.write(metadata.model_dump_json())
        temporary_path.replace(metadata_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
