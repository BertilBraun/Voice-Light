from __future__ import annotations

import wave
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from app.shared.audio.s3 import S3AudioCache, S3AudioSource, parse_s3_uri, source_cache_key


class RecordingDownloader:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.requests: list[tuple[str, str]] = []

    def download_fileobj(
        self,
        bucket: str,
        key: str,
        file_object: BinaryIO,
    ) -> None:
        self.requests.append((bucket, key))
        file_object.write(self.content)


def test_s3_audio_cache_downloads_immutable_uri_once(tmp_path: Path) -> None:
    content = wave_bytes()
    downloader = RecordingDownloader(content)
    cache = S3AudioCache(root=tmp_path / "cache", downloader=downloader)
    source = S3AudioSource(
        uri="s3://meeting-audio/samples/one/speaker.flac",
        size_bytes=len(content),
        etag='"immutable-etag"',
        duration_seconds=1.0,
    )

    first = cache.materialize(source)
    second = cache.materialize(source)

    assert first.path == second.path
    assert first.content_sha256 == second.content_sha256
    assert first.audio_metadata.duration_seconds == 1.0
    assert first.path.read_bytes() == content
    assert downloader.requests == [("meeting-audio", "samples/one/speaker.flac")]


def test_existing_cache_survives_new_cache_instance(tmp_path: Path) -> None:
    content = wave_bytes()
    source = S3AudioSource(
        uri="s3://meeting-audio/samples/two/speaker.wav",
        size_bytes=len(content),
        etag="etag",
        duration_seconds=1.0,
    )
    first_downloader = RecordingDownloader(content)
    S3AudioCache(tmp_path, first_downloader).materialize(source)
    second_downloader = RecordingDownloader(b"must not be downloaded")

    cached = S3AudioCache(tmp_path, second_downloader).materialize(source)

    assert cached.path.is_file()
    assert first_downloader.requests == [("meeting-audio", "samples/two/speaker.wav")]
    assert second_downloader.requests == []


def test_s3_uri_and_cache_key_are_deterministic() -> None:
    location = parse_s3_uri("s3://bucket-name/path/to/audio.wav")

    assert location.bucket == "bucket-name"
    assert location.key == "path/to/audio.wav"
    assert source_cache_key("s3://bucket-name/path/to/audio.wav") == source_cache_key(
        "s3://bucket-name/path/to/audio.wav"
    )


def wave_bytes() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * 16_000)
    return output.getvalue()
