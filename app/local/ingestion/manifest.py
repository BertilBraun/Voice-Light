from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from app.local.ingestion.discovery import track_durations_are_compatible
from app.shared.audio.s3 import S3AudioSource
from app.shared.base_model import FrozenBaseModel


class ManifestConnection(FrozenBaseModel):
    provider: str
    bucket: str
    region: str
    prefix: str
    s3_uri: str
    credential_mode: str


class ManifestFile(FrozenBaseModel):
    key: str
    s3_uri: str
    size_bytes: int = Field(gt=0)
    etag: str
    last_modified: str
    extension: str
    duration_seconds: float = Field(gt=0.0)
    bytes_per_second: float = Field(gt=0.0)

    def audio_source(self) -> S3AudioSource:
        return S3AudioSource(
            uri=self.s3_uri,
            size_bytes=self.size_bytes,
            etag=self.etag,
            duration_seconds=self.duration_seconds,
        )


class ManifestSample(FrozenBaseModel):
    sample_path: str
    sample_s3_uri: str
    file_count: int
    total_bytes: int
    duration_seconds: float = Field(gt=0.0)
    duration_delta_seconds: float
    files: tuple[ManifestFile, ...]


class MeetingsManifest(FrozenBaseModel):
    schema_version: int
    connection: ManifestConnection
    sample_count: int
    file_count: int
    total_bytes: int
    total_duration_seconds: float
    samples: tuple[ManifestSample, ...]


@dataclass(frozen=True)
class ManifestDiscoveredSample:
    external_id: str
    speaker1: S3AudioSource
    speaker2: S3AudioSource
    duration_seconds: float


def read_meetings_manifest(path: Path) -> MeetingsManifest:
    if not path.is_file():
        raise ValueError(f"Manifest file does not exist: {path}")
    manifest = MeetingsManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.sample_count != len(manifest.samples):
        raise ValueError("Manifest sample count does not match its sample array.")
    return manifest


def discover_manifest_samples(
    manifest: MeetingsManifest,
) -> tuple[ManifestDiscoveredSample, ...]:
    discovered: list[ManifestDiscoveredSample] = []
    for sample in manifest.samples:
        if sample.file_count != 2 or len(sample.files) != 2:
            raise ValueError(f"Manifest sample must contain two audio files: {sample.sample_path}")
        speaker1_file, speaker2_file = sorted(sample.files, key=lambda file: file.key)
        if not track_durations_are_compatible(
            speaker1_duration_seconds=speaker1_file.duration_seconds,
            speaker2_duration_seconds=speaker2_file.duration_seconds,
        ):
            continue
        discovered.append(
            ManifestDiscoveredSample(
                external_id=sample.sample_path,
                speaker1=speaker1_file.audio_source(),
                speaker2=speaker2_file.audio_source(),
                duration_seconds=max(
                    speaker1_file.duration_seconds,
                    speaker2_file.duration_seconds,
                ),
            )
        )
    return tuple(discovered)
