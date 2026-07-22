from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.local.db.models import SampleRecord, SampleTrackRecord
from app.local.db.repository import AudioMetadataInput, SampleTrackInput
from app.local.ingestion.manifest import ManifestDiscoveredSample
from app.local.ingestion.manifest_service import (
    register_manifest_sample,
    select_manifest_duration,
    update_materialized_track,
)
from app.shared.audio.s3 import S3AudioSource
from app.shared.compute_api import MaterializedAudio
from app.shared.quality import AudioMetadata


class RecordingManifestRepository:
    def __init__(self) -> None:
        self.sample_id = uuid4()
        self.track_inputs: list[SampleTrackInput] = []
        self.audio_metadata_inputs: list[AudioMetadataInput] = []

    def upsert_sample(self, dataset_id: UUID, external_id: str) -> SampleRecord:
        now = datetime.now(tz=UTC)
        return SampleRecord(
            id=self.sample_id,
            dataset_id=dataset_id,
            external_id=external_id,
            duration_seconds=None,
            quality_score=None,
            quality_flags=(),
            is_unusable=False,
            created_at=now,
            updated_at=now,
        )

    def update_sample_duration(
        self,
        sample_id: UUID,
        duration_seconds: float,
    ) -> SampleRecord:
        assert sample_id == self.sample_id
        assert duration_seconds > 0.0
        return self.upsert_sample(uuid4(), "sample")

    def upsert_sample_track(
        self,
        sample_id: UUID,
        track: SampleTrackInput,
    ) -> SampleTrackRecord:
        assert sample_id == self.sample_id
        self.track_inputs.append(track)
        now = datetime.now(tz=UTC)
        return SampleTrackRecord(
            id=uuid4(),
            sample_id=sample_id,
            side=track.side,
            speaker_index=track.speaker_index,
            storage_uri=track.storage_uri,
            access_uri=track.access_uri,
            duration_seconds=track.duration_seconds,
            sample_rate=track.sample_rate,
            channels=track.channels,
            sample_count=track.sample_count,
            audio_sha256=track.audio_sha256,
            source_size_bytes=track.source_size_bytes,
            source_etag=track.source_etag,
            created_at=now,
            updated_at=now,
        )

    def upsert_audio_metadata(self, metadata: AudioMetadataInput) -> None:
        self.audio_metadata_inputs.append(metadata)


def test_manifest_registration_persists_identity_without_downloading_audio() -> None:
    repository = RecordingManifestRepository()
    discovered = discovered_sample("sample-one", duration_seconds=20.0)

    registered = register_manifest_sample(repository, uuid4(), discovered)

    assert registered.sample_id == repository.sample_id
    assert [track.audio_sha256 for track in repository.track_inputs] == [None, None]
    assert [track.source_size_bytes for track in repository.track_inputs] == [1001, 1002]
    assert [track.source_etag for track in repository.track_inputs] == ["etag-1", "etag-2"]
    assert [track.access_uri for track in repository.track_inputs] == [
        discovered.speaker1.uri,
        discovered.speaker2.uri,
    ]
    assert repository.audio_metadata_inputs == []


def test_materialized_track_adds_content_provenance_and_audio_metadata() -> None:
    repository = RecordingManifestRepository()
    registered = register_manifest_sample(
        repository,
        uuid4(),
        discovered_sample("sample-one", duration_seconds=20.0),
    )
    source = registered.discovered.speaker1
    audio_metadata = AudioMetadata(
        duration_seconds=20.0,
        sample_rate=48_000,
        channels=1,
        sample_count=960_000,
    )

    updated = update_materialized_track(
        repository=repository,
        existing=registered.speaker1_track,
        audio=MaterializedAudio(
            source=source,
            filename="speaker1.flac",
            content_sha256="a" * 64,
            metadata=audio_metadata,
        ),
    )

    assert updated.audio_sha256 == "a" * 64
    assert updated.sample_rate == 48_000
    assert repository.audio_metadata_inputs[-1].payload == audio_metadata.model_dump()


def test_manifest_duration_limit_selects_whole_samples() -> None:
    samples = tuple(
        discovered_sample(f"sample-{index}", duration_seconds=duration_seconds)
        for index, duration_seconds in enumerate((1800.0, 1800.0, 60.0), start=1)
    )

    selected = select_manifest_duration(samples, max_duration_hours=1.0)

    assert tuple(sample.external_id for sample in selected) == ("sample-1", "sample-2")


def discovered_sample(
    external_id: str,
    duration_seconds: float,
) -> ManifestDiscoveredSample:
    return ManifestDiscoveredSample(
        external_id=external_id,
        speaker1=S3AudioSource(
            uri=f"s3://meetings/{external_id}/speaker1.flac",
            size_bytes=1001,
            etag="etag-1",
            duration_seconds=duration_seconds,
        ),
        speaker2=S3AudioSource(
            uri=f"s3://meetings/{external_id}/speaker2.flac",
            size_bytes=1002,
            etag="etag-2",
            duration_seconds=duration_seconds,
        ),
        duration_seconds=duration_seconds,
    )
