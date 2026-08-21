from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID

import pytest

from app.local.db.models import (
    DashboardSample,
    SampleRecord,
    SampleTrackRecord,
    TrackSide,
)
from app.local.training_corpus.audio_staging import (
    CorpusAudioPreparation,
    CorpusAudioVerification,
    LocalSourceAudio,
    RemoteSourceAudio,
)
from app.local.training_corpus.hub_audio_assets import (
    HubLfsAudioFile,
    build_dataset_1_hub_audio_manifest,
    build_meetings_hub_audio_manifest,
)
from app.shared.quality import SpeakerSide

GENERATED_AT = datetime(2026, 8, 21, tzinfo=UTC)
CREATED_AT = datetime(2026, 7, 1, tzinfo=UTC)
SPEAKER1_HASH = "1" * 64
SPEAKER2_HASH = "2" * 64


def test_dataset_1_maps_external_id_to_existing_hub_paths(tmp_path: Path) -> None:
    sample = dashboard_sample(
        external_id="sample_320",
        speaker1_uri=str(tmp_path / "speaker_1.wav"),
        speaker2_uri=str(tmp_path / "speaker_2.wav"),
    )
    inventory = (
        hub_file("dataset_1/samples/sample_001/speaker_1.flac", "a" * 64),
        hub_file("dataset_1/samples/sample_001/speaker_2.flac", "b" * 64),
    )

    manifest = build_dataset_1_hub_audio_manifest(
        (sample,), ("sample_320",), inventory, GENERATED_AT
    )

    assert manifest.dataset_name == "dataset_1-local"
    assert tuple(asset.corpus_relative_path for asset in manifest.assets) == (
        PurePosixPath("dataset_1/samples/sample_001/speaker_1.flac"),
        PurePosixPath("dataset_1/samples/sample_001/speaker_2.flac"),
    )
    assert tuple(asset.corpus_sha256 for asset in manifest.assets) == ("a" * 64, "b" * 64)
    assert isinstance(manifest.assets[0].source, LocalSourceAudio)
    assert all(
        asset.preparation is CorpusAudioPreparation.EXISTING_HUB_FLAC
        and asset.verification is CorpusAudioVerification.HUB_LFS_SHA256
        for asset in manifest.assets
    )


def test_dataset_1_rejects_missing_expected_path(tmp_path: Path) -> None:
    sample = dashboard_sample(
        external_id="sample_320",
        speaker1_uri=str(tmp_path / "speaker_1.wav"),
        speaker2_uri=str(tmp_path / "speaker_2.wav"),
    )

    with pytest.raises(ValueError, match="missing expected file.*speaker_2.flac"):
        build_dataset_1_hub_audio_manifest(
            (sample,),
            ("sample_320",),
            (hub_file("dataset_1/samples/sample_001/speaker_1.flac", "a" * 64),),
            GENERATED_AT,
        )


def test_meetings_match_database_hashes_to_hub_sample_directory() -> None:
    sample = dashboard_sample(
        external_id="meeting-7012",
        speaker1_uri="s3://meetings/meeting-7012/speaker_1.flac",
        speaker2_uri="s3://meetings/meeting-7012/speaker_2.flac",
    )
    inventory = (
        hub_file("dataset_4/samples/sample_111/speaker_2.flac", SPEAKER2_HASH),
        hub_file("dataset_4/samples/sample_111/speaker_1.flac", SPEAKER1_HASH),
    )

    manifest = build_meetings_hub_audio_manifest((sample,), inventory, GENERATED_AT)

    assert manifest.dataset_name == "meetings-s3"
    assert tuple(asset.sample_id for asset in manifest.assets) == (
        "meeting-7012",
        "meeting-7012",
    )
    assert tuple(asset.side for asset in manifest.assets) == (
        SpeakerSide.SPEAKER1,
        SpeakerSide.SPEAKER2,
    )
    assert tuple(asset.corpus_relative_path for asset in manifest.assets) == (
        PurePosixPath("dataset_4/samples/sample_111/speaker_1.flac"),
        PurePosixPath("dataset_4/samples/sample_111/speaker_2.flac"),
    )
    assert isinstance(manifest.assets[0].source, RemoteSourceAudio)


def test_meetings_reject_ambiguous_lfs_hash() -> None:
    sample = dashboard_sample(
        external_id="meeting-7012",
        speaker1_uri="s3://meetings/meeting-7012/speaker_1.flac",
        speaker2_uri="s3://meetings/meeting-7012/speaker_2.flac",
    )
    inventory = (
        hub_file("dataset_4/samples/sample_111/speaker_1.flac", SPEAKER1_HASH),
        hub_file("dataset_4/samples/sample_112/speaker_1.flac", SPEAKER1_HASH),
        hub_file("dataset_4/samples/sample_111/speaker_2.flac", SPEAKER2_HASH),
    )

    with pytest.raises(ValueError, match="duplicate SHA-256"):
        build_meetings_hub_audio_manifest((sample,), inventory, GENERATED_AT)


def test_meetings_reject_tracks_from_different_hub_samples() -> None:
    sample = dashboard_sample(
        external_id="meeting-7012",
        speaker1_uri="s3://meetings/meeting-7012/speaker_1.flac",
        speaker2_uri="s3://meetings/meeting-7012/speaker_2.flac",
    )
    inventory = (
        hub_file("dataset_4/samples/sample_111/speaker_1.flac", SPEAKER1_HASH),
        hub_file("dataset_4/samples/sample_112/speaker_2.flac", SPEAKER2_HASH),
    )

    with pytest.raises(ValueError, match="different Hub samples"):
        build_meetings_hub_audio_manifest((sample,), inventory, GENERATED_AT)


def test_builder_rejects_incomplete_database_audio_metadata(tmp_path: Path) -> None:
    sample = dashboard_sample(
        external_id="sample_320",
        speaker1_uri=str(tmp_path / "speaker_1.wav"),
        speaker2_uri=str(tmp_path / "speaker_2.wav"),
    )
    incomplete_track = sample.tracks[0].model_copy(update={"sample_count": None})
    sample = sample.model_copy(update={"tracks": (incomplete_track, sample.tracks[1])})
    inventory = (
        hub_file("dataset_1/samples/sample_001/speaker_1.flac", "a" * 64),
        hub_file("dataset_1/samples/sample_001/speaker_2.flac", "b" * 64),
    )

    with pytest.raises(ValueError, match="incomplete audio metadata"):
        build_dataset_1_hub_audio_manifest((sample,), ("sample_320",), inventory, GENERATED_AT)


def hub_file(path: str, sha256: str) -> HubLfsAudioFile:
    return HubLfsAudioFile(path=PurePosixPath(path), size_bytes=100, sha256=sha256)


def dashboard_sample(
    external_id: str,
    speaker1_uri: str,
    speaker2_uri: str,
) -> DashboardSample:
    sample_id = UUID("00000000-0000-0000-0000-000000000001")
    sample = SampleRecord(
        id=sample_id,
        dataset_id=UUID("00000000-0000-0000-0000-000000000002"),
        external_id=external_id,
        duration_seconds=10.0,
        quality_score=0.99,
        quality_flags=(),
        is_unusable=False,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )
    return DashboardSample(
        sample=sample,
        tracks=(
            track(sample_id, TrackSide.SPEAKER1, speaker1_uri, SPEAKER1_HASH, 1),
            track(sample_id, TrackSide.SPEAKER2, speaker2_uri, SPEAKER2_HASH, 2),
        ),
        latest_quality=None,
        latest_asr_run=None,
        latest_asr_evaluation=None,
        language_assessments=(),
    )


def track(
    sample_id: UUID,
    side: TrackSide,
    access_uri: str,
    audio_sha256: str,
    speaker_index: int,
) -> SampleTrackRecord:
    return SampleTrackRecord(
        id=UUID(f"00000000-0000-0000-0000-00000000001{speaker_index}"),
        sample_id=sample_id,
        side=side,
        speaker_index=speaker_index,
        storage_uri=access_uri,
        access_uri=access_uri,
        duration_seconds=10.0,
        sample_rate=16_000,
        channels=1,
        sample_count=160_000,
        audio_sha256=audio_sha256,
        source_size_bytes=320_000,
        source_etag=None,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )
