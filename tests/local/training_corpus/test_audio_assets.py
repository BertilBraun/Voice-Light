from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from app.local.training_corpus.audio_assets import (
    CorpusAudioAssetCatalog,
    load_audio_asset_catalog,
)
from app.local.training_corpus.audio_staging import (
    AUDIO_STAGING_SCHEMA_VERSION,
    CorpusAudioAsset,
    CorpusAudioPreparation,
    CorpusAudioStagingManifest,
    CorpusAudioVerification,
    LocalSourceAudio,
    RemoteSourceAudio,
)
from app.shared.quality import AudioMetadata, SpeakerSide

SOURCE_HASH = "1" * 64
CORPUS_HASH = "2" * 64
GENERATED_AT = datetime(2026, 8, 16, tzinfo=UTC)
AUDIO_METADATA = AudioMetadata(
    duration_seconds=10.0,
    sample_rate=16_000,
    channels=1,
    sample_count=160_000,
)


def test_catalog_resolves_verified_source_to_derived_flac(tmp_path: Path) -> None:
    source_path = tmp_path / "data" / "dataset_2" / "samples" / "sample_001" / "speaker_1.wav"
    source_path.parent.mkdir(parents=True)
    source_path.touch()
    catalog = CorpusAudioAssetCatalog(
        manifests=(staging_manifest(tmp_path, audio_asset(source_path)),)
    )

    resolved = resolve(catalog, source_path)

    assert resolved.corpus_relative_path == PurePosixPath(
        "dataset_2/samples/sample_001/speaker_1.flac"
    )
    assert resolved.source_sha256 == SOURCE_HASH
    assert resolved.corpus_sha256 == CORPUS_HASH


@pytest.mark.parametrize(
    ("source_path_name", "source_hash", "source_audio", "error"),
    (
        ("different.wav", SOURCE_HASH, AUDIO_METADATA, "source path"),
        ("speaker_1.wav", "3" * 64, AUDIO_METADATA, "source hash"),
        (
            "speaker_1.wav",
            SOURCE_HASH,
            AUDIO_METADATA.model_copy(update={"sample_rate": 48_000}),
            "sample rate",
        ),
        (
            "speaker_1.wav",
            SOURCE_HASH,
            AUDIO_METADATA.model_copy(update={"sample_count": 159_999}),
            "sample count",
        ),
    ),
)
def test_catalog_rejects_database_source_mismatch(
    tmp_path: Path,
    source_path_name: str,
    source_hash: str,
    source_audio: AudioMetadata,
    error: str,
) -> None:
    source_path = tmp_path / "speaker_1.wav"
    source_path.touch()
    catalog = CorpusAudioAssetCatalog(
        manifests=(staging_manifest(tmp_path, audio_asset(source_path)),)
    )

    with pytest.raises(ValueError, match=error):
        resolve(
            catalog,
            tmp_path / source_path_name,
            source_sha256=source_hash,
            source_audio=source_audio,
        )


def test_catalog_rejects_missing_recording_asset(tmp_path: Path) -> None:
    source_path = tmp_path / "speaker_1.wav"
    source_path.touch()
    catalog = CorpusAudioAssetCatalog(
        manifests=(staging_manifest(tmp_path, audio_asset(source_path)),)
    )

    with pytest.raises(ValueError, match="has no asset"):
        resolve(catalog, source_path, external_id="sample_002")


def test_load_catalog_rejects_duplicate_dataset_manifests(tmp_path: Path) -> None:
    source_path = tmp_path / "speaker_1.wav"
    source_path.touch()
    manifest = staging_manifest(tmp_path, audio_asset(source_path))
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    second_path.write_text(manifest.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate dataset names"):
        load_audio_asset_catalog((first_path, second_path))


def resolve(
    catalog: CorpusAudioAssetCatalog,
    source_path: Path,
    source_sha256: str = SOURCE_HASH,
    source_audio: AudioMetadata = AUDIO_METADATA,
    external_id: str = "sample_001",
) -> CorpusAudioAsset:
    return catalog.resolve(
        dataset_name="dataset_2",
        external_id=external_id,
        side=SpeakerSide.SPEAKER1,
        source_uri=str(source_path),
        source_sha256=source_sha256,
        source_audio=source_audio,
    )


def audio_asset(source_path: Path) -> CorpusAudioAsset:
    return CorpusAudioAsset(
        sample_id="sample_001",
        side=SpeakerSide.SPEAKER1,
        source=LocalSourceAudio(path=source_path.resolve()),
        source_sha256=SOURCE_HASH,
        corpus_relative_path=PurePosixPath("dataset_2/samples/sample_001/speaker_1.flac"),
        corpus_sha256=CORPUS_HASH,
        source_audio=AUDIO_METADATA,
        corpus_audio=AUDIO_METADATA,
        preparation=CorpusAudioPreparation.LOSSLESS_FLAC_TRANSCODE,
        verification=CorpusAudioVerification.METADATA,
    )


def staging_manifest(
    root: Path,
    asset: CorpusAudioAsset,
) -> CorpusAudioStagingManifest:
    return CorpusAudioStagingManifest(
        schema_version=AUDIO_STAGING_SCHEMA_VERSION,
        generated_at=GENERATED_AT,
        dataset_name="dataset_2",
        assets=(asset,),
    )


def test_catalog_resolves_remote_source_uri(tmp_path: Path) -> None:
    source_uri = "s3://audio/meeting/speaker_1.flac"
    asset = audio_asset(tmp_path / "unused.wav").model_copy(
        update={"source": RemoteSourceAudio(uri=source_uri)}
    )
    catalog = CorpusAudioAssetCatalog(manifests=(staging_manifest(tmp_path, asset),))

    resolved = catalog.resolve(
        dataset_name="dataset_2",
        external_id="sample_001",
        side=SpeakerSide.SPEAKER1,
        source_uri=source_uri,
        source_sha256=SOURCE_HASH,
        source_audio=AUDIO_METADATA,
    )

    assert resolved == asset
