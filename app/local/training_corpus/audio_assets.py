from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.local.training_corpus.audio_staging import (
    CorpusAudioAsset,
    CorpusAudioStagingManifest,
    LocalSourceAudio,
    RemoteSourceAudio,
)
from app.shared.quality import AudioMetadata, SpeakerSide


@dataclass(frozen=True)
class CorpusAudioAssetCatalog:
    manifests: tuple[CorpusAudioStagingManifest, ...]

    def corpus_dataset_directory(self, dataset_name: str) -> str:
        manifests = tuple(
            manifest for manifest in self.manifests if manifest.dataset_name == dataset_name
        )
        if len(manifests) != 1:
            raise ValueError(f"Expected exactly one audio manifest for {dataset_name}.")
        directories = {asset.corpus_relative_path.parts[0] for asset in manifests[0].assets}
        if len(directories) != 1:
            raise ValueError(f"Audio assets for {dataset_name} span multiple corpus directories.")
        return next(iter(directories))

    def resolve(
        self,
        dataset_name: str,
        external_id: str,
        side: SpeakerSide,
        source_uri: str,
        source_sha256: str,
        source_audio: AudioMetadata,
    ) -> CorpusAudioAsset:
        matches = tuple(
            asset
            for manifest in self.manifests
            if manifest.dataset_name == dataset_name
            for asset in manifest.assets
            if asset.sample_id == external_id and asset.side is side
        )
        if not matches:
            raise ValueError(
                f"Audio build manifest has no asset for {dataset_name}/{external_id}/{side.value}."
            )
        if len(matches) != 1:
            raise ValueError(
                "Audio build manifests contain duplicate assets for "
                f"{dataset_name}/{external_id}/{side.value}."
            )
        asset = matches[0]
        self._validate_source(
            source_uri=source_uri,
            source_sha256=source_sha256,
            source_audio=source_audio,
            asset=asset,
        )
        return asset

    @staticmethod
    def _validate_source(
        source_uri: str,
        source_sha256: str,
        source_audio: AudioMetadata,
        asset: CorpusAudioAsset,
    ) -> None:
        match asset.source:
            case LocalSourceAudio(sample_relative_path=sample_relative_path):
                if "://" in source_uri or not _source_path_matches(
                    source_uri, sample_relative_path
                ):
                    raise ValueError(f"Audio asset source path does not match {source_uri}.")
            case RemoteSourceAudio(uri=asset_source_uri):
                if source_uri != asset_source_uri:
                    raise ValueError(f"Audio asset source URI does not match {source_uri}.")
        if source_sha256 != asset.source_sha256:
            raise ValueError(f"Audio asset source hash does not match {source_uri}.")
        duration_tolerance_seconds = 1.0 / asset.source_audio.sample_rate
        if (
            abs(source_audio.duration_seconds - asset.source_audio.duration_seconds)
            > duration_tolerance_seconds
        ):
            raise ValueError(f"Audio asset source duration does not match {source_uri}.")
        if source_audio.sample_rate != asset.source_audio.sample_rate:
            raise ValueError(f"Audio asset source sample rate does not match {source_uri}.")
        if source_audio.channels != asset.source_audio.channels:
            raise ValueError(f"Audio asset source channel count does not match {source_uri}.")
        if source_audio.sample_count != asset.source_audio.sample_count:
            raise ValueError(f"Audio asset source sample count does not match {source_uri}.")


def load_audio_asset_catalog(
    manifest_paths: Sequence[Path],
) -> CorpusAudioAssetCatalog:
    if not manifest_paths:
        raise ValueError("At least one audio build manifest is required.")
    manifests = tuple(
        CorpusAudioStagingManifest.model_validate_json(path.read_text(encoding="utf-8"))
        for path in manifest_paths
    )
    dataset_names = tuple(manifest.dataset_name for manifest in manifests)
    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Audio build manifests contain duplicate dataset names.")
    return CorpusAudioAssetCatalog(manifests=manifests)


def _source_path_matches(source_uri: str, sample_relative_path: PurePosixPath) -> bool:
    source_parts = PurePosixPath(source_uri.replace("\\", "/")).parts
    relative_parts = sample_relative_path.parts
    return (
        len(source_parts) >= len(relative_parts)
        and source_parts[-len(relative_parts) :] == relative_parts
    )
