from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.local.training_corpus.audio_staging import (
    CorpusAudioAsset,
    CorpusAudioStagingManifest,
)
from app.shared.quality import AudioMetadata, SpeakerSide


@dataclass(frozen=True)
class CorpusAudioAssetCatalog:
    manifests: tuple[CorpusAudioStagingManifest, ...]

    def resolve(
        self,
        dataset_name: str,
        external_id: str,
        side: SpeakerSide,
        source_path: Path,
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
            source_path=source_path,
            source_sha256=source_sha256,
            source_audio=source_audio,
            asset=asset,
        )
        return asset

    @staticmethod
    def _validate_source(
        source_path: Path,
        source_sha256: str,
        source_audio: AudioMetadata,
        asset: CorpusAudioAsset,
    ) -> None:
        if source_path.resolve() != asset.source_path.resolve():
            raise ValueError(f"Audio asset source path does not match {source_path}.")
        if source_sha256 != asset.source_sha256:
            raise ValueError(f"Audio asset source hash does not match {source_path}.")
        duration_tolerance_seconds = 1.0 / asset.source_audio.sample_rate
        if (
            abs(source_audio.duration_seconds - asset.source_audio.duration_seconds)
            > duration_tolerance_seconds
        ):
            raise ValueError(f"Audio asset source duration does not match {source_path}.")
        if source_audio.sample_rate != asset.source_audio.sample_rate:
            raise ValueError(f"Audio asset source sample rate does not match {source_path}.")
        if source_audio.channels != asset.source_audio.channels:
            raise ValueError(f"Audio asset source channel count does not match {source_path}.")
        if source_audio.sample_count != asset.source_audio.sample_count:
            raise ValueError(f"Audio asset source sample count does not match {source_path}.")


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
