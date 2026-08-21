from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoSibling
from pydantic import Field, field_validator

from app.local.config import DATABASE_URL
from app.local.conversation_regions.models import CONVERSATION_REGION_ANALYSIS_VERSION
from app.local.corpus_audit.repository import CorpusAuditRepository
from app.local.corpus_review.repository import CorpusReviewRepository
from app.local.corpus_review.service import corpus_review_readiness
from app.local.db.models import DashboardSample, SampleListFilter, SampleTrackRecord, TrackSide
from app.local.db.repository import Repository
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.local.training_corpus.audio_staging import (
    AUDIO_STAGING_SCHEMA_VERSION,
    CorpusAudioAsset,
    CorpusAudioPreparation,
    CorpusAudioStagingManifest,
    CorpusAudioVerification,
    LocalSourceAudio,
    RemoteSourceAudio,
    SourceAudio,
    write_build_manifest,
)
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import METRIC_VERSION, AudioMetadata, SpeakerSide

DATASET_1_DATABASE_NAME = "dataset_1-local"
DATASET_1_HUB_DIRECTORY = "dataset_1"
MEETINGS_DATABASE_NAME = "meetings-s3"
MEETINGS_HUB_DIRECTORY = "dataset_4"
TRACK_FILENAME_BY_SIDE = {
    TrackSide.SPEAKER1: "speaker_1.flac",
    TrackSide.SPEAKER2: "speaker_2.flac",
}


class HubAudioManifestBuildRequest(FrozenBaseModel):
    repository_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    review_set_name: str = Field(min_length=1)
    output_root: Path


class HubLfsAudioFile(FrozenBaseModel):
    path: PurePosixPath
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_relative_flac_path(cls, path: PurePosixPath) -> PurePosixPath:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "Hub audio paths must be relative and must not contain parent traversal."
            )
        if path.suffix.lower() != ".flac":
            raise ValueError("Hub audio inventory entries must be FLAC files.")
        return path


def build_dataset_1_hub_audio_manifest(
    samples: Sequence[DashboardSample],
    source_external_ids: Sequence[str],
    inventory: Sequence[HubLfsAudioFile],
    generated_at: datetime,
) -> CorpusAudioStagingManifest:
    inventory_by_path = _unique_inventory_by_path(inventory)
    hub_sample_by_external_id = _dataset_1_hub_sample_mapping(
        source_external_ids=source_external_ids,
        inventory=inventory,
    )
    assets: list[CorpusAudioAsset] = []
    for sample in _ordered_unique_samples(samples):
        hub_sample_id = hub_sample_by_external_id.get(sample.sample.external_id)
        if hub_sample_id is None:
            raise ValueError(
                f"Dataset 1 source order does not contain {sample.sample.external_id}."
            )
        for track in _ordered_tracks(sample):
            corpus_path = (
                PurePosixPath(DATASET_1_HUB_DIRECTORY)
                / "samples"
                / hub_sample_id
                / TRACK_FILENAME_BY_SIDE[track.side]
            )
            hub_file = inventory_by_path.get(corpus_path)
            if hub_file is None:
                raise ValueError(f"Hub audio inventory is missing expected file {corpus_path}.")
            assets.append(
                _asset(
                    sample_id=sample.sample.external_id,
                    track=track,
                    corpus_path=corpus_path,
                    corpus_sha256=hub_file.sha256,
                )
            )
    return _manifest(
        dataset_name=DATASET_1_DATABASE_NAME,
        generated_at=generated_at,
        assets=assets,
    )


def _dataset_1_hub_sample_mapping(
    source_external_ids: Sequence[str],
    inventory: Sequence[HubLfsAudioFile],
) -> Mapping[str, str]:
    ordered_source_ids = tuple(sorted(source_external_ids))
    if len(set(ordered_source_ids)) != len(ordered_source_ids):
        raise ValueError("Dataset 1 source order contains duplicate external IDs.")
    hub_sample_ids = tuple(
        sorted(
            {
                hub_file.path.parts[2]
                for hub_file in inventory
                if len(hub_file.path.parts) == 4
                and hub_file.path.parts[:2] == (DATASET_1_HUB_DIRECTORY, "samples")
            }
        )
    )
    if len(hub_sample_ids) != len(ordered_source_ids):
        raise ValueError(
            "Dataset 1 source and Hub sample counts differ: "
            f"{len(ordered_source_ids)} sources versus {len(hub_sample_ids)} Hub samples."
        )
    return dict(zip(ordered_source_ids, hub_sample_ids, strict=True))


def build_meetings_hub_audio_manifest(
    samples: Sequence[DashboardSample],
    inventory: Sequence[HubLfsAudioFile],
    generated_at: datetime,
) -> CorpusAudioStagingManifest:
    inventory_by_hash = _unique_meeting_inventory_by_hash(inventory)
    assets: list[CorpusAudioAsset] = []
    for sample in _ordered_unique_samples(samples):
        sample_assets: list[CorpusAudioAsset] = []
        for track in _ordered_tracks(sample):
            source_sha256 = _source_sha256(track)
            hub_file = inventory_by_hash.get(source_sha256)
            if hub_file is None:
                raise ValueError(
                    "Hub meeting audio inventory has no file matching source SHA-256 "
                    f"{source_sha256}."
                )
            expected_filename = TRACK_FILENAME_BY_SIDE[track.side]
            if hub_file.path.name != expected_filename:
                raise ValueError(
                    f"Hub file {hub_file.path} does not match database side {track.side.value}."
                )
            sample_assets.append(
                _asset(
                    sample_id=sample.sample.external_id,
                    track=track,
                    corpus_path=hub_file.path,
                    corpus_sha256=hub_file.sha256,
                )
            )
        parent_directories = {asset.corpus_relative_path.parent for asset in sample_assets}
        if len(parent_directories) != 1:
            raise ValueError(
                f"Meeting {sample.sample.external_id} speaker tracks map to different Hub samples."
            )
        assets.extend(sample_assets)
    return _manifest(
        dataset_name=MEETINGS_DATABASE_NAME,
        generated_at=generated_at,
        assets=assets,
    )


def build_existing_hub_audio_manifests(
    request: HubAudioManifestBuildRequest,
) -> tuple[CorpusAudioStagingManifest, CorpusAudioStagingManifest]:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required to build Hub audio manifests.")
    review_plan = CorpusReviewRepository(DATABASE_URL).get(request.review_set_name)
    readiness = corpus_review_readiness(review_plan)
    if not readiness.ready_to_publish:
        raise ValueError(f"Corpus review set {request.review_set_name!r} is not ready to publish.")
    selection_by_dataset_id = {
        selection.dataset_id: selection for selection in review_plan.review_set.config.datasets
    }
    audit_repository = CorpusAuditRepository(DATABASE_URL)
    sample_repository = Repository(DATABASE_URL)
    samples_by_name: dict[str, tuple[DashboardSample, ...]] = {}
    dataset_1_source_external_ids: tuple[str, ...] = ()
    for dataset_name in (DATASET_1_DATABASE_NAME, MEETINGS_DATABASE_NAME):
        dataset = next(
            (item for item in sample_repository.list_datasets() if item.name == dataset_name),
            None,
        )
        if dataset is None:
            raise ValueError(f"Database dataset not found: {dataset_name}")
        selection = selection_by_dataset_id.get(dataset.id)
        if selection is None:
            raise ValueError(f"Review set does not select database dataset {dataset_name}.")
        evidence = audit_repository.load_evidence(
            dataset_ids=(dataset.id,),
            minimum_quality=selection.minimum_quality,
            metric_version=METRIC_VERSION,
            annotation_version=ANNOTATION_VERSION,
            region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        )
        samples_by_name[dataset_name] = tuple(
            sample_repository.get_dashboard_sample(item.sample_id) for item in evidence
        )
        if dataset_name == DATASET_1_DATABASE_NAME:
            all_dataset_samples = sample_repository.list_dashboard_samples(
                SampleListFilter(dataset_id=dataset.id, limit=200)
            )
            dataset_1_source_external_ids = tuple(
                sample.sample.external_id for sample in all_dataset_samples
            )
    inventory = load_hub_lfs_audio_inventory(
        repository_id=request.repository_id,
        revision=request.revision,
    )
    generated_at = datetime.now(UTC)
    manifests = (
        build_dataset_1_hub_audio_manifest(
            samples=samples_by_name[DATASET_1_DATABASE_NAME],
            source_external_ids=dataset_1_source_external_ids,
            inventory=inventory,
            generated_at=generated_at,
        ),
        build_meetings_hub_audio_manifest(
            samples=samples_by_name[MEETINGS_DATABASE_NAME],
            inventory=inventory,
            generated_at=generated_at,
        ),
    )
    for manifest in manifests:
        write_build_manifest(request.output_root, manifest)
    return manifests


def load_hub_lfs_audio_inventory(
    repository_id: str,
    revision: str,
) -> tuple[HubLfsAudioFile, ...]:
    info = HfApi().dataset_info(
        repo_id=repository_id,
        revision=revision,
        files_metadata=True,
    )
    if info.sha != revision:
        raise ValueError(
            f"Hub resolved revision {info.sha!r}, expected exact revision {revision!r}."
        )
    return tuple(
        hub_lfs_audio_file(sibling)
        for sibling in info.siblings
        if sibling.rfilename.lower().endswith(".flac")
    )


def hub_lfs_audio_file(sibling: RepoSibling) -> HubLfsAudioFile:
    if sibling.size is None or sibling.lfs is None:
        raise ValueError(f"Hub FLAC lacks LFS metadata: {sibling.rfilename}")
    return HubLfsAudioFile(
        path=PurePosixPath(sibling.rfilename),
        size_bytes=sibling.size,
        sha256=sibling.lfs.sha256,
    )


def _unique_inventory_by_path(
    inventory: Sequence[HubLfsAudioFile],
) -> Mapping[PurePosixPath, HubLfsAudioFile]:
    by_path: dict[PurePosixPath, HubLfsAudioFile] = {}
    for hub_file in inventory:
        if hub_file.path in by_path:
            raise ValueError(f"Hub audio inventory contains duplicate path {hub_file.path}.")
        by_path[hub_file.path] = hub_file
    return by_path


def _unique_meeting_inventory_by_hash(
    inventory: Sequence[HubLfsAudioFile],
) -> Mapping[str, HubLfsAudioFile]:
    by_hash: dict[str, HubLfsAudioFile] = {}
    for hub_file in inventory:
        if not hub_file.path.is_relative_to(MEETINGS_HUB_DIRECTORY):
            continue
        if hub_file.sha256 in by_hash:
            raise ValueError(
                f"Hub meeting audio inventory contains duplicate SHA-256 {hub_file.sha256}."
            )
        by_hash[hub_file.sha256] = hub_file
    return by_hash


def _ordered_unique_samples(samples: Sequence[DashboardSample]) -> tuple[DashboardSample, ...]:
    ordered = tuple(sorted(samples, key=lambda sample: sample.sample.external_id))
    external_ids = tuple(sample.sample.external_id for sample in ordered)
    if len(set(external_ids)) != len(external_ids):
        raise ValueError("Dashboard samples contain duplicate external IDs.")
    return ordered


def _ordered_tracks(sample: DashboardSample) -> tuple[SampleTrackRecord, SampleTrackRecord]:
    by_side = {track.side: track for track in sample.tracks}
    if len(by_side) != len(sample.tracks):
        raise ValueError(f"Sample {sample.sample.external_id} contains duplicate track sides.")
    if set(by_side) != {TrackSide.SPEAKER1, TrackSide.SPEAKER2}:
        raise ValueError(f"Sample {sample.sample.external_id} must contain two speaker tracks.")
    return by_side[TrackSide.SPEAKER1], by_side[TrackSide.SPEAKER2]


def _asset(
    sample_id: str,
    track: SampleTrackRecord,
    corpus_path: PurePosixPath,
    corpus_sha256: str,
) -> CorpusAudioAsset:
    audio_metadata = _audio_metadata(track)
    return CorpusAudioAsset(
        sample_id=sample_id,
        side=SpeakerSide(track.side.value),
        source=_source_audio(track),
        source_sha256=_source_sha256(track),
        corpus_relative_path=corpus_path,
        corpus_sha256=corpus_sha256,
        source_audio=audio_metadata,
        corpus_audio=audio_metadata,
        preparation=CorpusAudioPreparation.EXISTING_HUB_FLAC,
        verification=CorpusAudioVerification.HUB_LFS_SHA256,
    )


def _source_audio(track: SampleTrackRecord) -> SourceAudio:
    if "://" in track.access_uri:
        return RemoteSourceAudio(uri=track.access_uri)
    return LocalSourceAudio(path=Path(track.access_uri).resolve())


def _source_sha256(track: SampleTrackRecord) -> str:
    if track.audio_sha256 is None:
        raise ValueError(f"Track {track.id} has no source audio SHA-256.")
    return track.audio_sha256


def _audio_metadata(track: SampleTrackRecord) -> AudioMetadata:
    if (
        track.duration_seconds is None
        or track.sample_rate is None
        or track.channels is None
        or track.sample_count is None
    ):
        raise ValueError(f"Track {track.id} has incomplete audio metadata.")
    return AudioMetadata(
        duration_seconds=track.duration_seconds,
        sample_rate=track.sample_rate,
        channels=track.channels,
        sample_count=track.sample_count,
    )


def _manifest(
    dataset_name: str,
    generated_at: datetime,
    assets: Sequence[CorpusAudioAsset],
) -> CorpusAudioStagingManifest:
    return CorpusAudioStagingManifest(
        schema_version=AUDIO_STAGING_SCHEMA_VERSION,
        generated_at=generated_at,
        dataset_name=dataset_name,
        assets=tuple(assets),
    )


def parse_arguments() -> HubAudioManifestBuildRequest:
    parser = argparse.ArgumentParser(
        description="Build private audio manifests for FLAC files already stored on Hugging Face."
    )
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--review-set-name", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args()
    return HubAudioManifestBuildRequest(
        repository_id=arguments.repository_id,
        revision=arguments.revision,
        review_set_name=arguments.review_set_name,
        output_root=arguments.output_root,
    )


def main() -> None:
    manifests = build_existing_hub_audio_manifests(parse_arguments())
    for manifest in manifests:
        print(
            f"{manifest.dataset_name}: {len(manifest.assets)} audio assets",
            flush=True,
        )


if __name__ == "__main__":
    main()
