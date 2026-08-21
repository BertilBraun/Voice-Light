from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID

import pyarrow.parquet as pq
from pydantic import Field

from app.local.db.models import TrackSide
from app.local.training_corpus.audio_assets import load_audio_asset_catalog
from app.local.training_corpus.audio_staging import (
    CorpusAudioAsset,
    CorpusAudioPreparation,
    CorpusAudioStagingManifest,
    CorpusAudioVerification,
)
from app.local.training_corpus.export import (
    ExportManifest,
    ExportSplitSummary,
    MaterializedTrainingSample,
    RecordingMetadata,
)
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import SpeakerSide

HASH_CHUNK_BYTES = 1024 * 1024
TIMELINE_TOLERANCE_SECONDS = 0.08


class ExportedCorpusValidationRequest(FrozenBaseModel):
    corpus_directory: Path
    audio_build_manifest_paths: tuple[Path, ...] = Field(min_length=1)


class DatasetSplitConcentration(FrozenBaseModel):
    dataset_id: UUID
    dataset_name: str
    split: TrainingCorpusSplit
    recording_count: int = Field(ge=0)
    window_count: int = Field(ge=0)
    maximum_recording_window_count: int = Field(ge=0)
    maximum_recording_concentration: float = Field(ge=0.0, le=1.0)


class ExportedCorpusValidationReport(FrozenBaseModel):
    recording_count: int = Field(ge=0)
    training_sample_count: int = Field(ge=0)
    shard_count: int = Field(ge=0)
    audio_reference_count: int = Field(ge=0)
    local_audio_file_count: int = Field(ge=0)
    hub_audio_reference_count: int = Field(ge=0)
    concentrations: tuple[DatasetSplitConcentration, ...]


@dataclass(frozen=True)
class RecordingKey:
    dataset_id: UUID
    sample_id: UUID


@dataclass(frozen=True)
class AudioAssetKey:
    dataset_name: str
    external_id: str
    side: SpeakerSide


@dataclass(frozen=True)
class LogicalWindowKey:
    recording: RecordingKey
    user_side: TrackSide
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class AudioValidationCounts:
    reference_count: int
    local_file_count: int
    hub_reference_count: int


def validate_exported_corpus(
    request: ExportedCorpusValidationRequest,
) -> ExportedCorpusValidationReport:
    corpus_directory = request.corpus_directory.resolve()
    manifest_path = corpus_directory / "corpus.json"
    manifest = ExportManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_inventory(manifest)
    audio_catalog = load_audio_asset_catalog(request.audio_build_manifest_paths)
    asset_by_key = _audio_asset_index(audio_catalog.manifests)
    metadata_by_key = _load_recording_metadata(corpus_directory)
    _validate_recording_inventory(manifest=manifest, metadata_by_key=metadata_by_key)
    audio_counts = _validate_recording_audio(
        corpus_directory=corpus_directory,
        metadata_by_key=metadata_by_key,
        asset_by_key=asset_by_key,
    )
    samples = _load_and_validate_shards(
        corpus_directory=corpus_directory,
        manifest=manifest,
        metadata_by_key=metadata_by_key,
        asset_by_key=asset_by_key,
    )
    _validate_sample_inventory(
        manifest=manifest,
        samples=samples,
        metadata_by_key=metadata_by_key,
    )
    return ExportedCorpusValidationReport(
        recording_count=len(metadata_by_key),
        training_sample_count=len(samples),
        shard_count=len(manifest.shards),
        audio_reference_count=audio_counts.reference_count,
        local_audio_file_count=audio_counts.local_file_count,
        hub_audio_reference_count=audio_counts.hub_reference_count,
        concentrations=_concentrations(samples=samples, metadata_by_key=metadata_by_key),
    )


def _validate_manifest_inventory(manifest: ExportManifest) -> None:
    summaries_by_split = Counter(summary.split for summary in manifest.splits)
    if summaries_by_split != Counter({split: 1 for split in TrainingCorpusSplit}):
        raise ValueError("Corpus manifest must contain exactly one summary for every split.")
    shard_paths = tuple(shard.path for shard in manifest.shards)
    if len(set(shard_paths)) != len(shard_paths):
        raise ValueError("Corpus manifest contains duplicate shard paths.")
    for path in shard_paths:
        _validated_relative_path(path, "Shard")
    if sum(shard.row_count for shard in manifest.shards) != manifest.training_sample_count:
        raise ValueError("Manifest shard row counts do not match its training sample count.")
    if sum(summary.training_sample_count for summary in manifest.splits) != (
        manifest.training_sample_count
    ):
        raise ValueError("Manifest split counts do not match its training sample count.")
    if sum(summary.recording_count for summary in manifest.splits) != manifest.recording_count:
        raise ValueError("Manifest split counts do not match its recording count.")


def _audio_asset_index(
    manifests: Sequence[CorpusAudioStagingManifest],
) -> Mapping[AudioAssetKey, CorpusAudioAsset]:
    by_key: dict[AudioAssetKey, CorpusAudioAsset] = {}
    paths: set[PurePosixPath] = set()
    for manifest in manifests:
        for asset in manifest.assets:
            key = AudioAssetKey(
                dataset_name=manifest.dataset_name,
                external_id=asset.sample_id,
                side=asset.side,
            )
            if key in by_key:
                raise ValueError(
                    "Audio manifests contain duplicate asset for "
                    f"{key.dataset_name}/{key.external_id}/{key.side.value}."
                )
            if asset.corpus_relative_path in paths:
                raise ValueError(
                    f"Audio manifests contain duplicate corpus path {asset.corpus_relative_path}."
                )
            _validated_relative_path(asset.corpus_relative_path.as_posix(), "Audio")
            by_key[key] = asset
            paths.add(asset.corpus_relative_path)
    return by_key


def _load_recording_metadata(
    corpus_directory: Path,
) -> Mapping[RecordingKey, RecordingMetadata]:
    by_key: dict[RecordingKey, RecordingMetadata] = {}
    paired_hashes: dict[tuple[str, str], RecordingKey] = {}
    for path in sorted(corpus_directory.glob("**/metadata.json")):
        metadata = RecordingMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        key = RecordingKey(dataset_id=metadata.dataset_id, sample_id=metadata.sample_id)
        if key in by_key:
            raise ValueError(f"Corpus contains duplicate recording metadata for {key.sample_id}.")
        audio_by_side = {reference.side: reference for reference in metadata.audio}
        if set(audio_by_side) != {TrackSide.SPEAKER1, TrackSide.SPEAKER2}:
            raise ValueError(f"Recording {metadata.external_id} must contain two audio sides.")
        paired_hash = (
            audio_by_side[TrackSide.SPEAKER1].audio_sha256,
            audio_by_side[TrackSide.SPEAKER2].audio_sha256,
        )
        duplicate = paired_hashes.get(paired_hash)
        if duplicate is not None:
            raise ValueError(
                f"Recordings {duplicate.sample_id} and {metadata.sample_id} reference duplicate "
                "paired audio."
            )
        _validate_metadata_timeline(metadata)
        by_key[key] = metadata
        paired_hashes[paired_hash] = key
    return by_key


def _validate_recording_inventory(
    manifest: ExportManifest,
    metadata_by_key: Mapping[RecordingKey, RecordingMetadata],
) -> None:
    assignments = manifest.split_plan.assignments
    assignment_by_key = {
        RecordingKey(assignment.dataset_id, assignment.sample_id): assignment
        for assignment in assignments
    }
    if len(assignment_by_key) != len(assignments):
        raise ValueError("Split plan contains duplicate recording assignments.")
    if set(assignment_by_key) != set(metadata_by_key):
        raise ValueError("Recording metadata inventory does not match the split plan.")
    if len(metadata_by_key) != manifest.recording_count:
        raise ValueError("Recording metadata count does not match the manifest.")
    for key, metadata in metadata_by_key.items():
        if metadata.split is not assignment_by_key[key].split:
            raise ValueError(
                f"Recording {metadata.external_id} metadata does not match its split assignment."
            )
        if (
            metadata.schema_version != manifest.schema_version
            or metadata.metric_version != manifest.metric_version
            or metadata.annotation.annotation_version != manifest.annotation_version
            or metadata.conversation_regions.analysis_version != manifest.region_analysis_version
            or metadata.conversation_regions.annotation_version != manifest.annotation_version
        ):
            raise ValueError(
                f"Recording {metadata.external_id} metadata does not match manifest versions."
            )


def _validate_metadata_timeline(metadata: RecordingMetadata) -> None:
    if metadata.annotation.analyzed_duration_seconds > (
        metadata.duration_seconds + TIMELINE_TOLERANCE_SECONDS
    ):
        raise ValueError(f"Recording {metadata.external_id} annotation exceeds its duration.")
    if abs(metadata.conversation_regions.duration_seconds - metadata.duration_seconds) > (
        TIMELINE_TOLERANCE_SECONDS
    ):
        raise ValueError(
            f"Recording {metadata.external_id} region duration does not match its duration."
        )
    for vad in (metadata.speaker1_vad, metadata.speaker2_vad):
        if vad.speech_segments and vad.speech_segments[-1].end_seconds > (
            metadata.duration_seconds + TIMELINE_TOLERANCE_SECONDS
        ):
            raise ValueError(f"Recording {metadata.external_id} VAD exceeds its duration.")
    for exclusion in metadata.review_interval_exclusions:
        if exclusion.end_seconds > metadata.duration_seconds + TIMELINE_TOLERANCE_SECONDS:
            raise ValueError(f"Recording {metadata.external_id} exclusion exceeds its duration.")


def _validate_recording_audio(
    corpus_directory: Path,
    metadata_by_key: Mapping[RecordingKey, RecordingMetadata],
    asset_by_key: Mapping[AudioAssetKey, CorpusAudioAsset],
) -> AudioValidationCounts:
    referenced_paths: set[PurePosixPath] = set()
    local_paths: set[PurePosixPath] = set()
    hub_paths: set[PurePosixPath] = set()
    for metadata in metadata_by_key.values():
        for reference in metadata.audio:
            side = SpeakerSide(reference.side.value)
            asset_key = AudioAssetKey(metadata.dataset_name, metadata.external_id, side)
            asset = asset_by_key.get(asset_key)
            if asset is None:
                raise ValueError(
                    "Audio manifests have no asset for "
                    f"{metadata.dataset_name}/{metadata.external_id}/{side.value}."
                )
            path = _validated_relative_path(reference.path, "Audio")
            if path != asset.corpus_relative_path:
                raise ValueError(f"Recording {metadata.external_id} audio path is not canonical.")
            if reference.audio_sha256 != asset.corpus_sha256:
                raise ValueError(f"Recording {metadata.external_id} audio hash is not canonical.")
            _validate_audio_metadata(
                duration_seconds=reference.duration_seconds,
                sample_rate=reference.sample_rate,
                channels=reference.channels,
                asset=asset,
            )
            local_path = corpus_directory.joinpath(*path.parts)
            if local_path.is_file():
                if _file_sha256(local_path) != asset.corpus_sha256:
                    raise ValueError(f"Local audio hash does not match {path}.")
                local_paths.add(path)
            elif not (
                asset.preparation is CorpusAudioPreparation.EXISTING_HUB_FLAC
                and asset.verification is CorpusAudioVerification.HUB_LFS_SHA256
            ):
                raise ValueError(f"Required staged audio file is missing: {path}.")
            else:
                hub_paths.add(path)
            referenced_paths.add(path)
    return AudioValidationCounts(
        reference_count=len(referenced_paths),
        local_file_count=len(local_paths),
        hub_reference_count=len(hub_paths),
    )


def _validate_audio_metadata(
    duration_seconds: float,
    sample_rate: int,
    channels: int,
    asset: CorpusAudioAsset,
) -> None:
    tolerance_seconds = 1.0 / asset.corpus_audio.sample_rate
    if abs(duration_seconds - asset.corpus_audio.duration_seconds) > tolerance_seconds:
        raise ValueError(
            f"Audio duration does not match manifest asset {asset.corpus_relative_path}."
        )
    if sample_rate != asset.corpus_audio.sample_rate:
        raise ValueError(
            f"Audio sample rate does not match manifest asset {asset.corpus_relative_path}."
        )
    if channels != asset.corpus_audio.channels:
        raise ValueError(
            f"Audio channel count does not match manifest asset {asset.corpus_relative_path}."
        )


def _load_and_validate_shards(
    corpus_directory: Path,
    manifest: ExportManifest,
    metadata_by_key: Mapping[RecordingKey, RecordingMetadata],
    asset_by_key: Mapping[AudioAssetKey, CorpusAudioAsset],
) -> tuple[MaterializedTrainingSample, ...]:
    samples: list[MaterializedTrainingSample] = []
    for shard in manifest.shards:
        relative_path = _validated_relative_path(shard.path, "Shard")
        path = corpus_directory.joinpath(*relative_path.parts)
        if not path.is_file():
            raise ValueError(f"Manifest shard is missing: {shard.path}.")
        if path.stat().st_size != shard.size_bytes:
            raise ValueError(f"Shard size does not match manifest: {shard.path}.")
        if _file_sha256(path) != shard.sha256:
            raise ValueError(f"Shard hash does not match manifest: {shard.path}.")
        rows = pq.read_table(path).to_pylist()
        if len(rows) != shard.row_count:
            raise ValueError(f"Shard row count does not match manifest: {shard.path}.")
        for row in rows:
            sample = MaterializedTrainingSample.model_validate(row)
            if sample.split is not shard.split:
                raise ValueError(f"Sample {sample.window_id} is stored in the wrong split shard.")
            _validate_sample(
                sample=sample,
                manifest=manifest,
                metadata_by_key=metadata_by_key,
                asset_by_key=asset_by_key,
            )
            samples.append(sample)
    return tuple(samples)


def _validate_sample(
    sample: MaterializedTrainingSample,
    manifest: ExportManifest,
    metadata_by_key: Mapping[RecordingKey, RecordingMetadata],
    asset_by_key: Mapping[AudioAssetKey, CorpusAudioAsset],
) -> None:
    key = RecordingKey(sample.dataset_id, sample.sample_id)
    metadata = metadata_by_key.get(key)
    if metadata is None:
        raise ValueError(f"Sample {sample.window_id} has no recording metadata.")
    if (
        sample.schema_version != manifest.schema_version
        or sample.training_label_version != manifest.training_label_version
    ):
        raise ValueError(f"Sample {sample.window_id} does not match manifest versions.")
    if (
        sample.dataset_name != metadata.dataset_name
        or sample.external_id != metadata.external_id
        or sample.split is not metadata.split
        or sample.quality_score != metadata.quality_score
    ):
        raise ValueError(f"Sample {sample.window_id} does not match its recording metadata.")
    if sample.assistant_side is sample.user_side:
        raise ValueError(f"Sample {sample.window_id} uses the same side for both speakers.")
    expected_assistant_side = (
        TrackSide.SPEAKER2 if sample.user_side is TrackSide.SPEAKER1 else TrackSide.SPEAKER1
    )
    if sample.assistant_side is not expected_assistant_side:
        raise ValueError(f"Sample {sample.window_id} has an invalid assistant side.")
    if abs(sample.end_seconds - sample.start_seconds - manifest.input_duration_seconds) > 1e-9:
        raise ValueError(f"Sample {sample.window_id} has an invalid window duration.")
    if sample.end_seconds > metadata.duration_seconds + TIMELINE_TOLERANCE_SECONDS:
        raise ValueError(f"Sample {sample.window_id} exceeds its recording duration.")
    if sample.end_seconds > (
        metadata.annotation.analyzed_duration_seconds + TIMELINE_TOLERANCE_SECONDS
    ):
        raise ValueError(f"Sample {sample.window_id} exceeds its annotation duration.")
    audio_by_side = {reference.side: reference for reference in metadata.audio}
    expected_user_path = audio_by_side[sample.user_side].path
    expected_assistant_path = audio_by_side[sample.assistant_side].path
    if (
        sample.user_audio_path != expected_user_path
        or sample.assistant_audio_path != expected_assistant_path
    ):
        raise ValueError(f"Sample {sample.window_id} audio references are not canonical.")
    for side, path in (
        (SpeakerSide(sample.user_side.value), sample.user_audio_path),
        (SpeakerSide(sample.assistant_side.value), sample.assistant_audio_path),
    ):
        asset = asset_by_key[AudioAssetKey(sample.dataset_name, sample.external_id, side)]
        if PurePosixPath(path) != asset.corpus_relative_path:
            raise ValueError(f"Sample {sample.window_id} audio asset does not match its path.")


def _validate_sample_inventory(
    manifest: ExportManifest,
    samples: Sequence[MaterializedTrainingSample],
    metadata_by_key: Mapping[RecordingKey, RecordingMetadata],
) -> None:
    if len(samples) != manifest.training_sample_count:
        raise ValueError("Loaded training sample count does not match the manifest.")
    window_ids = tuple(sample.window_id for sample in samples)
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("Corpus contains duplicate window IDs.")
    logical_windows = tuple(
        LogicalWindowKey(
            recording=RecordingKey(sample.dataset_id, sample.sample_id),
            user_side=sample.user_side,
            start_seconds=sample.start_seconds,
            end_seconds=sample.end_seconds,
        )
        for sample in samples
    )
    if len(set(logical_windows)) != len(logical_windows):
        raise ValueError("Corpus contains duplicate logical training windows.")
    splits_by_recording: dict[RecordingKey, set[TrainingCorpusSplit]] = {}
    for sample in samples:
        key = RecordingKey(sample.dataset_id, sample.sample_id)
        splits_by_recording.setdefault(key, set()).add(sample.split)
    if any(len(splits) != 1 for splits in splits_by_recording.values()):
        raise ValueError("A recording appears in more than one corpus split.")
    for summary in manifest.splits:
        _validate_split_summary(
            summary=summary,
            samples=samples,
            metadata_by_key=metadata_by_key,
        )


def _validate_split_summary(
    summary: ExportSplitSummary,
    samples: Sequence[MaterializedTrainingSample],
    metadata_by_key: Mapping[RecordingKey, RecordingMetadata],
) -> None:
    split_metadata = tuple(
        metadata for metadata in metadata_by_key.values() if metadata.split is summary.split
    )
    if len(split_metadata) != summary.recording_count:
        raise ValueError(f"Recording count does not match {summary.split.value} summary.")
    if sum(sample.split is summary.split for sample in samples) != summary.training_sample_count:
        raise ValueError(f"Training sample count does not match {summary.split.value} summary.")
    duration_seconds = sum(metadata.duration_seconds for metadata in split_metadata)
    if abs(duration_seconds - summary.source_duration_seconds) > TIMELINE_TOLERANCE_SECONDS:
        raise ValueError(f"Source duration does not match {summary.split.value} summary.")


def _concentrations(
    samples: Sequence[MaterializedTrainingSample],
    metadata_by_key: Mapping[RecordingKey, RecordingMetadata],
) -> tuple[DatasetSplitConcentration, ...]:
    dataset_names = {
        metadata.dataset_id: metadata.dataset_name for metadata in metadata_by_key.values()
    }
    dataset_ids = tuple(sorted(dataset_names, key=lambda value: value.hex))
    reports: list[DatasetSplitConcentration] = []
    for dataset_id in dataset_ids:
        for split in TrainingCorpusSplit:
            recording_counts = Counter(
                sample.sample_id
                for sample in samples
                if sample.dataset_id == dataset_id and sample.split is split
            )
            window_count = sum(recording_counts.values())
            maximum_count = max(recording_counts.values(), default=0)
            recording_count = sum(
                metadata.dataset_id == dataset_id and metadata.split is split
                for metadata in metadata_by_key.values()
            )
            reports.append(
                DatasetSplitConcentration(
                    dataset_id=dataset_id,
                    dataset_name=dataset_names[dataset_id],
                    split=split,
                    recording_count=recording_count,
                    window_count=window_count,
                    maximum_recording_window_count=maximum_count,
                    maximum_recording_concentration=(
                        maximum_count / window_count if window_count else 0.0
                    ),
                )
            )
    return tuple(reports)


def _validated_relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if "\\" in value or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} path must be a safe corpus-relative path: {value!r}.")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments() -> ExportedCorpusValidationRequest:
    parser = argparse.ArgumentParser(description="Validate a materialized turn-taking corpus.")
    parser.add_argument("--corpus-directory", required=True, type=Path)
    parser.add_argument("--audio-build-manifest", action="append", required=True, type=Path)
    arguments = parser.parse_args()
    return ExportedCorpusValidationRequest(
        corpus_directory=arguments.corpus_directory,
        audio_build_manifest_paths=tuple(arguments.audio_build_manifest),
    )


def main() -> None:
    report = validate_exported_corpus(parse_arguments())
    print(json.dumps(report.model_dump(mode="json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
