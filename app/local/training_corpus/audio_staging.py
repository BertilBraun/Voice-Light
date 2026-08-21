from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from uuid import uuid4

import av
from pydantic import Field

from app.shared.base_model import FrozenBaseModel
from app.shared.quality import AudioMetadata, SpeakerSide

AUDIO_STAGING_SCHEMA_VERSION = "voice-light-corpus-audio-staging-v1"
DECODED_HASH_CHUNK_BYTES = 1024 * 1024
TRACK_FILENAME_BY_SIDE = {
    SpeakerSide.SPEAKER1: "speaker_1",
    SpeakerSide.SPEAKER2: "speaker_2",
}


class CorpusAudioPreparation(StrEnum):
    LOSSLESS_FLAC_TRANSCODE = "lossless_flac_transcode"
    HARD_LINKED_FLAC = "hard_linked_flac"
    EXISTING_HUB_FLAC = "existing_hub_flac"


class CorpusAudioVerification(StrEnum):
    METADATA = "metadata"
    DECODED_PCM = "decoded_pcm"
    HARD_LINK_IDENTITY = "hard_link_identity"
    HUB_LFS_SHA256 = "hub_lfs_sha256"


class LocalSourceAudio(FrozenBaseModel):
    kind: Literal["local"] = "local"
    sample_relative_path: PurePosixPath


class RemoteSourceAudio(FrozenBaseModel):
    kind: Literal["remote"] = "remote"
    uri: str = Field(pattern=r"^[a-z][a-z0-9+.-]*://.+")


SourceAudio = Annotated[LocalSourceAudio | RemoteSourceAudio, Field(discriminator="kind")]


class WaveToFlacStagingRequest(FrozenBaseModel):
    dataset_name: str = Field(min_length=1)
    source_samples_root: Path
    corpus_root: Path
    exact_verification_file_count: int = Field(default=2, ge=0, le=2)


class ExistingFlacStagingRequest(FrozenBaseModel):
    dataset_name: str = Field(min_length=1)
    source_samples_root: Path
    corpus_root: Path


DatasetAudioStagingRequest = WaveToFlacStagingRequest | ExistingFlacStagingRequest


class CorpusAudioAsset(FrozenBaseModel):
    sample_id: str
    side: SpeakerSide
    source: SourceAudio
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_relative_path: PurePosixPath
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio: AudioMetadata
    corpus_audio: AudioMetadata
    preparation: CorpusAudioPreparation
    verification: CorpusAudioVerification


class CorpusAudioStagingManifest(FrozenBaseModel):
    schema_version: str
    generated_at: datetime
    dataset_name: str
    assets: tuple[CorpusAudioAsset, ...]


class CorpusAudioStagingSummary(FrozenBaseModel):
    dataset_name: str
    asset_count: int = Field(ge=0)
    manifest_path: Path


def stage_dataset_audio(request: DatasetAudioStagingRequest) -> CorpusAudioStagingManifest:
    source_extension, preparation = _request_contract(request)
    discovered = discover_source_tracks(
        source_samples_root=request.source_samples_root,
        source_extension=source_extension,
    )
    match request:
        case WaveToFlacStagingRequest():
            verification_indices = deterministic_verification_indices(
                item_count=len(discovered),
                verification_count=request.exact_verification_file_count,
            )
        case ExistingFlacStagingRequest():
            verification_indices = frozenset()
    corpus_dataset_root = request.corpus_root / request.dataset_name
    if corpus_dataset_root.exists():
        raise ValueError(f"Corpus dataset output already exists: {corpus_dataset_root}")
    manifest_path = build_manifest_path(request.corpus_root, request.dataset_name)
    if manifest_path.exists():
        raise ValueError(f"Audio build manifest already exists: {manifest_path}")
    request.corpus_root.mkdir(parents=True, exist_ok=True)
    staging_root = request.corpus_root / f".{request.dataset_name}-staging-{uuid4().hex}"
    assets: list[CorpusAudioAsset] = []
    try:
        for index, source in enumerate(discovered):
            relative_path = corpus_track_relative_path(
                dataset_name=request.dataset_name,
                sample_id=source.sample_id,
                side=source.side,
            )
            staging_path = staging_root.joinpath(*relative_path.parts[1:])
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            match request:
                case WaveToFlacStagingRequest():
                    transcode_lossless_flac(source.path, staging_path)
                    verification = (
                        CorpusAudioVerification.DECODED_PCM
                        if index in verification_indices
                        else CorpusAudioVerification.METADATA
                    )
                case ExistingFlacStagingRequest():
                    os.link(source.path, staging_path)
                    if not os.path.samefile(source.path, staging_path):
                        raise ValueError(f"Staged FLAC is not a hard link: {staging_path}")
                    verification = CorpusAudioVerification.HARD_LINK_IDENTITY
            source_audio = probe_archive_audio_metadata(source.path)
            corpus_audio = probe_archive_audio_metadata(staging_path)
            validate_preserved_audio_metadata(
                source_path=source.path,
                source_audio=source_audio,
                corpus_path=staging_path,
                corpus_audio=corpus_audio,
            )
            if verification is CorpusAudioVerification.DECODED_PCM:
                validate_decoded_pcm_equal(source.path, staging_path)
            source_sha256 = encoded_file_sha256(source.path)
            corpus_sha256 = (
                source_sha256
                if preparation is CorpusAudioPreparation.HARD_LINKED_FLAC
                else encoded_file_sha256(staging_path)
            )
            assets.append(
                CorpusAudioAsset(
                    sample_id=source.sample_id,
                    side=source.side,
                    source=LocalSourceAudio(
                        sample_relative_path=PurePosixPath(
                            source.path.relative_to(request.source_samples_root).as_posix()
                        )
                    ),
                    source_sha256=source_sha256,
                    corpus_relative_path=relative_path,
                    corpus_sha256=corpus_sha256,
                    source_audio=source_audio,
                    corpus_audio=corpus_audio,
                    preparation=preparation,
                    verification=verification,
                )
            )
        staging_root.replace(corpus_dataset_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    manifest = CorpusAudioStagingManifest(
        schema_version=AUDIO_STAGING_SCHEMA_VERSION,
        generated_at=datetime.now(UTC),
        dataset_name=request.dataset_name,
        assets=tuple(assets),
    )
    write_build_manifest(request.corpus_root, manifest)
    return manifest


class DiscoveredSourceTrack(FrozenBaseModel):
    sample_id: str
    side: SpeakerSide
    path: Path


def discover_source_tracks(
    source_samples_root: Path,
    source_extension: str,
) -> tuple[DiscoveredSourceTrack, ...]:
    if not source_samples_root.is_dir():
        raise ValueError(f"Source samples directory does not exist: {source_samples_root}")
    discovered: list[DiscoveredSourceTrack] = []
    sample_directories = sorted(path for path in source_samples_root.iterdir() if path.is_dir())
    if not sample_directories:
        raise ValueError(f"Source samples directory contains no samples: {source_samples_root}")
    for sample_directory in sample_directories:
        if not (sample_directory / "metadata.json").is_file():
            raise ValueError(f"Sample metadata is missing: {sample_directory}")
        for side, filename_stem in TRACK_FILENAME_BY_SIDE.items():
            path = sample_directory / f"{filename_stem}{source_extension}"
            if not path.is_file():
                raise ValueError(f"Sample track is missing: {path}")
            discovered.append(
                DiscoveredSourceTrack(
                    sample_id=sample_directory.name,
                    side=side,
                    path=path,
                )
            )
    return tuple(discovered)


def corpus_track_relative_path(
    dataset_name: str,
    sample_id: str,
    side: SpeakerSide,
) -> PurePosixPath:
    validate_path_component(dataset_name, "Dataset name")
    validate_path_component(sample_id, "Sample ID")
    return (
        PurePosixPath(dataset_name) / "samples" / sample_id / f"{TRACK_FILENAME_BY_SIDE[side]}.flac"
    )


def validate_path_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be one non-relative path component.")


def transcode_lossless_flac(source_path: Path, output_path: Path) -> None:
    subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "-1",
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            str(output_path),
        ),
        check=True,
    )


def probe_archive_audio_metadata(path: Path) -> AudioMetadata:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"Audio stream not found: {path}")
        stream = container.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        channels = stream.codec_context.channels
        if sample_rate is None or channels is None:
            raise ValueError(f"Audio stream metadata is incomplete: {path}")
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = container.duration / 1_000_000
        else:
            raise ValueError(f"Audio duration is unavailable: {path}")
    return AudioMetadata(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        sample_count=round(duration_seconds * sample_rate),
    )


def validate_preserved_audio_metadata(
    source_path: Path,
    source_audio: AudioMetadata,
    corpus_path: Path,
    corpus_audio: AudioMetadata,
) -> None:
    if source_audio.sample_rate != corpus_audio.sample_rate:
        raise ValueError(f"FLAC conversion changed the sample rate: {corpus_path}")
    if source_audio.channels != corpus_audio.channels:
        raise ValueError(f"FLAC conversion changed the channel count: {corpus_path}")
    duration_tolerance_seconds = 1.0 / source_audio.sample_rate
    if (
        abs(source_audio.duration_seconds - corpus_audio.duration_seconds)
        > duration_tolerance_seconds
    ):
        raise ValueError(
            f"FLAC conversion changed the duration beyond one source sample: {source_path}"
        )


def validate_decoded_pcm_equal(source_path: Path, corpus_path: Path) -> None:
    source_hash = decoded_pcm_sha256(source_path)
    corpus_hash = decoded_pcm_sha256(corpus_path)
    if source_hash != corpus_hash:
        raise ValueError(f"Decoded PCM differs after FLAC conversion: {source_path}")


def decoded_pcm_sha256(path: Path) -> str:
    process = subprocess.Popen(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s32le",
            "-f",
            "s32le",
            "pipe:1",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    digest = hashlib.sha256()
    while chunk := process.stdout.read(DECODED_HASH_CHUNK_BYTES):
        digest.update(chunk)
    error_output = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise ValueError(f"Could not decode PCM from {path}: {error_output.strip()}")
    return digest.hexdigest()


def encoded_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as audio_file:
        while chunk := audio_file.read(DECODED_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_verification_indices(
    item_count: int,
    verification_count: int,
) -> frozenset[int]:
    if item_count < 0:
        raise ValueError("Item count must not be negative.")
    if verification_count < 0 or verification_count > 2:
        raise ValueError("Verification count must be between zero and two.")
    if item_count == 0 or verification_count == 0:
        return frozenset()
    if verification_count == 1 or item_count == 1:
        return frozenset((0,))
    return frozenset((0, item_count - 1))


def write_build_manifest(
    corpus_root: Path,
    manifest: CorpusAudioStagingManifest,
) -> None:
    path = build_manifest_path(corpus_root, manifest.dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def build_manifest_path(corpus_root: Path, dataset_name: str) -> Path:
    return corpus_root / ".build" / f"{dataset_name}-audio-assets.json"


def _request_contract(
    request: DatasetAudioStagingRequest,
) -> tuple[str, CorpusAudioPreparation]:
    match request:
        case WaveToFlacStagingRequest():
            return ".wav", CorpusAudioPreparation.LOSSLESS_FLAC_TRANSCODE
        case ExistingFlacStagingRequest():
            return ".flac", CorpusAudioPreparation.HARD_LINKED_FLAC


def parse_arguments(arguments: Sequence[str] | None = None) -> DatasetAudioStagingRequest:
    parser = argparse.ArgumentParser(description="Stage dataset audio for the training corpus.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    wave_parser = subparsers.add_parser("transcode-wave")
    flac_parser = subparsers.add_parser("link-flac")
    for operation_parser in (wave_parser, flac_parser):
        operation_parser.add_argument("--dataset-name", required=True)
        operation_parser.add_argument("--source-samples-root", required=True, type=Path)
        operation_parser.add_argument("--corpus-root", required=True, type=Path)
    wave_parser.add_argument("--exact-verification-file-count", default=2, type=int)
    parsed = parser.parse_args(arguments)
    if parsed.operation == "transcode-wave":
        return WaveToFlacStagingRequest(
            dataset_name=parsed.dataset_name,
            source_samples_root=parsed.source_samples_root,
            corpus_root=parsed.corpus_root,
            exact_verification_file_count=parsed.exact_verification_file_count,
        )
    return ExistingFlacStagingRequest(
        dataset_name=parsed.dataset_name,
        source_samples_root=parsed.source_samples_root,
        corpus_root=parsed.corpus_root,
    )


def main() -> None:
    request = parse_arguments()
    manifest = stage_dataset_audio(request)
    summary = CorpusAudioStagingSummary(
        dataset_name=manifest.dataset_name,
        asset_count=len(manifest.assets),
        manifest_path=build_manifest_path(
            corpus_root=request.corpus_root,
            dataset_name=manifest.dataset_name,
        ),
    )
    print(summary.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
