from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import uuid
import wave
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

SOURCE_PAIR_PATTERN = re.compile(r"^(?P<sample>.+)_ID\d+\.wav$", re.IGNORECASE)
BUFFER_SIZE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class WaveHeader:
    channels: int
    sample_width_bytes: int
    sample_rate: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


@dataclass(frozen=True)
class SourceTrack:
    archive_path: str
    sha256: str
    wave_header: WaveHeader


@dataclass(frozen=True)
class SamplePreparation:
    sample_id: str
    speaker_1: SourceTrack
    speaker_2: SourceTrack


@dataclass(frozen=True)
class PreparationPlan:
    archive_sha256: str
    samples: tuple[SamplePreparation, ...]


def build_preparation_plan(archive_path: Path) -> PreparationPlan:
    if not archive_path.is_file():
        raise ValueError(f"Archive does not exist: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        tracks_by_source_sample: dict[str, list[SourceTrack]] = {}
        for archive_member in archive.infolist():
            archive_path_in_zip = PurePosixPath(archive_member.filename)
            match = SOURCE_PAIR_PATTERN.fullmatch(archive_path_in_zip.name)
            if match is None:
                continue
            if archive_path_in_zip.parent.name != "WAV":
                continue
            source_sample_id = match.group("sample")
            with archive.open(archive_member) as source_file:
                source_bytes = source_file.read()
            tracks_by_source_sample.setdefault(source_sample_id, []).append(
                SourceTrack(
                    archive_path=archive_member.filename,
                    sha256=hashlib.sha256(source_bytes).hexdigest(),
                    wave_header=read_wave_header(source_bytes),
                )
            )

    if not tracks_by_source_sample:
        raise ValueError("Archive contains no paired WAV audio files.")
    samples: list[SamplePreparation] = []
    for index, source_sample_id in enumerate(sorted(tracks_by_source_sample), start=1):
        tracks = sorted(
            tracks_by_source_sample[source_sample_id], key=lambda track: track.archive_path
        )
        if len(tracks) != 2:
            raise ValueError(f"Expected exactly two tracks for source sample {source_sample_id}.")
        validate_pair(tracks[0], tracks[1], source_sample_id)
        samples.append(
            SamplePreparation(
                sample_id=f"sample_{index:03d}",
                speaker_1=tracks[0],
                speaker_2=tracks[1],
            )
        )
    return PreparationPlan(
        archive_sha256=hash_file(archive_path),
        samples=tuple(samples),
    )


def apply_preparation(
    archive_path: Path,
    target_samples_root: Path,
    restricted_record_path: Path,
) -> PreparationPlan:
    plan = build_preparation_plan(archive_path)
    ensure_empty_target(target_samples_root)
    if restricted_record_path.is_relative_to(target_samples_root.parent):
        raise ValueError("Restricted record must be outside the dataset target directory.")

    staging_root = (
        target_samples_root.parent / f".{target_samples_root.name}-staging-{uuid.uuid4().hex}"
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for sample in plan.samples:
                sample_root = staging_root / sample.sample_id
                sample_root.mkdir(parents=True)
                extract_verified_track(
                    archive=archive,
                    source_track=sample.speaker_1,
                    target_path=sample_root / "speaker_1.wav",
                )
                extract_verified_track(
                    archive=archive,
                    source_track=sample.speaker_2,
                    target_path=sample_root / "speaker_2.wav",
                )
                write_minimal_metadata(sample_root / "metadata.json", sample.sample_id)
        verify_prepared_layout(staging_root, plan)
        target_samples_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root.replace(target_samples_root)
        write_restricted_record(restricted_record_path, plan)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return plan


def verify_prepared_layout(target_samples_root: Path, plan: PreparationPlan) -> None:
    expected_sample_ids = {sample.sample_id for sample in plan.samples}
    actual_sample_ids = {path.name for path in target_samples_root.iterdir() if path.is_dir()}
    if actual_sample_ids != expected_sample_ids:
        raise ValueError("Prepared sample directories do not match the migration plan.")
    for sample in plan.samples:
        sample_root = target_samples_root / sample.sample_id
        expected_paths = {
            sample_root / "speaker_1.wav",
            sample_root / "speaker_2.wav",
            sample_root / "metadata.json",
        }
        actual_paths = {path for path in sample_root.iterdir() if path.is_file()}
        if actual_paths != expected_paths:
            raise ValueError(f"Prepared sample has unexpected files: {sample.sample_id}")
        verify_extracted_track(sample_root / "speaker_1.wav", sample.speaker_1)
        verify_extracted_track(sample_root / "speaker_2.wav", sample.speaker_2)
        metadata = json.loads((sample_root / "metadata.json").read_text(encoding="utf-8"))
        expected_metadata = minimal_metadata(sample.sample_id)
        if metadata != expected_metadata:
            raise ValueError(f"Prepared metadata is invalid: {sample.sample_id}")


def minimal_metadata(sample_id: str) -> dict[str, str]:
    return {
        "name": sample_id,
        "speaker1_audioURL": "speaker_1.wav",
        "speaker2_audioURL": "speaker_2.wav",
    }


def read_wave_header(source_bytes: bytes) -> WaveHeader:
    from io import BytesIO

    with wave.open(BytesIO(source_bytes), "rb") as wave_file:
        if wave_file.getcomptype() != "NONE":
            raise ValueError("Compressed WAV audio is not supported.")
        return WaveHeader(
            channels=wave_file.getnchannels(),
            sample_width_bytes=wave_file.getsampwidth(),
            sample_rate=wave_file.getframerate(),
            frame_count=wave_file.getnframes(),
        )


def read_wave_header_from_path(path: Path) -> WaveHeader:
    with wave.open(str(path), "rb") as wave_file:
        if wave_file.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV audio is not supported: {path}")
        return WaveHeader(
            channels=wave_file.getnchannels(),
            sample_width_bytes=wave_file.getsampwidth(),
            sample_rate=wave_file.getframerate(),
            frame_count=wave_file.getnframes(),
        )


def validate_pair(
    first_track: SourceTrack, second_track: SourceTrack, source_sample_id: str
) -> None:
    if first_track.wave_header != second_track.wave_header:
        raise ValueError(f"WAV headers differ for source sample {source_sample_id}.")
    if first_track.wave_header.channels != 1:
        raise ValueError(f"Source sample {source_sample_id} is not mono.")


def extract_verified_track(
    archive: zipfile.ZipFile,
    source_track: SourceTrack,
    target_path: Path,
) -> None:
    digest = hashlib.sha256()
    with (
        archive.open(source_track.archive_path) as source_file,
        target_path.open("wb") as target_file,
    ):
        copy_and_hash(source_file, target_file, digest)
    if digest.hexdigest() != source_track.sha256:
        raise ValueError(f"Extracted audio hash does not match: {source_track.archive_path}")


def copy_and_hash(source_file: BinaryIO, target_file: BinaryIO, digest: hashlib._Hash) -> None:
    while chunk := source_file.read(BUFFER_SIZE_BYTES):
        target_file.write(chunk)
        digest.update(chunk)


def verify_extracted_track(path: Path, source_track: SourceTrack) -> None:
    if hash_file(path) != source_track.sha256:
        raise ValueError(f"Prepared audio hash does not match: {path}")
    if read_wave_header_from_path(path) != source_track.wave_header:
        raise ValueError(f"Prepared WAV header does not match: {path}")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(BUFFER_SIZE_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_minimal_metadata(path: Path, sample_id: str) -> None:
    path.write_text(json.dumps(minimal_metadata(sample_id), indent=2) + "\n", encoding="utf-8")


def write_restricted_record(path: Path, plan: PreparationPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"Restricted record already exists: {path}")
    record = {
        "source_attribution": "Multi-stream Spontaneous Conversation Training Dataset",
        "source_url": "https://magichub.com/datasets/multi-stream-spontaneous-conversation-training-datasets_english/",
        "archive_sha256": plan.archive_sha256,
        "samples": [asdict(sample) for sample in plan.samples],
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def ensure_empty_target(target_samples_root: Path) -> None:
    if target_samples_root.exists() and any(target_samples_root.iterdir()):
        raise ValueError(f"Target sample root is not empty: {target_samples_root}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local paired-WAV archive as generic samples."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target-samples-root", type=Path, required=True)
    parser.add_argument("--restricted-record", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    plan = build_preparation_plan(arguments.archive)
    if arguments.apply:
        apply_preparation(
            archive_path=arguments.archive,
            target_samples_root=arguments.target_samples_root,
            restricted_record_path=arguments.restricted_record,
        )
    print(f"Validated {len(plan.samples)} paired samples.")


if __name__ == "__main__":
    main()
