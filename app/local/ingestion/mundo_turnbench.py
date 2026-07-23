from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as parquet

from app.local.source_annotations.models import (
    SourceAnnotationEvent,
    SourceAnnotationTrack,
    SourceSampleAnnotations,
)
from app.shared.base_model import FrozenBaseModel

DATASET_NAME = "dataset_3"
SOURCE_ANNOTATION_VERSION = "source-human-annotations-v1"
_REQUIRED_COLUMNS = (
    "conversation_id",
    "speaker_1_audio",
    "speaker_2_audio",
    "speaker_1_annotation_a",
    "speaker_1_annotation_b",
    "speaker_1_annotation_c",
    "speaker_2_annotation_a",
    "speaker_2_annotation_b",
    "speaker_2_annotation_c",
)
_ANNOTATOR_IDS = ("a", "b", "c")


class MinimalSampleMetadata(FrozenBaseModel):
    name: str
    speaker1_audioURL: str
    speaker2_audioURL: str


@dataclass(frozen=True)
class SourceAudio:
    content: bytes
    sha256: str


@dataclass(frozen=True)
class TurnBenchSamplePreparation:
    source_conversation_id: str
    sample_id: str
    speaker_1: SourceAudio
    speaker_2: SourceAudio
    annotations: SourceSampleAnnotations


@dataclass(frozen=True)
class TurnBenchPreparationPlan:
    samples: tuple[TurnBenchSamplePreparation, ...]


def build_preparation_plan(source_parquet_path: Path | Sequence[Path]) -> TurnBenchPreparationPlan:
    source_paths = normalized_source_paths(source_parquet_path)
    samples: list[TurnBenchSamplePreparation] = []
    for source_path in source_paths:
        table = parquet.read_table(source_path, columns=list(_REQUIRED_COLUMNS))
        missing = set(_REQUIRED_COLUMNS) - set(table.column_names)
        if missing:
            raise ValueError(f"TurnBench parquet is missing required columns: {sorted(missing)}")
        for row in table.to_pylist():
            samples.append(preparation_from_row(row, len(samples) + 1))
    if not samples:
        raise ValueError("TurnBench parquet contains no conversations.")
    return TurnBenchPreparationPlan(samples=tuple(samples))


def download_source_parquet(download_root: Path) -> tuple[Path, ...]:
    from huggingface_hub import snapshot_download

    snapshot_path = Path(
        snapshot_download(
            repo_id="mundo-ai/turn-benchmark-dev",
            repo_type="dataset",
            allow_patterns="*.parquet",
            local_dir=download_root,
        )
    )
    parquet_paths = tuple(sorted(snapshot_path.rglob("*.parquet")))
    if not parquet_paths:
        raise ValueError("TurnBench download contains no parquet source files.")
    return parquet_paths


def normalized_source_paths(source_parquet_path: Path | Sequence[Path]) -> tuple[Path, ...]:
    source_paths = (
        (source_parquet_path,)
        if isinstance(source_parquet_path, Path)
        else tuple(source_parquet_path)
    )
    if not source_paths:
        raise ValueError("TurnBench preparation requires at least one parquet source file.")
    for path in source_paths:
        if not path.is_file():
            raise ValueError(f"TurnBench parquet file does not exist: {path}")
    return tuple(sorted(source_paths))


def preparation_from_row(row: Mapping[str, object], index: int) -> TurnBenchSamplePreparation:
    conversation_id = required_string(row, "conversation_id")
    return TurnBenchSamplePreparation(
        source_conversation_id=conversation_id,
        sample_id=f"sample_{index:03d}",
        speaker_1=source_audio(row, "speaker_1_audio"),
        speaker_2=source_audio(row, "speaker_2_audio"),
        annotations=SourceSampleAnnotations(
            annotation_version=SOURCE_ANNOTATION_VERSION,
            speaker_1=annotation_tracks(row, "speaker_1"),
            speaker_2=annotation_tracks(row, "speaker_2"),
        ),
    )


def apply_preparation(
    source_parquet_path: Path | Sequence[Path],
    target_samples_root: Path,
    restricted_record_path: Path,
) -> TurnBenchPreparationPlan:
    plan = build_preparation_plan(source_parquet_path)
    ensure_empty_target(target_samples_root)
    if restricted_record_path.is_relative_to(target_samples_root.parent):
        raise ValueError("Restricted record must be outside the dataset target directory.")
    staging_root = (
        target_samples_root.parent / f".{target_samples_root.name}-staging-{uuid.uuid4().hex}"
    )
    try:
        for sample in plan.samples:
            sample_root = staging_root / sample.sample_id
            sample_root.mkdir(parents=True)
            write_audio(sample_root / "speaker_1.flac", sample.speaker_1)
            write_audio(sample_root / "speaker_2.flac", sample.speaker_2)
            write_json(sample_root / "metadata.json", minimal_metadata(sample.sample_id))
            write_json(sample_root / "source_annotations.json", sample.annotations)
        verify_prepared_layout(staging_root, plan)
        target_samples_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root.replace(target_samples_root)
        write_restricted_record(restricted_record_path, plan)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return plan


def source_audio(row: Mapping[str, object], column_name: str) -> SourceAudio:
    value = row[column_name]
    if not isinstance(value, Mapping):
        raise ValueError(f"TurnBench {column_name} must be an audio struct.")
    content = value.get("bytes")
    if not isinstance(content, bytes):
        raise ValueError(f"TurnBench {column_name} must contain embedded FLAC bytes.")
    if not content.startswith(b"fLaC"):
        raise ValueError(f"TurnBench {column_name} is not FLAC audio.")
    return SourceAudio(content=content, sha256=hashlib.sha256(content).hexdigest())


def annotation_tracks(
    row: Mapping[str, object], speaker_column_prefix: str
) -> tuple[SourceAnnotationTrack, ...]:
    tracks: list[SourceAnnotationTrack] = []
    for annotator_id in _ANNOTATOR_IDS:
        value = row[f"{speaker_column_prefix}_annotation_{annotator_id}"]
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise ValueError("TurnBench annotation column must be a list of events.")
        tracks.append(
            SourceAnnotationTrack(
                annotator_id=annotator_id,
                events=tuple(annotation_event(event) for event in value),
            )
        )
    return tuple(tracks)


def annotation_event(value: object) -> SourceAnnotationEvent:
    if not isinstance(value, Mapping):
        raise ValueError("TurnBench annotation event must be an object.")
    start_seconds = value.get("start_s")
    end_seconds = value.get("end_s")
    label = value.get("label")
    text = value.get("text")
    if not isinstance(start_seconds, int | float) or not isinstance(end_seconds, int | float):
        raise ValueError("TurnBench annotation event timestamps must be numeric.")
    if end_seconds < start_seconds:
        raise ValueError("TurnBench annotation event end precedes start.")
    if not isinstance(label, str) or not isinstance(text, str):
        raise ValueError("TurnBench annotation event label and text must be strings.")
    return SourceAnnotationEvent(
        start_seconds=float(start_seconds),
        end_seconds=float(end_seconds),
        label=label,
        text=text,
    )


def minimal_metadata(sample_id: str) -> MinimalSampleMetadata:
    return MinimalSampleMetadata(
        name=sample_id,
        speaker1_audioURL="speaker_1.flac",
        speaker2_audioURL="speaker_2.flac",
    )


def write_audio(path: Path, source_audio: SourceAudio) -> None:
    path.write_bytes(source_audio.content)


def write_json(path: Path, model: FrozenBaseModel) -> None:
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def verify_prepared_layout(target_samples_root: Path, plan: TurnBenchPreparationPlan) -> None:
    expected_sample_ids = {sample.sample_id for sample in plan.samples}
    actual_sample_ids = {path.name for path in target_samples_root.iterdir() if path.is_dir()}
    if actual_sample_ids != expected_sample_ids:
        raise ValueError("Prepared sample directories do not match the migration plan.")
    for sample in plan.samples:
        sample_root = target_samples_root / sample.sample_id
        expected_paths = {
            sample_root / "speaker_1.flac",
            sample_root / "speaker_2.flac",
            sample_root / "metadata.json",
            sample_root / "source_annotations.json",
        }
        actual_paths = {path for path in sample_root.iterdir() if path.is_file()}
        if actual_paths != expected_paths:
            raise ValueError(f"Prepared sample has unexpected files: {sample.sample_id}")
        for filename, source in (
            ("speaker_1.flac", sample.speaker_1),
            ("speaker_2.flac", sample.speaker_2),
        ):
            content = (sample_root / filename).read_bytes()
            if hashlib.sha256(content).hexdigest() != source.sha256:
                raise ValueError(
                    f"Prepared audio hash does not match: {sample.sample_id}/{filename}"
                )
        if MinimalSampleMetadata.model_validate_json(
            (sample_root / "metadata.json").read_text()
        ) != minimal_metadata(sample.sample_id):
            raise ValueError(f"Prepared metadata is invalid: {sample.sample_id}")
        annotation_path = sample_root / "source_annotations.json"
        if (
            SourceSampleAnnotations.model_validate_json(annotation_path.read_text())
            != sample.annotations
        ):
            raise ValueError(f"Prepared annotations are invalid: {sample.sample_id}")


def write_restricted_record(path: Path, plan: TurnBenchPreparationPlan) -> None:
    if path.exists():
        raise ValueError(f"Restricted record already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "source_conversation_id": sample.source_conversation_id,
                "sample_id": sample.sample_id,
                "speaker_1_sha256": sample.speaker_1.sha256,
                "speaker_2_sha256": sample.speaker_2.sha256,
            },
            sort_keys=True,
        )
        for sample in plan.samples
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_empty_target(target_samples_root: Path) -> None:
    if target_samples_root.exists() and any(target_samples_root.iterdir()):
        raise ValueError(f"Target sample root is not empty: {target_samples_root}")


def required_string(row: Mapping[str, object], column_name: str) -> str:
    value = row[column_name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"TurnBench {column_name} must be a non-empty string.")
    return value
