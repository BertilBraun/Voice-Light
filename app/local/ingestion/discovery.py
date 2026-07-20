from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from app.shared.storage.base import StorageBackend

AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".flac"})
MAXIMUM_TRACK_DURATION_DIFFERENCE_SECONDS = 20.0
MAXIMUM_TRACK_DURATION_DIFFERENCE_RATIO = 0.01


class DatasetLayout(StrEnum):
    TWO_AUDIO_FILES = "two_audio_files"
    MIX_AND_TWO_SPEAKERS = "mix_and_two_speakers"


@dataclass(frozen=True)
class DiscoveredSample:
    external_id: str
    speaker1_path: str
    speaker2_path: str


def track_durations_are_compatible(
    speaker1_duration_seconds: float,
    speaker2_duration_seconds: float,
) -> bool:
    shorter_duration_seconds = min(speaker1_duration_seconds, speaker2_duration_seconds)
    difference_seconds = abs(speaker1_duration_seconds - speaker2_duration_seconds)
    return (
        difference_seconds <= MAXIMUM_TRACK_DURATION_DIFFERENCE_SECONDS
        and difference_seconds <= shorter_duration_seconds * MAXIMUM_TRACK_DURATION_DIFFERENCE_RATIO
    )


def discover_samples(
    storage: StorageBackend, root: str, layout: DatasetLayout
) -> list[DiscoveredSample]:
    audio_paths = [
        storage_path(path)
        for path in storage.walk(root)
        if PurePosixPath(path).suffix.lower() in AUDIO_SUFFIXES
    ]
    paths_by_parent: dict[PurePosixPath, list[PurePosixPath]] = defaultdict(list)
    for audio_path in audio_paths:
        paths_by_parent[audio_path.parent].append(audio_path)

    samples: list[DiscoveredSample] = []
    for parent_path, parent_audio_paths in sorted(paths_by_parent.items()):
        selected_pair = select_speaker_pair(parent_audio_paths, layout)
        if selected_pair is None:
            continue
        samples.append(
            DiscoveredSample(
                external_id=sample_id_from_root(root, parent_path, selected_pair[0]),
                speaker1_path=str(selected_pair[0]),
                speaker2_path=str(selected_pair[1]),
            )
        )
    return samples


def select_speaker_pair(
    audio_paths: list[PurePosixPath], layout: DatasetLayout
) -> tuple[PurePosixPath, PurePosixPath] | None:
    match layout:
        case DatasetLayout.TWO_AUDIO_FILES:
            if len(audio_paths) != 2:
                return None
            candidates = audio_paths
        case DatasetLayout.MIX_AND_TWO_SPEAKERS:
            if len(audio_paths) != 3:
                return None
            mix_paths = [path for path in audio_paths if is_mix_path(path)]
            if len(mix_paths) != 1:
                return None
            candidates = [path for path in audio_paths if path != mix_paths[0]]
    return order_speaker_paths(candidates)


def order_speaker_paths(
    audio_paths: list[PurePosixPath],
) -> tuple[PurePosixPath, PurePosixPath]:
    speaker1_candidates = sorted(
        path
        for path in audio_paths
        if normalized_name(path).endswith("speaker1") or normalized_name(path).endswith("speaker_a")
    )
    speaker2_candidates = sorted(
        path
        for path in audio_paths
        if normalized_name(path).endswith("speaker2") or normalized_name(path).endswith("speaker_b")
    )
    if len(speaker1_candidates) == 1 and len(speaker2_candidates) == 1:
        return (speaker1_candidates[0], speaker2_candidates[0])
    first_path, second_path = sorted(audio_paths)
    return (first_path, second_path)


def is_mix_path(path: PurePosixPath) -> bool:
    normalized_parts = normalized_name(path).split("_")
    return normalized_parts[-1] == "mix" or normalized_parts[0] == "mix"


def normalized_name(path: PurePosixPath) -> str:
    return path.stem.lower().replace("-", "_").replace(".", "_")


def sample_id_from_root(root: str, parent_path: PurePosixPath, speaker1_path: PurePosixPath) -> str:
    root_path = storage_path(root)
    try:
        relative_parent = parent_path.relative_to(root_path)
    except ValueError:
        return speaker1_path.stem
    if str(relative_parent) == ".":
        return speaker1_path.stem
    return relative_parent.as_posix()


def storage_path(path: str) -> PurePosixPath:
    return PurePosixPath(path.replace("\\", "/"))
