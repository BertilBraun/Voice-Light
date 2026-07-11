from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.storage.base import StorageBackend

AUDIO_SUFFIXES = {".wav"}


@dataclass(frozen=True)
class DiscoveredSample:
    external_id: str
    speaker1_path: Path
    speaker2_path: Path


def discover_two_track_samples(storage: StorageBackend, root: Path) -> list[DiscoveredSample]:
    audio_paths = [
        Path(path)
        for path in storage.walk(str(root))
        if Path(path).suffix.lower() in AUDIO_SUFFIXES
    ]
    by_parent: dict[Path, list[Path]] = defaultdict(list)
    for audio_path in audio_paths:
        by_parent[audio_path.parent].append(audio_path)

    samples: list[DiscoveredSample] = []
    for parent_path, parent_audio_paths in sorted(by_parent.items()):
        selected_pair = select_speaker_pair(parent_audio_paths)
        if selected_pair is None:
            continue
        samples.append(
            DiscoveredSample(
                external_id=sample_id_from_root(root, parent_path, selected_pair[0]),
                speaker1_path=selected_pair[0],
                speaker2_path=selected_pair[1],
            )
        )
    return samples


def select_speaker_pair(audio_paths: list[Path]) -> tuple[Path, Path] | None:
    if len(audio_paths) < 2:
        return None
    speaker1_candidates = sorted(
        audio_path
        for audio_path in audio_paths
        if normalized_name(audio_path).endswith("speaker1")
        or normalized_name(audio_path).endswith("speaker_a")
    )
    speaker2_candidates = sorted(
        audio_path
        for audio_path in audio_paths
        if normalized_name(audio_path).endswith("speaker2")
        or normalized_name(audio_path).endswith("speaker_b")
    )
    if speaker1_candidates and speaker2_candidates:
        return (speaker1_candidates[0], speaker2_candidates[0])
    if len(audio_paths) == 2:
        sorted_audio_paths = sorted(audio_paths)
        return (sorted_audio_paths[0], sorted_audio_paths[1])
    return None


def normalized_name(path: Path) -> str:
    return path.stem.lower().replace("-", "_")


def sample_id_from_root(root: Path, parent_path: Path, speaker1_path: Path) -> str:
    try:
        relative_parent = parent_path.resolve().relative_to(root.resolve())
    except ValueError:
        return speaker1_path.stem
    if str(relative_parent) == ".":
        return speaker1_path.stem
    return relative_parent.as_posix()
