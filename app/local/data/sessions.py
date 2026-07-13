from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from app.local.config import SESSIONS_ROOT


class SpeakerName(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"


@dataclass(frozen=True)
class SessionEntry:
    identifier: str
    duration_seconds: float
    topic: str
    sample_rate: int
    speaker1_audio_url: str
    speaker2_audio_url: str


def session_to_json(session_entry: SessionEntry) -> dict[str, object]:
    return asdict(session_entry)


def list_sessions() -> list[SessionEntry]:
    session_entries: list[SessionEntry] = []
    for session_directory in SESSIONS_ROOT.glob("pmt_*"):
        if not session_directory.is_dir():
            continue
        identifier = session_directory.name
        speaker1_path = session_directory / f"{identifier}_speaker1.wav"
        speaker2_path = session_directory / f"{identifier}_speaker2.wav"
        metadata_path = session_directory / f"{identifier}.json"
        if not speaker1_path.exists() or not speaker2_path.exists() or not metadata_path.exists():
            continue
        session_entries.append(_session_entry_from_metadata(metadata_path=metadata_path))

    return sorted(session_entries, key=lambda session_entry: session_entry.identifier)


def session_audio_path(identifier: str, speaker_name: SpeakerName) -> Path:
    if not identifier.startswith("pmt_") or "/" in identifier or "\\" in identifier:
        raise ValueError("Invalid session identifier.")
    wave_path = SESSIONS_ROOT / identifier / f"{identifier}_{speaker_name.value}.wav"
    if not wave_path.exists():
        raise ValueError(f"Missing audio file for {identifier} {speaker_name.value}.")
    return wave_path


def session_metadata_path(identifier: str) -> Path:
    if not identifier.startswith("pmt_") or "/" in identifier or "\\" in identifier:
        raise ValueError("Invalid session identifier.")
    metadata_path = SESSIONS_ROOT / identifier / f"{identifier}.json"
    if not metadata_path.exists():
        raise ValueError(f"Missing metadata file for {identifier}.")
    return metadata_path


def _session_entry_from_metadata(metadata_path: Path) -> SessionEntry:
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata_payload = json.load(metadata_file)

    identifier = str(metadata_payload["name"])
    return SessionEntry(
        identifier=identifier,
        duration_seconds=float(metadata_payload["durationSeconds"]),
        topic=str(metadata_payload["topic"]),
        sample_rate=int(metadata_payload["sampleRate"]),
        speaker1_audio_url=f"/api/audio/{identifier}/{SpeakerName.SPEAKER1.value}",
        speaker2_audio_url=f"/api/audio/{identifier}/{SpeakerName.SPEAKER2.value}",
    )
