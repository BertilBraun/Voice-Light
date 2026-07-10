from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptTurn:
    speaker: str
    text: str
    start_seconds: float
    end_seconds: float


def transcript_turn_to_json(transcript_turn: TranscriptTurn) -> dict[str, object]:
    return asdict(transcript_turn)


def read_transcript_turns(metadata_path: Path) -> list[TranscriptTurn]:
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = _required_mapping(value=raw_metadata, name=str(metadata_path))
    raw_turns = _required_sequence(source=metadata, key="speakerTranscript")
    transcript_turns: list[TranscriptTurn] = []
    for raw_turn in raw_turns:
        turn = _required_mapping(value=raw_turn, name="speakerTranscript[]")
        transcript_turn = TranscriptTurn(
            speaker=_required_string(source=turn, key="speaker"),
            text=_required_string(source=turn, key="text"),
            start_seconds=_required_number(source=turn, key="startTime"),
            end_seconds=_required_number(source=turn, key="endTime"),
        )
        if transcript_turn.text:
            transcript_turns.append(transcript_turn)
    return transcript_turns


def _required_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected {name} to be an object.")
    return value


def _required_sequence(source: Mapping[str, object], key: str) -> Sequence[object]:
    value = source.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Expected {key} to be a list.")
    return value


def _required_string(source: Mapping[str, object], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected {key} to be a string.")
    return value


def _required_number(source: Mapping[str, object], key: str) -> float:
    value = source.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"Expected {key} to be a number.")
    return float(value)
