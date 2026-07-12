from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from app.frozen_base_config import FrozenBaseModel


class PolicyClass(StrEnum):
    WAIT = "wait"
    TAKE_TURN = "take_turn"
    MAY_BACKCHANNEL = "may_backchannel"
    YIELD = "yield"


class EventType(StrEnum):
    TARGET_SPEECH = "target_speech"
    PARTNER_SPEECH = "partner_speech"
    TURN_SHIFT = "turn_shift"
    HOLD = "hold"
    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"
    COOPERATIVE_OVERLAP = "cooperative_overlap"
    COMPETITIVE_OVERLAP = "competitive_overlap"
    LAUGHTER = "laughter"
    NONSPEECH_VOCALIZATION = "nonspeech_vocalization"


class AnnotationSource(StrEnum):
    HUMAN = "human"
    DATASET = "dataset"
    WEAK = "weak"


class TurnEvent(FrozenBaseModel):
    event_id: str
    event_type: EventType
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source: AnnotationSource

    @model_validator(mode="after")
    def validate_interval(self) -> TurnEvent:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Event end_seconds must be greater than start_seconds.")
        return self


class AlignedWord(FrozenBaseModel):
    speaker_id: str
    text: str
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class TurnTakingSample(FrozenBaseModel):
    sample_id: str
    conversation_id: str
    target_speaker_id: str
    target_audio_path: Path
    reference_audio_path: Path | None
    sample_rate_hz: int = Field(gt=0)
    context_start_seconds: float = Field(ge=0.0)
    decision_start_seconds: float = Field(ge=0.0)
    decision_end_seconds: float = Field(gt=0.0)
    events: tuple[TurnEvent, ...]
    words: tuple[AlignedWord, ...] = ()
    source_dataset: str
    source_license: str

    @model_validator(mode="after")
    def validate_times(self) -> TurnTakingSample:
        if not (
            self.context_start_seconds <= self.decision_start_seconds < self.decision_end_seconds
        ):
            raise ValueError(
                "Sample times must satisfy context_start <= decision_start < decision_end."
            )
        return self


def read_manifest(path: Path) -> tuple[TurnTakingSample, ...]:
    samples: list[TurnTakingSample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"Manifest line {line_number} must contain an object.")
        samples.append(TurnTakingSample.model_validate(value))
    return tuple(samples)


def write_manifest(path: Path, samples: Sequence[TurnTakingSample]) -> None:
    content = "\n".join(sample.model_dump_json() for sample in samples)
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")
