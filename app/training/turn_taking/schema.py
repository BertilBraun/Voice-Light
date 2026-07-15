from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import Field, model_validator

from app.shared.base_model import FrozenBaseModel


class EventTargetDistribution(FrozenBaseModel):
    turn_completion: float = Field(ge=0.0, le=1.0)
    continuation_pause: float = Field(ge=0.0, le=1.0)
    backchannel: float = Field(ge=0.0, le=1.0)
    interruption: float = Field(ge=0.0, le=1.0)
    other: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_distribution(self) -> EventTargetDistribution:
        if not math.isclose(sum(self.as_tuple()), 1.0, abs_tol=1e-5):
            raise ValueError("Event target probabilities must sum to one.")
        return self

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.turn_completion,
            self.continuation_pause,
            self.backchannel,
            self.interruption,
            self.other,
        )


class DecisionTarget(FrozenBaseModel):
    time_seconds: float = Field(ge=0.0)
    yield_probability: float = Field(ge=0.0, le=1.0)
    primary_reliability: float | None = Field(ge=0.0, le=1.0)
    event_distribution: EventTargetDistribution | None
    event_reliability: float | None = Field(ge=0.0, le=1.0)
    future_user_activity: tuple[
        bool | None,
        bool | None,
        bool | None,
        bool | None,
    ]


class ActivitySpan(FrozenBaseModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_interval(self) -> ActivitySpan:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Activity end_seconds must be greater than start_seconds.")
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
    annotation_reference_audio_path: Path | None
    sample_rate_hz: int = Field(gt=0)
    context_start_seconds: float = Field(ge=0.0)
    decision_start_seconds: float = Field(ge=0.0)
    decision_end_seconds: float = Field(gt=0.0)
    assistant_speech_spans: tuple[ActivitySpan, ...]
    decisions: tuple[DecisionTarget, ...]
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
        if any(
            decision.time_seconds < self.decision_start_seconds
            or decision.time_seconds >= self.decision_end_seconds
            for decision in self.decisions
        ):
            raise ValueError("Decision targets must fall inside the decision interval.")
        if any(
            span.start_seconds < self.context_start_seconds
            or span.end_seconds > self.decision_end_seconds
            for span in self.assistant_speech_spans
        ):
            raise ValueError("Assistant speech spans must fall inside the sample interval.")
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
