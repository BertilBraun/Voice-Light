from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from app.analyses.end_of_turn.service import BaselineResult


class EndOfTurnDetectorMode(StrEnum):
    NAIVE_VAD_FLOOR = "naive_vad_floor"
    NAIVE_VAD_FAST = "naive_vad_fast"
    PIPECAT_SMART_TURN_V2 = "pipecat_smart_turn_v2"
    PIPECAT_SMART_TURN_V3 = "pipecat_smart_turn_v3"


@dataclass(frozen=True)
class EndOfTurnDetectorInfo:
    mode: EndOfTurnDetectorMode
    label: str
    description: str


class EndOfTurnDetector(Protocol):
    @property
    def info(self) -> EndOfTurnDetectorInfo:
        raise NotImplementedError

    def analyze(self, speaker1_path: Path) -> BaselineResult:
        raise NotImplementedError
