from pathlib import Path

from app.analyses.end_of_turn.base import (
    EndOfTurnDetector,
    EndOfTurnDetectorInfo,
    EndOfTurnDetectorMode,
)
from app.analyses.end_of_turn.detectors.naive_vad import NaiveVadFloorDetector
from app.analyses.end_of_turn.service import BaselineResult


def available_detectors() -> list[EndOfTurnDetectorInfo]:
    return [detector.info for detector in _detectors()]


def run_detector(mode: EndOfTurnDetectorMode, speaker1_path: Path) -> BaselineResult:
    for detector in _detectors():
        if detector.info.mode == mode:
            return detector.analyze(speaker1_path=speaker1_path)
    raise ValueError(f"Unsupported end-of-turn detector mode: {mode}")


def _detectors() -> list[EndOfTurnDetector]:
    return [NaiveVadFloorDetector()]
