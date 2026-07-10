from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from app.analyses.end_of_turn.base import (
    EndOfTurnDetector,
    EndOfTurnDetectorInfo,
    EndOfTurnDetectorMode,
)
from app.analyses.end_of_turn.detectors.naive_vad import (
    naive_vad_fast_detector,
    naive_vad_floor_detector,
)
from app.analyses.end_of_turn.service import BaselineResult

DETECTOR_CACHE_SIZE = 20


@dataclass(frozen=True)
class DetectorCacheKey:
    mode: EndOfTurnDetectorMode
    speaker1_path: Path
    modified_ns: int
    size_bytes: int


class DetectorResultCache:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._results: OrderedDict[DetectorCacheKey, BaselineResult] = OrderedDict()
        self._lock = Lock()

    def get(self, cache_key: DetectorCacheKey) -> BaselineResult | None:
        with self._lock:
            result = self._results.get(cache_key)
            if result is None:
                return None
            self._results.move_to_end(cache_key)
            return result

    def put(self, cache_key: DetectorCacheKey, result: BaselineResult) -> None:
        with self._lock:
            self._results[cache_key] = result
            self._results.move_to_end(cache_key)
            while len(self._results) > self._capacity:
                self._results.popitem(last=False)


_DETECTOR_RESULT_CACHE = DetectorResultCache(capacity=DETECTOR_CACHE_SIZE)
_DETECTORS: tuple[EndOfTurnDetector, ...] = (
    naive_vad_floor_detector(),
    naive_vad_fast_detector(),
)


def available_detectors() -> list[EndOfTurnDetectorInfo]:
    return [detector.info for detector in _DETECTORS]


def run_all_detectors(speaker1_path: Path) -> list[BaselineResult]:
    with ThreadPoolExecutor(max_workers=len(_DETECTORS)) as executor:
        return list(
            executor.map(lambda detector: _run_detector_cached(detector, speaker1_path), _DETECTORS)
        )


def _run_detector_cached(
    detector: EndOfTurnDetector,
    speaker1_path: Path,
) -> BaselineResult:
    cache_key = _cache_key(mode=detector.info.mode, speaker1_path=speaker1_path)
    cached_result = _DETECTOR_RESULT_CACHE.get(cache_key=cache_key)
    if cached_result is not None:
        return cached_result

    result = detector.analyze(speaker1_path=speaker1_path)
    _DETECTOR_RESULT_CACHE.put(cache_key=cache_key, result=result)
    return result


def _cache_key(mode: EndOfTurnDetectorMode, speaker1_path: Path) -> DetectorCacheKey:
    file_status = speaker1_path.stat()
    return DetectorCacheKey(
        mode=mode,
        speaker1_path=speaker1_path,
        modified_ns=file_status.st_mtime_ns,
        size_bytes=file_status.st_size,
    )
