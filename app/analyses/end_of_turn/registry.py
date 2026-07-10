from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.analyses.end_of_turn.base import (
    EndOfTurnDetector,
    EndOfTurnDetectorInfo,
    EndOfTurnDetectorMode,
)
from app.analyses.end_of_turn.cache import (
    DetectorCacheKey,
    LeastRecentlyUsedCache,
)
from app.analyses.end_of_turn.detectors.naive_vad import (
    naive_vad_fast_detector,
    naive_vad_floor_detector,
)
from app.analyses.end_of_turn.detectors.pipecat_smart_turn_v2 import (
    pipecat_smart_turn_v2_detector,
)
from app.analyses.end_of_turn.detectors.pipecat_smart_turn_v3 import (
    pipecat_smart_turn_v3_detector,
)
from app.analyses.end_of_turn.service import BaselineResult

DETECTOR_CACHE_SIZE = 20


_DETECTOR_RESULT_CACHE = LeastRecentlyUsedCache[DetectorCacheKey, BaselineResult](
    capacity=DETECTOR_CACHE_SIZE
)
_DETECTORS: tuple[EndOfTurnDetector, ...] = (
    naive_vad_floor_detector(),
    naive_vad_fast_detector(),
    pipecat_smart_turn_v2_detector(),
    pipecat_smart_turn_v3_detector(),
)


def available_detectors() -> list[EndOfTurnDetectorInfo]:
    return [detector.info for detector in _DETECTORS]


def run_all_detectors(speaker1_path: Path, max_duration_seconds: float) -> list[BaselineResult]:
    with ThreadPoolExecutor(max_workers=len(_DETECTORS)) as executor:
        return list(
            executor.map(
                lambda detector: _run_detector_cached(
                    detector=detector,
                    speaker1_path=speaker1_path,
                    max_duration_seconds=max_duration_seconds,
                ),
                _DETECTORS,
            )
        )


def _run_detector_cached(
    detector: EndOfTurnDetector,
    speaker1_path: Path,
    max_duration_seconds: float,
) -> BaselineResult:
    cache_key = _cache_key(
        mode=detector.info.mode,
        speaker1_path=speaker1_path,
        max_duration_seconds=max_duration_seconds,
    )
    cached_result = _DETECTOR_RESULT_CACHE.get(cache_key=cache_key)
    if cached_result is not None:
        return cached_result

    result = detector.analyze(
        speaker1_path=speaker1_path,
        max_duration_seconds=max_duration_seconds,
    )
    _DETECTOR_RESULT_CACHE.put(cache_key=cache_key, cache_value=result)
    return result


def _cache_key(
    mode: EndOfTurnDetectorMode,
    speaker1_path: Path,
    max_duration_seconds: float,
) -> DetectorCacheKey:
    file_status = speaker1_path.stat()
    return DetectorCacheKey(
        mode=mode,
        speaker1_path=speaker1_path,
        max_duration_seconds=max_duration_seconds,
        modified_ns=file_status.st_mtime_ns,
        size_bytes=file_status.st_size,
    )
