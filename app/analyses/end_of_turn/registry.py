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
from app.analyses.end_of_turn.detectors.livekit_v1_mini import livekit_v1_mini_detector
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
from app.analyses.end_of_turn.detectors.silero_vad import silero_vad_detector
from app.analyses.end_of_turn.detectors.transcript_gap import transcript_gap_detector
from app.analyses.end_of_turn.detectors.turnsense import turnsense_detector
from app.analyses.end_of_turn.service import BaselineResult

DETECTOR_CACHE_SIZE = 20


_DETECTOR_RESULT_CACHE = LeastRecentlyUsedCache[DetectorCacheKey, BaselineResult](
    capacity=DETECTOR_CACHE_SIZE
)
_DETECTORS: tuple[EndOfTurnDetector, ...] = (
    naive_vad_floor_detector(),
    naive_vad_fast_detector(),
    silero_vad_detector(),
    livekit_v1_mini_detector(),
    pipecat_smart_turn_v2_detector(),
    pipecat_smart_turn_v3_detector(),
    transcript_gap_detector(),
    turnsense_detector(),
)


def available_detectors() -> list[EndOfTurnDetectorInfo]:
    return [detector.info for detector in _DETECTORS]


def run_all_detectors(speaker1_path: Path) -> list[BaselineResult]:
    return run_detectors(
        speaker1_path=speaker1_path,
        selected_modes=[detector.info.mode for detector in _DETECTORS],
    )


def run_detectors(
    speaker1_path: Path,
    selected_modes: list[EndOfTurnDetectorMode],
) -> list[BaselineResult]:
    selected_mode_set = set(selected_modes)
    selected_detectors = [
        detector for detector in _DETECTORS if detector.info.mode in selected_mode_set
    ]
    if not selected_detectors:
        return []

    results_by_mode = _run_detectors_parallel(
        detectors=selected_detectors,
        speaker1_path=speaker1_path,
    )
    return [results_by_mode[detector.info.mode] for detector in selected_detectors]


def _run_detectors_parallel(
    detectors: list[EndOfTurnDetector],
    speaker1_path: Path,
) -> dict[EndOfTurnDetectorMode, BaselineResult]:
    if not detectors:
        return {}

    with ThreadPoolExecutor(max_workers=len(detectors)) as executor:
        detector_results = executor.map(
            lambda detector: _run_detector_cached(
                detector=detector,
                speaker1_path=speaker1_path,
            ),
            detectors,
        )
        return {
            detector.info.mode: result
            for detector, result in zip(detectors, detector_results, strict=True)
        }


def _run_detector_cached(
    detector: EndOfTurnDetector,
    speaker1_path: Path,
) -> BaselineResult:
    cache_key = _cache_key(
        mode=detector.info.mode,
        speaker1_path=speaker1_path,
    )
    cached_result = _DETECTOR_RESULT_CACHE.get(cache_key=cache_key)
    if cached_result is not None:
        return cached_result

    result = detector.analyze(speaker1_path=speaker1_path)
    _DETECTOR_RESULT_CACHE.put(cache_key=cache_key, cache_value=result)
    return result


def _cache_key(
    mode: EndOfTurnDetectorMode,
    speaker1_path: Path,
) -> DetectorCacheKey:
    file_status = speaker1_path.stat()
    return DetectorCacheKey(
        mode=mode,
        speaker1_path=speaker1_path,
        modified_ns=file_status.st_mtime_ns,
        size_bytes=file_status.st_size,
    )
