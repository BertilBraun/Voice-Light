from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from app.analyses.end_of_turn.base import EndOfTurnDetectorMode

CacheKeyType = TypeVar("CacheKeyType")
CacheValueType = TypeVar("CacheValueType")


@dataclass(frozen=True)
class DetectorCacheKey:
    mode: EndOfTurnDetectorMode
    speaker1_path: Path
    max_duration_seconds: float
    modified_ns: int
    size_bytes: int


@dataclass(frozen=True)
class WaveformCacheKey:
    wave_path: Path
    target_bins: int
    max_duration_seconds: float
    modified_ns: int
    size_bytes: int


class LeastRecentlyUsedCache(Generic[CacheKeyType, CacheValueType]):
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._values: OrderedDict[CacheKeyType, CacheValueType] = OrderedDict()
        self._lock = Lock()

    def get(self, cache_key: CacheKeyType) -> CacheValueType | None:
        with self._lock:
            cache_value = self._values.get(cache_key)
            if cache_value is None:
                return None
            self._values.move_to_end(cache_key)
            return cache_value

    def put(self, cache_key: CacheKeyType, cache_value: CacheValueType) -> None:
        with self._lock:
            self._values[cache_key] = cache_value
            self._values.move_to_end(cache_key)
            while len(self._values) > self._capacity:
                self._values.popitem(last=False)
