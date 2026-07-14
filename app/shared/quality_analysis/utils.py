from __future__ import annotations

import math


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def per_hour(count: int, duration_seconds: float) -> float:
    if duration_seconds <= 0.0:
        return 0.0
    return count * 3600.0 / duration_seconds


def score_between(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("high must be greater than low")
    return clamp((value - low) / (high - low), 0.0, 1.0)


def penalty_above(value: float, allowed: float, rejected: float) -> float:
    if rejected <= allowed:
        raise ValueError("rejected must be greater than allowed")
    if value <= allowed:
        return 1.0
    return 1.0 - score_between(value, allowed, rejected)


def binary_entropy(positive_ratio: float) -> float:
    ratio = clamp(positive_ratio, 0.0, 1.0)
    if ratio in {0.0, 1.0}:
        return 0.0
    return -(ratio * math.log2(ratio) + (1.0 - ratio) * math.log2(1.0 - ratio))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0
