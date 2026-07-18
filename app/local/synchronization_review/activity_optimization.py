from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ActivityLagCurve:
    shifts_seconds: NDArray[np.float64]
    overlap_ratios: NDArray[np.float64]
    dual_silence_ratios: NDArray[np.float64]


@dataclass(frozen=True)
class StableOffsetEstimate:
    shift_seconds: float
    point_loss: float
    neighborhood_loss: float
    zero_shift_neighborhood_loss: float
    improvement_over_zero: float
    competing_margin: float
    basin_width_seconds: float


@dataclass(frozen=True)
class ActivityInterval:
    start_seconds: float
    end_seconds: float


def interval_activity_mask(
    intervals: tuple[ActivityInterval, ...],
    duration_seconds: float,
    frame_duration_seconds: float,
) -> NDArray[np.bool_]:
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    if frame_duration_seconds <= 0.0:
        raise ValueError("frame_duration_seconds must be positive")
    mask = np.zeros(math.ceil(duration_seconds / frame_duration_seconds), dtype=bool)
    for interval in intervals:
        _mark_activity(
            mask=mask,
            start_seconds=interval.start_seconds,
            end_seconds=interval.end_seconds,
            frame_duration_seconds=frame_duration_seconds,
        )
    return mask


def full_recording_activity_lag_curve(
    speaker1_mask: NDArray[np.bool_],
    speaker2_mask: NDArray[np.bool_],
    frame_duration_seconds: float,
    maximum_shift_seconds: float,
    analysis_margin_seconds: float,
) -> ActivityLagCurve:
    if frame_duration_seconds <= 0.0:
        raise ValueError("frame_duration_seconds must be positive")
    if maximum_shift_seconds <= 0.0:
        raise ValueError("maximum_shift_seconds must be positive")
    if analysis_margin_seconds < maximum_shift_seconds:
        raise ValueError("analysis_margin_seconds must cover maximum_shift_seconds")
    frame_count = min(len(speaker1_mask), len(speaker2_mask))
    maximum_shift_frames = round(maximum_shift_seconds / frame_duration_seconds)
    margin_frames = round(analysis_margin_seconds / frame_duration_seconds)
    if frame_count <= 2 * margin_frames:
        raise ValueError("Activity timelines are too short for the requested margins")

    speaker1 = speaker1_mask[:frame_count]
    speaker2 = speaker2_mask[:frame_count]
    evaluation_mask = np.zeros(frame_count, dtype=np.float64)
    evaluation_mask[margin_frames : frame_count - margin_frames] = 1.0
    evaluated_frame_count = float(np.sum(evaluation_mask))
    overlap_counts = _cross_correlation_at_lags(
        first=speaker1.astype(np.float64) * evaluation_mask,
        second=speaker2.astype(np.float64),
        maximum_lag=maximum_shift_frames,
    )
    dual_silence_counts = _cross_correlation_at_lags(
        first=(~speaker1).astype(np.float64) * evaluation_mask,
        second=(~speaker2).astype(np.float64),
        maximum_lag=maximum_shift_frames,
    )
    lag_frames = np.arange(
        -maximum_shift_frames,
        maximum_shift_frames + 1,
        dtype=np.float64,
    )
    return ActivityLagCurve(
        shifts_seconds=lag_frames * frame_duration_seconds,
        overlap_ratios=overlap_counts / evaluated_frame_count,
        dual_silence_ratios=dual_silence_counts / evaluated_frame_count,
    )


def stable_offset_estimate(
    curve: ActivityLagCurve,
    overlap_weight: float,
    neighborhood_radius_seconds: float,
    competing_exclusion_seconds: float,
    basin_loss_tolerance: float,
) -> StableOffsetEstimate:
    if overlap_weight <= 0.0:
        raise ValueError("overlap_weight must be positive")
    if neighborhood_radius_seconds < 0.0:
        raise ValueError("neighborhood_radius_seconds cannot be negative")
    if competing_exclusion_seconds <= 0.0:
        raise ValueError("competing_exclusion_seconds must be positive")
    if basin_loss_tolerance < 0.0:
        raise ValueError("basin_loss_tolerance cannot be negative")
    frame_duration_seconds = _frame_duration_seconds(curve.shifts_seconds)
    point_losses = overlap_weight * curve.overlap_ratios + curve.dual_silence_ratios
    neighborhood_radius_frames = round(neighborhood_radius_seconds / frame_duration_seconds)
    neighborhood_losses = _centered_mean(
        values=point_losses,
        radius=neighborhood_radius_frames,
    )
    best_index = int(np.argmin(neighborhood_losses))
    exclusion_frames = max(
        1,
        round(competing_exclusion_seconds / frame_duration_seconds),
    )
    competing_losses = np.concatenate(
        (
            neighborhood_losses[: max(0, best_index - exclusion_frames)],
            neighborhood_losses[min(len(neighborhood_losses), best_index + exclusion_frames + 1) :],
        )
    )
    competing_loss = (
        float(np.min(competing_losses))
        if len(competing_losses)
        else float(neighborhood_losses[best_index])
    )
    zero_index = int(np.argmin(np.abs(curve.shifts_seconds)))
    basin_start = best_index
    basin_end = best_index
    basin_limit = float(neighborhood_losses[best_index]) + basin_loss_tolerance
    while basin_start > 0 and neighborhood_losses[basin_start - 1] <= basin_limit:
        basin_start -= 1
    while (
        basin_end + 1 < len(neighborhood_losses)
        and neighborhood_losses[basin_end + 1] <= basin_limit
    ):
        basin_end += 1
    return StableOffsetEstimate(
        shift_seconds=float(curve.shifts_seconds[best_index]),
        point_loss=float(point_losses[best_index]),
        neighborhood_loss=float(neighborhood_losses[best_index]),
        zero_shift_neighborhood_loss=float(neighborhood_losses[zero_index]),
        improvement_over_zero=max(
            0.0,
            float(neighborhood_losses[zero_index] - neighborhood_losses[best_index]),
        ),
        competing_margin=max(
            0.0,
            competing_loss - float(neighborhood_losses[best_index]),
        ),
        basin_width_seconds=(basin_end - basin_start + 1) * frame_duration_seconds,
    )


def _mark_activity(
    mask: NDArray[np.bool_],
    start_seconds: float,
    end_seconds: float,
    frame_duration_seconds: float,
) -> None:
    start_index = max(0, math.floor(start_seconds / frame_duration_seconds))
    end_index = min(len(mask), math.ceil(end_seconds / frame_duration_seconds))
    mask[start_index:end_index] = True


def _cross_correlation_at_lags(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    maximum_lag: int,
) -> NDArray[np.float64]:
    convolution_length = len(first) + len(second) - 1
    transform_length = 1 << (convolution_length - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(first, transform_length) * np.conjugate(np.fft.rfft(second, transform_length)),
        transform_length,
    )
    negative = correlation[transform_length - maximum_lag : transform_length]
    nonnegative = correlation[: maximum_lag + 1]
    return np.concatenate((negative, nonnegative))


def _centered_mean(
    values: NDArray[np.float64],
    radius: int,
) -> NDArray[np.float64]:
    if radius == 0:
        return values.copy()
    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    sums = np.convolve(values, kernel, mode="same")
    counts = np.convolve(np.ones_like(values), kernel, mode="same")
    return sums / counts


def _frame_duration_seconds(shifts_seconds: NDArray[np.float64]) -> float:
    if len(shifts_seconds) < 2:
        raise ValueError("Lag curve must contain at least two shifts")
    return float(shifts_seconds[1] - shifts_seconds[0])
