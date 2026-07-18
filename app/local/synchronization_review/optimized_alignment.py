from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.local.analyses.end_of_turn.detectors.naive_vad import run_naive_vad_floor
from app.local.synchronization_review.activity_optimization import (
    ActivityInterval,
    ActivityLagCurve,
    StableOffsetEstimate,
    full_recording_activity_lag_curve,
    interval_activity_mask,
    stable_offset_estimate,
)
from app.local.synchronization_review.models import (
    OffsetPattern,
    SynchronizationCandidate,
    SynchronizationEvidence,
    SynchronizationEvidenceSource,
    SynchronizationWindowEstimate,
)
from app.shared.audio.loading import probe_local_audio_metadata

ESTIMATOR_VERSION = "full-recording-vad-basin-v2"
FRAME_DURATION_SECONDS = 0.03
MAXIMUM_SHIFT_SECONDS = 12.0
ANALYSIS_MARGIN_SECONDS = 12.0
OVERLAP_WEIGHT = 1.5
NEIGHBORHOOD_RADIUS_SECONDS = 0.15
COMPETING_EXCLUSION_SECONDS = 0.5
BASIN_LOSS_TOLERANCE = 0.0005
MINIMUM_IMPROVEMENT_OVER_ZERO = 0.002
MINIMUM_COMPETING_MARGIN = 0.0001
WINDOW_DURATION_SECONDS = 300.0
MINIMUM_WINDOW_DURATION_SECONDS = 180.0
MINIMUM_DRIFT_WINDOW_COUNT = 4
MINIMUM_DRIFT_RANGE_SECONDS = 1.5
MINIMUM_DRIFT_END_TO_END_SECONDS = 1.0
MAXIMUM_DRIFT_RESIDUAL_SECONDS = 0.6


@dataclass(frozen=True)
class ActivityOffsetAnalysis:
    estimate: StableOffsetEstimate
    overlap_reduction: float
    dual_silence_reduction: float
    accepted: bool


def optimized_alignment_candidate(
    candidate: SynchronizationCandidate,
    speaker1_path: Path,
    speaker2_path: Path,
) -> SynchronizationCandidate:
    speaker1_mask, speaker2_mask = full_recording_audio_activity_masks(
        speaker1_path=speaker1_path,
        speaker2_path=speaker2_path,
    )
    full_analysis = _analyze_masks(
        speaker1_mask=speaker1_mask,
        speaker2_mask=speaker2_mask,
    )
    window_estimates = _window_estimates(
        speaker1_mask=speaker1_mask,
        speaker2_mask=speaker2_mask,
    )
    drift_warning = _drift_warning(windows=window_estimates)
    predicted_shift_seconds = (
        _quantized_shift(full_analysis.estimate.shift_seconds) if full_analysis.accepted else 0.0
    )
    evidence = SynchronizationEvidence(
        source=SynchronizationEvidenceSource.AUDIO_ACTIVITY,
        estimated_b_shift_seconds=full_analysis.estimate.shift_seconds,
        bad_state_improvement=full_analysis.estimate.improvement_over_zero,
        overlap_reduction=full_analysis.overlap_reduction,
        silence_reduction=full_analysis.dual_silence_reduction,
    )
    existing_evidence = tuple(
        item
        for item in candidate.evidence
        if item.source is not SynchronizationEvidenceSource.AUDIO_ACTIVITY
    )
    existing_windows = tuple(
        window
        for window in candidate.window_estimates
        if window.source is not SynchronizationEvidenceSource.AUDIO_ACTIVITY
    )
    is_variable = drift_warning is not None
    return candidate.model_copy(
        update={
            "is_offset_candidate": (full_analysis.accepted and abs(predicted_shift_seconds) >= 0.3),
            "estimated_b_shift_seconds": predicted_shift_seconds,
            "full_recording_estimated_b_shift_seconds": (full_analysis.estimate.shift_seconds),
            "offset_confidence_score": _confidence_score(analysis=full_analysis),
            "offset_pattern": (
                OffsetPattern.VARIABLE
                if is_variable
                else (OffsetPattern.CONSTANT if full_analysis.accepted else OffsetPattern.UNCERTAIN)
            ),
            "static_offset_valid": full_analysis.accepted and not is_variable,
            "drift_warning": drift_warning,
            "evidence": (*existing_evidence, evidence),
            "window_estimates": (*existing_windows, *window_estimates),
        }
    )


def full_recording_audio_activity_masks(
    speaker1_path: Path,
    speaker2_path: Path,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    speaker1_metadata = probe_local_audio_metadata(path=speaker1_path)
    speaker2_metadata = probe_local_audio_metadata(path=speaker2_path)
    duration_seconds = min(
        speaker1_metadata.duration_seconds,
        speaker2_metadata.duration_seconds,
    )
    speaker1_mask = _vad_activity_mask(
        wave_path=speaker1_path,
        duration_seconds=duration_seconds,
    )
    speaker2_mask = _vad_activity_mask(
        wave_path=speaker2_path,
        duration_seconds=duration_seconds,
    )
    return speaker1_mask, speaker2_mask


def _vad_activity_mask(
    wave_path: Path,
    duration_seconds: float,
) -> NDArray[np.bool_]:
    vad_result = run_naive_vad_floor(
        wave_path=wave_path,
        result_name="synchronization_audio_activity",
        description="Full-recording audio activity for synchronization",
        frame_seconds=FRAME_DURATION_SECONDS,
        min_speech_seconds=0.12,
        min_silence_seconds=0.4,
    )
    return interval_activity_mask(
        intervals=tuple(
            ActivityInterval(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
            )
            for segment in vad_result.speech_segments
        ),
        duration_seconds=duration_seconds,
        frame_duration_seconds=FRAME_DURATION_SECONDS,
    )


def _analyze_masks(
    speaker1_mask: NDArray[np.bool_],
    speaker2_mask: NDArray[np.bool_],
) -> ActivityOffsetAnalysis:
    curve = full_recording_activity_lag_curve(
        speaker1_mask=speaker1_mask,
        speaker2_mask=speaker2_mask,
        frame_duration_seconds=FRAME_DURATION_SECONDS,
        maximum_shift_seconds=MAXIMUM_SHIFT_SECONDS,
        analysis_margin_seconds=ANALYSIS_MARGIN_SECONDS,
    )
    estimate = stable_offset_estimate(
        curve=curve,
        overlap_weight=OVERLAP_WEIGHT,
        neighborhood_radius_seconds=NEIGHBORHOOD_RADIUS_SECONDS,
        competing_exclusion_seconds=COMPETING_EXCLUSION_SECONDS,
        basin_loss_tolerance=BASIN_LOSS_TOLERANCE,
    )
    overlap_reduction, dual_silence_reduction = _state_reductions(
        curve=curve,
        shift_seconds=estimate.shift_seconds,
    )
    return ActivityOffsetAnalysis(
        estimate=estimate,
        overlap_reduction=overlap_reduction,
        dual_silence_reduction=dual_silence_reduction,
        accepted=(
            estimate.improvement_over_zero >= MINIMUM_IMPROVEMENT_OVER_ZERO
            and estimate.competing_margin >= MINIMUM_COMPETING_MARGIN
        ),
    )


def _window_estimates(
    speaker1_mask: NDArray[np.bool_],
    speaker2_mask: NDArray[np.bool_],
) -> tuple[SynchronizationWindowEstimate, ...]:
    window_frame_count = round(WINDOW_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    minimum_window_frame_count = round(MINIMUM_WINDOW_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    estimates: list[SynchronizationWindowEstimate] = []
    shared_frame_count = min(len(speaker1_mask), len(speaker2_mask))
    for start_index in range(0, shared_frame_count, window_frame_count):
        end_index = min(shared_frame_count, start_index + window_frame_count)
        if end_index - start_index < minimum_window_frame_count:
            continue
        analysis = _analyze_masks(
            speaker1_mask=speaker1_mask[start_index:end_index],
            speaker2_mask=speaker2_mask[start_index:end_index],
        )
        estimates.append(
            SynchronizationWindowEstimate(
                source=SynchronizationEvidenceSource.AUDIO_ACTIVITY,
                start_seconds=start_index * FRAME_DURATION_SECONDS,
                end_seconds=end_index * FRAME_DURATION_SECONDS,
                estimated_b_shift_seconds=analysis.estimate.shift_seconds,
                bad_state_improvement=analysis.estimate.improvement_over_zero,
                overlap_reduction=analysis.overlap_reduction,
                silence_reduction=analysis.dual_silence_reduction,
                meaningful=analysis.accepted,
            )
        )
    return tuple(estimates)


def _state_reductions(
    curve: ActivityLagCurve,
    shift_seconds: float,
) -> tuple[float, float]:
    zero_index = int(np.argmin(np.abs(curve.shifts_seconds)))
    selected_index = int(np.argmin(np.abs(curve.shifts_seconds - shift_seconds)))
    return (
        max(
            0.0,
            float(curve.overlap_ratios[zero_index] - curve.overlap_ratios[selected_index]),
        ),
        max(
            0.0,
            float(
                curve.dual_silence_ratios[zero_index] - curve.dual_silence_ratios[selected_index]
            ),
        ),
    )


def _drift_warning(
    windows: tuple[SynchronizationWindowEstimate, ...],
) -> str | None:
    meaningful = tuple(window for window in windows if window.meaningful)
    if len(meaningful) < MINIMUM_DRIFT_WINDOW_COUNT:
        return None
    shifts = np.array(
        [window.estimated_b_shift_seconds for window in meaningful],
        dtype=np.float64,
    )
    if float(np.max(shifts) - np.min(shifts)) < MINIMUM_DRIFT_RANGE_SECONDS:
        return None
    midpoints = np.array(
        [(window.start_seconds + window.end_seconds) / 2.0 for window in meaningful],
        dtype=np.float64,
    )
    slope, intercept = np.polyfit(midpoints, shifts, 1)
    fitted = slope * midpoints + intercept
    end_to_end_seconds = abs(float(slope * (midpoints[-1] - midpoints[0])))
    median_residual_seconds = statistics.median(
        abs(float(actual - predicted)) for actual, predicted in zip(shifts, fitted, strict=True)
    )
    if (
        end_to_end_seconds >= MINIMUM_DRIFT_END_TO_END_SECONDS
        and median_residual_seconds <= MAXIMUM_DRIFT_RESIDUAL_SECONDS
    ):
        return (
            "Audio-activity windows indicate likely clock drift "
            f"({end_to_end_seconds:.2f} s end-to-end); no single static offset is safe."
        )
    return "Audio-activity windows disagree structurally; no single static offset is safe."


def _confidence_score(analysis: ActivityOffsetAnalysis) -> float:
    if not analysis.accepted:
        return 0.0
    improvement_score = min(
        1.0,
        analysis.estimate.improvement_over_zero / 0.01,
    )
    margin_score = min(
        1.0,
        analysis.estimate.competing_margin / 0.002,
    )
    basin_score = min(
        1.0,
        analysis.estimate.basin_width_seconds / 0.3,
    )
    return math.sqrt(improvement_score * margin_score) * basin_score


def _quantized_shift(shift_seconds: float) -> float:
    return round(round(shift_seconds / FRAME_DURATION_SECONDS) * FRAME_DURATION_SECONDS, 2)
