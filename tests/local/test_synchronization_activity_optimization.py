from __future__ import annotations

import numpy as np
import pytest

from app.local.synchronization_review.activity_evaluation import (
    ActivityEstimatorConfiguration,
    ReviewedActivityCurve,
    evaluate_activity_offset_estimator,
)
from app.local.synchronization_review.activity_optimization import (
    ActivityLagCurve,
    full_recording_activity_lag_curve,
    stable_offset_estimate,
)
from app.local.synchronization_review.evaluation import OffsetLabel, ReviewProvenance


def test_activity_lag_curve_recovers_delayed_speaker2_activity() -> None:
    random_generator = np.random.default_rng(seed=42)
    speaker1 = random_generator.random(2_000) >= 0.7
    speaker2 = np.zeros_like(speaker1)
    speaker2[23:] = ~speaker1[:-23]

    curve = full_recording_activity_lag_curve(
        speaker1_mask=speaker1,
        speaker2_mask=speaker2,
        frame_duration_seconds=0.03,
        maximum_shift_seconds=2.0,
        analysis_margin_seconds=2.0,
    )
    estimate = stable_offset_estimate(
        curve=curve,
        overlap_weight=1.0,
        neighborhood_radius_seconds=0.0,
        competing_exclusion_seconds=0.5,
        basin_loss_tolerance=0.0005,
    )

    assert estimate.shift_seconds == pytest.approx(-0.69)
    assert estimate.improvement_over_zero > 0.1
    assert estimate.competing_margin > 0.1


def test_neighborhood_loss_prefers_broad_minimum_over_isolated_bin() -> None:
    curve = ActivityLagCurve(
        shifts_seconds=np.arange(-0.3, 0.4, 0.1),
        overlap_ratios=np.array(
            [1.0, 0.0, 1.0, 0.3, 0.2, 0.3, 1.0],
            dtype=np.float64,
        ),
        dual_silence_ratios=np.zeros(7, dtype=np.float64),
    )

    estimate = stable_offset_estimate(
        curve=curve,
        overlap_weight=1.0,
        neighborhood_radius_seconds=0.1,
        competing_exclusion_seconds=0.15,
        basin_loss_tolerance=0.05,
    )

    assert estimate.shift_seconds == pytest.approx(0.1)
    assert estimate.basin_width_seconds >= 0.1


def test_activity_evaluation_reports_recording_level_out_of_fold_metrics() -> None:
    configuration = ActivityEstimatorConfiguration(
        overlap_weight=1.0,
        neighborhood_radius_seconds=0.0,
        minimum_improvement_over_zero=0.0,
        minimum_competing_margin=0.0,
    )
    examples = tuple(
        _reviewed_curve(
            external_id=f"pmt_{index:03d}",
            reviewed_shift_seconds=-0.1 if index % 2 else 0.1,
        )
        for index in range(10)
    )

    report = evaluate_activity_offset_estimator(
        examples=examples,
        configurations=(configuration,),
        fold_count=5,
    )

    assert report.tuned_out_of_fold_metrics.mean_absolute_error_seconds == pytest.approx(0.0)
    assert report.tuned_out_of_fold_metrics.within_0_05_seconds == 1.0
    assert {sample.fold_index for sample in report.samples} == {0, 1, 2, 3, 4}
    assert all(fold.training_label_count == 8 for fold in report.folds)
    assert all(fold.validation_label_count == 2 for fold in report.folds)


def _reviewed_curve(
    external_id: str,
    reviewed_shift_seconds: float,
) -> ReviewedActivityCurve:
    shifts = np.arange(-0.2, 0.3, 0.1)
    losses = np.ones(len(shifts), dtype=np.float64)
    losses[int(np.argmin(np.abs(shifts - reviewed_shift_seconds)))] = 0.0
    return ReviewedActivityCurve(
        label=OffsetLabel(
            external_id=external_id,
            speaker2_shift_seconds=reviewed_shift_seconds,
            provenance=ReviewProvenance.DATABASE,
        ),
        curve=ActivityLagCurve(
            shifts_seconds=shifts,
            overlap_ratios=losses,
            dual_silence_ratios=np.zeros(len(shifts), dtype=np.float64),
        ),
    )
