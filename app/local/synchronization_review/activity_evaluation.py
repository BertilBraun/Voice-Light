from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from app.local.data.sessions import SpeakerName, session_audio_path
from app.local.synchronization_review.activity_optimization import (
    ActivityLagCurve,
    full_recording_activity_lag_curve,
    stable_offset_estimate,
)
from app.local.synchronization_review.evaluation import (
    DEFAULT_FOLD_COUNT,
    DEFAULT_SPLIT_SEED,
    OffsetErrorMetrics,
    OffsetLabel,
    offset_error_metrics,
    stratified_label_fold_assignments,
)
from app.local.synchronization_review.optimized_alignment import (
    ANALYSIS_MARGIN_SECONDS,
    BASIN_LOSS_TOLERANCE,
    COMPETING_EXCLUSION_SECONDS,
    ESTIMATOR_VERSION,
    FRAME_DURATION_SECONDS,
    MAXIMUM_SHIFT_SECONDS,
    MINIMUM_COMPETING_MARGIN,
    MINIMUM_IMPROVEMENT_OVER_ZERO,
    NEIGHBORHOOD_RADIUS_SECONDS,
    OVERLAP_WEIGHT,
    full_recording_audio_activity_masks,
)
from app.shared.base_model import FrozenBaseModel


class ActivityEstimatorConfiguration(FrozenBaseModel):
    overlap_weight: float = Field(gt=0.0)
    neighborhood_radius_seconds: float = Field(ge=0.0)
    minimum_improvement_over_zero: float = Field(ge=0.0)
    minimum_competing_margin: float = Field(ge=0.0)


@dataclass(frozen=True)
class ReviewedActivityCurve:
    label: OffsetLabel
    curve: ActivityLagCurve


class ActivitySampleEvaluation(FrozenBaseModel):
    external_id: str
    fold_index: int
    reviewed_shift_seconds: float
    predicted_shift_seconds: float
    absolute_error_seconds: float


class ActivityFoldEvaluation(FrozenBaseModel):
    fold_index: int
    training_label_count: int
    validation_label_count: int
    selected_configuration: ActivityEstimatorConfiguration
    training_metrics: OffsetErrorMetrics
    validation_metrics: OffsetErrorMetrics


class ActivityOffsetEvaluationReport(FrozenBaseModel):
    estimator_version: str
    split_seed: str
    fold_count: int
    configuration_count: int
    production_configuration: ActivityEstimatorConfiguration
    production_all_reviewed_metrics: OffsetErrorMetrics
    selected_full_training_configuration: ActivityEstimatorConfiguration
    selected_full_training_metrics: OffsetErrorMetrics
    tuned_out_of_fold_metrics: OffsetErrorMetrics
    folds: tuple[ActivityFoldEvaluation, ...]
    samples: tuple[ActivitySampleEvaluation, ...]


PRODUCTION_CONFIGURATION = ActivityEstimatorConfiguration(
    overlap_weight=OVERLAP_WEIGHT,
    neighborhood_radius_seconds=NEIGHBORHOOD_RADIUS_SECONDS,
    minimum_improvement_over_zero=MINIMUM_IMPROVEMENT_OVER_ZERO,
    minimum_competing_margin=MINIMUM_COMPETING_MARGIN,
)


def reviewed_activity_curves(
    labels: tuple[OffsetLabel, ...],
) -> tuple[ReviewedActivityCurve, ...]:
    curves: list[ReviewedActivityCurve] = []
    for label in labels:
        speaker1_mask, speaker2_mask = full_recording_audio_activity_masks(
            speaker1_path=session_audio_path(
                identifier=label.external_id,
                speaker_name=SpeakerName.SPEAKER1,
            ),
            speaker2_path=session_audio_path(
                identifier=label.external_id,
                speaker_name=SpeakerName.SPEAKER2,
            ),
        )
        curves.append(
            ReviewedActivityCurve(
                label=label,
                curve=full_recording_activity_lag_curve(
                    speaker1_mask=speaker1_mask,
                    speaker2_mask=speaker2_mask,
                    frame_duration_seconds=FRAME_DURATION_SECONDS,
                    maximum_shift_seconds=MAXIMUM_SHIFT_SECONDS,
                    analysis_margin_seconds=ANALYSIS_MARGIN_SECONDS,
                ),
            )
        )
    return tuple(curves)


def activity_configuration_grid() -> tuple[ActivityEstimatorConfiguration, ...]:
    return tuple(
        ActivityEstimatorConfiguration(
            overlap_weight=overlap_weight,
            neighborhood_radius_seconds=neighborhood_radius_seconds,
            minimum_improvement_over_zero=minimum_improvement,
            minimum_competing_margin=minimum_margin,
        )
        for overlap_weight in (1.0, 1.5)
        for neighborhood_radius_seconds in (0.05, 0.15, 0.3)
        for minimum_improvement in (0.001, 0.002, 0.005)
        for minimum_margin in (0.0001, 0.0005)
    )


def evaluate_activity_offset_estimator(
    examples: tuple[ReviewedActivityCurve, ...],
    configurations: tuple[ActivityEstimatorConfiguration, ...] | None = None,
    fold_count: int = DEFAULT_FOLD_COUNT,
    split_seed: str = DEFAULT_SPLIT_SEED,
) -> ActivityOffsetEvaluationReport:
    if not examples:
        raise ValueError("At least one reviewed activity curve is required")
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(examples) < fold_count:
        raise ValueError("fold_count cannot exceed the reviewed example count")
    evaluated_configurations = configurations or activity_configuration_grid()
    if not evaluated_configurations:
        raise ValueError("At least one estimator configuration is required")
    fold_by_external_id = stratified_label_fold_assignments(
        labels=tuple(example.label for example in examples),
        fold_count=fold_count,
        split_seed=split_seed,
    )

    folds: list[ActivityFoldEvaluation] = []
    samples: list[ActivitySampleEvaluation] = []
    for fold_index in range(fold_count):
        training_examples = tuple(
            example
            for example in examples
            if fold_by_external_id[example.label.external_id] != fold_index
        )
        validation_examples = tuple(
            example
            for example in examples
            if fold_by_external_id[example.label.external_id] == fold_index
        )
        selected_configuration = _select_configuration(
            examples=training_examples,
            configurations=evaluated_configurations,
        )
        folds.append(
            ActivityFoldEvaluation(
                fold_index=fold_index,
                training_label_count=len(training_examples),
                validation_label_count=len(validation_examples),
                selected_configuration=selected_configuration,
                training_metrics=_metrics(
                    examples=training_examples,
                    configuration=selected_configuration,
                ),
                validation_metrics=_metrics(
                    examples=validation_examples,
                    configuration=selected_configuration,
                ),
            )
        )
        samples.extend(
            _sample_evaluation(
                example=example,
                fold_index=fold_index,
                configuration=selected_configuration,
            )
            for example in validation_examples
        )

    ordered_samples = tuple(sorted(samples, key=lambda sample: sample.external_id))
    selected_full_configuration = _select_configuration(
        examples=examples,
        configurations=evaluated_configurations,
    )
    return ActivityOffsetEvaluationReport(
        estimator_version=ESTIMATOR_VERSION,
        split_seed=split_seed,
        fold_count=fold_count,
        configuration_count=len(evaluated_configurations),
        production_configuration=PRODUCTION_CONFIGURATION,
        production_all_reviewed_metrics=_metrics(
            examples=examples,
            configuration=PRODUCTION_CONFIGURATION,
        ),
        selected_full_training_configuration=selected_full_configuration,
        selected_full_training_metrics=_metrics(
            examples=examples,
            configuration=selected_full_configuration,
        ),
        tuned_out_of_fold_metrics=offset_error_metrics(
            errors=tuple(sample.absolute_error_seconds for sample in ordered_samples)
        ),
        folds=tuple(folds),
        samples=ordered_samples,
    )


def _select_configuration(
    examples: tuple[ReviewedActivityCurve, ...],
    configurations: tuple[ActivityEstimatorConfiguration, ...],
) -> ActivityEstimatorConfiguration:
    return min(
        configurations,
        key=lambda configuration: _selection_key(
            metrics=_metrics(examples=examples, configuration=configuration),
            configuration=configuration,
        ),
    )


def _metrics(
    examples: tuple[ReviewedActivityCurve, ...],
    configuration: ActivityEstimatorConfiguration,
) -> OffsetErrorMetrics:
    return offset_error_metrics(
        errors=tuple(
            abs(
                _prediction(curve=example.curve, configuration=configuration)
                - example.label.speaker2_shift_seconds
            )
            for example in examples
        )
    )


def _sample_evaluation(
    example: ReviewedActivityCurve,
    fold_index: int,
    configuration: ActivityEstimatorConfiguration,
) -> ActivitySampleEvaluation:
    predicted_shift_seconds = _prediction(
        curve=example.curve,
        configuration=configuration,
    )
    return ActivitySampleEvaluation(
        external_id=example.label.external_id,
        fold_index=fold_index,
        reviewed_shift_seconds=example.label.speaker2_shift_seconds,
        predicted_shift_seconds=predicted_shift_seconds,
        absolute_error_seconds=abs(predicted_shift_seconds - example.label.speaker2_shift_seconds),
    )


def _prediction(
    curve: ActivityLagCurve,
    configuration: ActivityEstimatorConfiguration,
) -> float:
    estimate = stable_offset_estimate(
        curve=curve,
        overlap_weight=configuration.overlap_weight,
        neighborhood_radius_seconds=configuration.neighborhood_radius_seconds,
        competing_exclusion_seconds=COMPETING_EXCLUSION_SECONDS,
        basin_loss_tolerance=BASIN_LOSS_TOLERANCE,
    )
    if (
        estimate.improvement_over_zero < configuration.minimum_improvement_over_zero
        or estimate.competing_margin < configuration.minimum_competing_margin
    ):
        return 0.0
    return estimate.shift_seconds


def _selection_key(
    metrics: OffsetErrorMetrics,
    configuration: ActivityEstimatorConfiguration,
) -> tuple[float, float, float, str]:
    return (
        metrics.mean_absolute_error_seconds,
        metrics.root_mean_squared_error_seconds,
        -metrics.within_0_1_seconds,
        configuration.model_dump_json(),
    )
