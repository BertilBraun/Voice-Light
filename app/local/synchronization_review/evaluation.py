from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from app.local.synchronization_review.calibration import REVIEWED_ALIGNMENTS, ReviewedAlignment
from app.local.synchronization_review.models import SynchronizationEvidenceSource
from app.shared.base_model import FrozenBaseModel

DEFAULT_FOLD_COUNT = 5
DEFAULT_SPLIT_SEED = "voice-light-synchronization-v1"
OFFSET_RESOLUTION_SECONDS = 0.1
LEGACY_OFFSET_RESOLUTION_SECONDS = 0.2
LEGACY_VARIABLE_WINDOW_SPREAD_SECONDS = 1.5


class ReviewProvenance(StrEnum):
    DATABASE = "database"
    STATIC_CALIBRATION = "static_calibration"


class WeakEvidenceFallback(StrEnum):
    ALL_SOURCES = "all_sources"
    ZERO_SHIFT = "zero_shift"


class EvidenceScope(StrEnum):
    INITIAL_180_SECONDS = "initial_180_seconds"
    FULL_RECORDING = "full_recording"


@dataclass(frozen=True)
class OffsetLabel:
    external_id: str
    speaker2_shift_seconds: float
    provenance: ReviewProvenance


@dataclass(frozen=True)
class OffsetEvidence:
    source: SynchronizationEvidenceSource
    estimated_shift_seconds: float
    bad_state_improvement: float
    overlap_reduction: float
    silence_reduction: float

    @property
    def joint_reduction(self) -> float:
        return max(0.0, min(self.overlap_reduction, self.silence_reduction))


@dataclass(frozen=True)
class OffsetWindowEvidence:
    source: SynchronizationEvidenceSource
    start_seconds: float
    end_seconds: float
    estimated_shift_seconds: float
    bad_state_improvement: float
    overlap_reduction: float
    silence_reduction: float

    @property
    def joint_reduction(self) -> float:
        return max(0.0, min(self.overlap_reduction, self.silence_reduction))


@dataclass(frozen=True)
class OffsetEvidenceRecord:
    external_id: str
    scope: EvidenceScope
    sources: tuple[OffsetEvidence, ...]
    windows: tuple[OffsetWindowEvidence, ...]


@dataclass(frozen=True)
class ReviewedOffsetExample:
    label: OffsetLabel
    evidence: OffsetEvidenceRecord


class OffsetEstimatorConfiguration(FrozenBaseModel):
    minimum_meaningful_lag_seconds: float = Field(ge=0.0)
    minimum_meaningful_improvement: float = Field(ge=0.0)
    minimum_joint_reduction: float = Field(ge=0.0)
    minimum_meaningful_source_count: int = Field(ge=1, le=3)
    weak_evidence_fallback: WeakEvidenceFallback


class OffsetErrorMetrics(FrozenBaseModel):
    label_count: int
    mean_absolute_error_seconds: float
    median_absolute_error_seconds: float
    root_mean_squared_error_seconds: float
    maximum_absolute_error_seconds: float
    within_0_05_seconds: float
    within_0_1_seconds: float
    within_0_2_seconds: float
    within_0_5_seconds: float


class OffsetSampleEvaluation(FrozenBaseModel):
    external_id: str
    provenance: ReviewProvenance
    fold_index: int
    reviewed_shift_seconds: float
    baseline_prediction_seconds: float
    baseline_absolute_error_seconds: float
    tuned_prediction_seconds: float
    tuned_absolute_error_seconds: float


class OffsetProductionPrediction(FrozenBaseModel):
    external_id: str
    provenance: ReviewProvenance
    reviewed_shift_seconds: float
    predicted_shift_seconds: float
    absolute_error_seconds: float


class OffsetDriftSummary(FrozenBaseModel):
    external_id: str
    source: SynchronizationEvidenceSource
    window_count: int
    first_window_start_seconds: float
    last_window_end_seconds: float
    first_shift_seconds: float
    last_shift_seconds: float
    end_to_start_change_seconds: float
    maximum_window_spread_seconds: float


class OffsetFoldEvaluation(FrozenBaseModel):
    fold_index: int
    training_label_count: int
    validation_label_count: int
    selected_configuration: OffsetEstimatorConfiguration
    training_metrics: OffsetErrorMetrics
    validation_metrics: OffsetErrorMetrics


class OffsetCohortEvaluation(FrozenBaseModel):
    cohort_name: str
    baseline_metrics: OffsetErrorMetrics
    tuned_out_of_fold_metrics: OffsetErrorMetrics


class OffsetEvaluationReport(FrozenBaseModel):
    evidence_scope: EvidenceScope
    split_seed: str
    fold_count: int
    configuration_count: int
    baseline_configuration: OffsetEstimatorConfiguration
    selected_full_training_configuration: OffsetEstimatorConfiguration
    selected_full_training_metrics: OffsetErrorMetrics
    combined_reviewed: OffsetCohortEvaluation
    database_reviews: OffsetCohortEvaluation
    folds: tuple[OffsetFoldEvaluation, ...]
    samples: tuple[OffsetSampleEvaluation, ...]
    production_selected_predictions: tuple[OffsetProductionPrediction, ...]
    drift_summaries: tuple[OffsetDriftSummary, ...]


BASELINE_CONFIGURATION = OffsetEstimatorConfiguration(
    minimum_meaningful_lag_seconds=0.8,
    minimum_meaningful_improvement=0.03,
    minimum_joint_reduction=0.008,
    minimum_meaningful_source_count=1,
    weak_evidence_fallback=WeakEvidenceFallback.ALL_SOURCES,
)


def reviewed_offset_labels(
    stored_alignments: tuple[ReviewedAlignment, ...],
) -> tuple[OffsetLabel, ...]:
    labels_by_external_id = {
        alignment.external_id: OffsetLabel(
            external_id=alignment.external_id,
            speaker2_shift_seconds=alignment.speaker2_shift_seconds,
            provenance=ReviewProvenance.STATIC_CALIBRATION,
        )
        for alignment in REVIEWED_ALIGNMENTS
    }
    labels_by_external_id.update(
        {
            alignment.external_id: OffsetLabel(
                external_id=alignment.external_id,
                speaker2_shift_seconds=alignment.speaker2_shift_seconds,
                provenance=ReviewProvenance.DATABASE,
            )
            for alignment in stored_alignments
        }
    )
    return tuple(sorted(labels_by_external_id.values(), key=lambda label: label.external_id))


def reviewed_examples(
    labels: tuple[OffsetLabel, ...],
    evidence_records: tuple[OffsetEvidenceRecord, ...],
) -> tuple[ReviewedOffsetExample, ...]:
    evidence_by_external_id = {record.external_id: record for record in evidence_records}
    missing_evidence = tuple(
        label.external_id for label in labels if label.external_id not in evidence_by_external_id
    )
    if missing_evidence:
        raise ValueError(
            f"Missing offset evidence for reviewed samples: {', '.join(missing_evidence)}"
        )
    return tuple(
        ReviewedOffsetExample(label=label, evidence=evidence_by_external_id[label.external_id])
        for label in labels
    )


def predict_offset(
    evidence: OffsetEvidenceRecord,
    configuration: OffsetEstimatorConfiguration,
) -> float:
    static_sources = (
        tuple(
            source
            for source in evidence.sources
            if source.source is not SynchronizationEvidenceSource.CONVERSATION_ANNOTATION
        )
        if evidence.scope is EvidenceScope.FULL_RECORDING
        else evidence.sources
    )
    meaningful_sources = tuple(
        source
        for source in static_sources
        if _meaningful_source(source=source, configuration=configuration)
    )
    if len(meaningful_sources) < configuration.minimum_meaningful_source_count:
        if configuration.weak_evidence_fallback is WeakEvidenceFallback.ZERO_SHIFT:
            return 0.0
        estimation_sources = static_sources
    else:
        estimation_sources = meaningful_sources
    if not estimation_sources:
        return 0.0

    full_estimate = _weighted_median_shift(estimation_sources)
    if evidence.scope is EvidenceScope.FULL_RECORDING:
        selected_windows = _strongest_window_series(windows=evidence.windows)
        meaningful_windows = tuple(
            window
            for window in selected_windows
            if window.bad_state_improvement >= configuration.minimum_meaningful_improvement
            and window.joint_reduction >= configuration.minimum_joint_reduction
        )
        if (
            len(meaningful_windows) >= 2
            and _spread(tuple(window.estimated_shift_seconds for window in meaningful_windows))
            >= LEGACY_VARIABLE_WINDOW_SPREAD_SECONDS
        ):
            return _quantized_shift(selected_windows[0].estimated_shift_seconds)
    return _quantized_shift(full_estimate)


def predict_legacy_offset(
    evidence: OffsetEvidenceRecord,
    configuration: OffsetEstimatorConfiguration,
) -> float:
    meaningful_sources = tuple(
        source
        for source in evidence.sources
        if _meaningful_source(source=source, configuration=configuration)
    )
    if len(meaningful_sources) < configuration.minimum_meaningful_source_count:
        if configuration.weak_evidence_fallback is WeakEvidenceFallback.ZERO_SHIFT:
            return 0.0
        estimation_sources = evidence.sources
    else:
        estimation_sources = meaningful_sources
    if not estimation_sources:
        return 0.0
    selected_windows = _strongest_window_series(windows=evidence.windows)
    meaningful_windows = tuple(
        window
        for window in selected_windows
        if window.bad_state_improvement >= configuration.minimum_meaningful_improvement
        and window.joint_reduction >= configuration.minimum_joint_reduction
    )
    variable = (
        len(meaningful_windows) >= 2
        and _spread(tuple(window.estimated_shift_seconds for window in meaningful_windows))
        >= LEGACY_VARIABLE_WINDOW_SPREAD_SECONDS
    )
    full_estimate = _weighted_median_shift(estimation_sources)
    raw_estimate = (
        selected_windows[0].estimated_shift_seconds
        if variable and selected_windows
        else full_estimate
    )
    quantized_steps = round(raw_estimate / LEGACY_OFFSET_RESOLUTION_SECONDS)
    return round(quantized_steps * LEGACY_OFFSET_RESOLUTION_SECONDS, 1)


def estimator_configuration_grid() -> tuple[OffsetEstimatorConfiguration, ...]:
    return tuple(
        OffsetEstimatorConfiguration(
            minimum_meaningful_lag_seconds=minimum_lag,
            minimum_meaningful_improvement=minimum_improvement,
            minimum_joint_reduction=minimum_joint_reduction,
            minimum_meaningful_source_count=minimum_source_count,
            weak_evidence_fallback=weak_evidence_fallback,
        )
        for minimum_lag in (0.4, 0.8, 1.2)
        for minimum_improvement in (0.02, 0.03, 0.05)
        for minimum_joint_reduction in (0.0, 0.008, 0.015)
        for minimum_source_count in (1, 2)
        for weak_evidence_fallback in WeakEvidenceFallback
    )


def evaluate_offset_estimator(
    examples: tuple[ReviewedOffsetExample, ...],
    configurations: tuple[OffsetEstimatorConfiguration, ...] | None = None,
    fold_count: int = DEFAULT_FOLD_COUNT,
    split_seed: str = DEFAULT_SPLIT_SEED,
) -> OffsetEvaluationReport:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(examples) < fold_count:
        raise ValueError("fold_count cannot exceed the reviewed example count")
    if not examples:
        raise ValueError("At least one reviewed example is required")
    scopes = {example.evidence.scope for example in examples}
    if len(scopes) != 1:
        raise ValueError("All evaluated evidence must use the same scope")
    evaluated_configurations = configurations or estimator_configuration_grid()
    if not evaluated_configurations:
        raise ValueError("At least one estimator configuration is required")

    fold_by_external_id = _stratified_fold_assignments(
        examples=examples,
        fold_count=fold_count,
        split_seed=split_seed,
    )
    sample_evaluations: list[OffsetSampleEvaluation] = []
    fold_evaluations: list[OffsetFoldEvaluation] = []
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
        selected_configuration = select_configuration(
            examples=training_examples,
            configurations=evaluated_configurations,
        )
        training_metrics = metrics_for_configuration(
            examples=training_examples,
            configuration=selected_configuration,
        )
        validation_metrics = metrics_for_configuration(
            examples=validation_examples,
            configuration=selected_configuration,
        )
        fold_evaluations.append(
            OffsetFoldEvaluation(
                fold_index=fold_index,
                training_label_count=len(training_examples),
                validation_label_count=len(validation_examples),
                selected_configuration=selected_configuration,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
            )
        )
        sample_evaluations.extend(
            _sample_evaluation(
                example=example,
                fold_index=fold_index,
                tuned_configuration=selected_configuration,
            )
            for example in validation_examples
        )

    ordered_samples = tuple(sorted(sample_evaluations, key=lambda sample: sample.external_id))
    full_training_configuration = select_configuration(
        examples=examples,
        configurations=evaluated_configurations,
    )
    database_samples = tuple(
        sample for sample in ordered_samples if sample.provenance is ReviewProvenance.DATABASE
    )
    return OffsetEvaluationReport(
        evidence_scope=next(iter(scopes)),
        split_seed=split_seed,
        fold_count=fold_count,
        configuration_count=len(evaluated_configurations),
        baseline_configuration=BASELINE_CONFIGURATION,
        selected_full_training_configuration=full_training_configuration,
        selected_full_training_metrics=metrics_for_configuration(
            examples=examples,
            configuration=full_training_configuration,
        ),
        combined_reviewed=_cohort_evaluation(
            cohort_name="combined_reviewed",
            samples=ordered_samples,
        ),
        database_reviews=_cohort_evaluation(
            cohort_name="database_reviews",
            samples=database_samples,
        ),
        folds=tuple(fold_evaluations),
        samples=ordered_samples,
        production_selected_predictions=tuple(
            _production_prediction(
                example=example,
                configuration=full_training_configuration,
            )
            for example in sorted(examples, key=lambda item: item.label.external_id)
        ),
        drift_summaries=tuple(
            summary
            for example in sorted(examples, key=lambda item: item.label.external_id)
            for summary in _drift_summaries(evidence=example.evidence)
        ),
    )


def select_configuration(
    examples: tuple[ReviewedOffsetExample, ...],
    configurations: tuple[OffsetEstimatorConfiguration, ...],
) -> OffsetEstimatorConfiguration:
    return min(
        configurations,
        key=lambda configuration: _selection_key(
            metrics=metrics_for_configuration(examples=examples, configuration=configuration),
            configuration=configuration,
        ),
    )


def metrics_for_configuration(
    examples: tuple[ReviewedOffsetExample, ...],
    configuration: OffsetEstimatorConfiguration,
) -> OffsetErrorMetrics:
    errors = tuple(
        abs(
            predict_offset(evidence=example.evidence, configuration=configuration)
            - example.label.speaker2_shift_seconds
        )
        for example in examples
    )
    return offset_error_metrics(errors=errors)


def offset_error_metrics(errors: tuple[float, ...]) -> OffsetErrorMetrics:
    if not errors:
        raise ValueError("At least one error is required")
    return OffsetErrorMetrics(
        label_count=len(errors),
        mean_absolute_error_seconds=statistics.fmean(errors),
        median_absolute_error_seconds=statistics.median(errors),
        root_mean_squared_error_seconds=math.sqrt(statistics.fmean(error**2 for error in errors)),
        maximum_absolute_error_seconds=max(errors),
        within_0_05_seconds=sum(error <= 0.05 for error in errors) / len(errors),
        within_0_1_seconds=sum(error <= 0.1 for error in errors) / len(errors),
        within_0_2_seconds=sum(error <= 0.2 for error in errors) / len(errors),
        within_0_5_seconds=sum(error <= 0.5 for error in errors) / len(errors),
    )


def _sample_evaluation(
    example: ReviewedOffsetExample,
    fold_index: int,
    tuned_configuration: OffsetEstimatorConfiguration,
) -> OffsetSampleEvaluation:
    baseline_prediction = (
        predict_legacy_offset(
            evidence=example.evidence,
            configuration=BASELINE_CONFIGURATION,
        )
        if example.evidence.scope is EvidenceScope.INITIAL_180_SECONDS
        else predict_offset(
            evidence=example.evidence,
            configuration=BASELINE_CONFIGURATION,
        )
    )
    tuned_prediction = predict_offset(
        evidence=example.evidence,
        configuration=tuned_configuration,
    )
    return OffsetSampleEvaluation(
        external_id=example.label.external_id,
        provenance=example.label.provenance,
        fold_index=fold_index,
        reviewed_shift_seconds=example.label.speaker2_shift_seconds,
        baseline_prediction_seconds=baseline_prediction,
        baseline_absolute_error_seconds=abs(
            baseline_prediction - example.label.speaker2_shift_seconds
        ),
        tuned_prediction_seconds=tuned_prediction,
        tuned_absolute_error_seconds=abs(tuned_prediction - example.label.speaker2_shift_seconds),
    )


def _production_prediction(
    example: ReviewedOffsetExample,
    configuration: OffsetEstimatorConfiguration,
) -> OffsetProductionPrediction:
    predicted_shift = predict_offset(evidence=example.evidence, configuration=configuration)
    return OffsetProductionPrediction(
        external_id=example.label.external_id,
        provenance=example.label.provenance,
        reviewed_shift_seconds=example.label.speaker2_shift_seconds,
        predicted_shift_seconds=predicted_shift,
        absolute_error_seconds=abs(predicted_shift - example.label.speaker2_shift_seconds),
    )


def _drift_summaries(evidence: OffsetEvidenceRecord) -> tuple[OffsetDriftSummary, ...]:
    windows_by_source: dict[SynchronizationEvidenceSource, list[OffsetWindowEvidence]] = {}
    for window in evidence.windows:
        windows_by_source.setdefault(window.source, []).append(window)
    summaries: list[OffsetDriftSummary] = []
    for source, source_windows in sorted(
        windows_by_source.items(),
        key=lambda item: item[0].value,
    ):
        ordered_windows = sorted(source_windows, key=lambda window: window.start_seconds)
        first_window = ordered_windows[0]
        last_window = ordered_windows[-1]
        shifts = tuple(window.estimated_shift_seconds for window in ordered_windows)
        summaries.append(
            OffsetDriftSummary(
                external_id=evidence.external_id,
                source=source,
                window_count=len(ordered_windows),
                first_window_start_seconds=first_window.start_seconds,
                last_window_end_seconds=last_window.end_seconds,
                first_shift_seconds=first_window.estimated_shift_seconds,
                last_shift_seconds=last_window.estimated_shift_seconds,
                end_to_start_change_seconds=(
                    last_window.estimated_shift_seconds - first_window.estimated_shift_seconds
                ),
                maximum_window_spread_seconds=_spread(shifts),
            )
        )
    return tuple(summaries)


def _cohort_evaluation(
    cohort_name: str,
    samples: tuple[OffsetSampleEvaluation, ...],
) -> OffsetCohortEvaluation:
    return OffsetCohortEvaluation(
        cohort_name=cohort_name,
        baseline_metrics=offset_error_metrics(
            errors=tuple(sample.baseline_absolute_error_seconds for sample in samples)
        ),
        tuned_out_of_fold_metrics=offset_error_metrics(
            errors=tuple(sample.tuned_absolute_error_seconds for sample in samples)
        ),
    )


def _meaningful_source(
    source: OffsetEvidence,
    configuration: OffsetEstimatorConfiguration,
) -> bool:
    return (
        abs(source.estimated_shift_seconds) >= configuration.minimum_meaningful_lag_seconds
        and source.bad_state_improvement >= configuration.minimum_meaningful_improvement
        and source.joint_reduction >= configuration.minimum_joint_reduction
    )


def _weighted_median_shift(sources: tuple[OffsetEvidence, ...]) -> float:
    weighted_shifts = tuple(
        sorted(
            (
                source.estimated_shift_seconds,
                max(0.001, source.bad_state_improvement + 2.0 * source.joint_reduction),
            )
            for source in sources
        )
    )
    midpoint = sum(weight for _, weight in weighted_shifts) / 2.0
    accumulated_weight = 0.0
    for shift_seconds, weight in weighted_shifts:
        accumulated_weight += weight
        if accumulated_weight >= midpoint:
            return shift_seconds
    return weighted_shifts[-1][0]


def _strongest_window_series(
    windows: tuple[OffsetWindowEvidence, ...],
) -> tuple[OffsetWindowEvidence, ...]:
    windows_by_source: dict[SynchronizationEvidenceSource, list[OffsetWindowEvidence]] = {}
    for window in windows:
        windows_by_source.setdefault(window.source, []).append(window)
    if not windows_by_source:
        return ()
    selected_source = max(
        windows_by_source,
        key=lambda source: (
            sum(
                window.bad_state_improvement + 2.0 * window.joint_reduction
                for window in windows_by_source[source]
            ),
            source.value,
        ),
    )
    return tuple(
        sorted(windows_by_source[selected_source], key=lambda window: window.start_seconds)
    )


def _quantized_shift(shift_seconds: float) -> float:
    quantized_steps = round(shift_seconds / OFFSET_RESOLUTION_SECONDS)
    return round(quantized_steps * OFFSET_RESOLUTION_SECONDS, 1)


def _selection_key(
    metrics: OffsetErrorMetrics,
    configuration: OffsetEstimatorConfiguration,
) -> tuple[float, float, float, float, str]:
    return (
        metrics.root_mean_squared_error_seconds,
        metrics.mean_absolute_error_seconds,
        metrics.maximum_absolute_error_seconds,
        -metrics.within_0_1_seconds,
        configuration.model_dump_json(),
    )


def _stratified_fold_assignments(
    examples: tuple[ReviewedOffsetExample, ...],
    fold_count: int,
    split_seed: str,
) -> dict[str, int]:
    examples_by_stratum: dict[tuple[ReviewProvenance, str], list[ReviewedOffsetExample]] = {}
    for example in examples:
        stratum = (
            example.label.provenance,
            _label_magnitude_stratum(example.label.speaker2_shift_seconds),
        )
        examples_by_stratum.setdefault(stratum, []).append(example)

    assignments: dict[str, int] = {}
    fold_sizes = [0] * fold_count
    for stratum, stratum_examples in sorted(
        examples_by_stratum.items(),
        key=lambda item: (item[0][0].value, item[0][1]),
    ):
        stratum_fold_sizes = [0] * fold_count
        ordered_examples = sorted(
            stratum_examples,
            key=lambda example: hashlib.sha256(
                f"{split_seed}:{example.label.external_id}".encode()
            ).digest(),
        )
        starting_fold = (
            int.from_bytes(
                hashlib.sha256(f"{split_seed}:{stratum[0].value}:{stratum[1]}".encode()).digest()[
                    :2
                ],
                byteorder="big",
            )
            % fold_count
        )
        for example in ordered_examples:
            fold_index = min(
                range(fold_count),
                key=lambda candidate_fold: (
                    stratum_fold_sizes[candidate_fold],
                    fold_sizes[candidate_fold],
                    (candidate_fold - starting_fold) % fold_count,
                ),
            )
            assignments[example.label.external_id] = fold_index
            stratum_fold_sizes[fold_index] += 1
            fold_sizes[fold_index] += 1
    return assignments


def _label_magnitude_stratum(shift_seconds: float) -> str:
    if abs(shift_seconds) <= 0.25:
        return "zero"
    if shift_seconds > 0.0:
        return "positive"
    if abs(shift_seconds) <= 2.0:
        return "negative_small"
    return "negative_large"


def _spread(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)
