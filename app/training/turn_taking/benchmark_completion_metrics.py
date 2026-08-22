from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from pydantic import Field

from app.training.turn_taking.benchmark_metrics import (
    CalibrationBin,
    CalibrationMetrics,
    PolicyConfiguration,
    ScorePersistence,
)
from app.training.turn_taking.benchmark_models import (
    BenchmarkModel,
    CompletionCandidatePrediction,
    CompletionDetectorProvenance,
    TurnCompletionCandidate,
)

PROBABILITY_EPSILON = 1e-7
TIMESTAMP_TOLERANCE_SECONDS = 1e-6


class CompletionEvaluationConfiguration(BenchmarkModel):
    hold_completion_maximum: float = Field(ge=0.0, le=1.0)
    hold_continuation_minimum: float = Field(ge=0.0, le=1.0)
    eot_completion_minimum: float = Field(ge=0.0, le=1.0)
    conflicting_continuation_minimum: float = Field(ge=0.0, le=1.0)
    score_persistence: ScorePersistence


class CompletionPolicyMetrics(BenchmarkModel):
    inventory_support: int = Field(ge=0)
    evaluated_support: int = Field(ge=0)
    ignored_support: int = Field(ge=0)
    conflicting_support: int = Field(ge=0)
    hold_support: int = Field(ge=0)
    eot_support: int = Field(ge=0)
    false_cutoff_count: int = Field(ge=0)
    false_cutoff_rate: float = Field(ge=0.0, le=1.0)
    detector_eot_count: int = Field(ge=0)
    eot_recall: float = Field(ge=0.0, le=1.0)
    mean_latency_seconds: float | None = Field(default=None, ge=0.0)
    latency_p50_seconds: float | None = Field(default=None, ge=0.0)
    latency_p90_seconds: float | None = Field(default=None, ge=0.0)
    latency_p95_seconds: float | None = Field(default=None, ge=0.0)
    latency_p99_seconds: float | None = Field(default=None, ge=0.0)


class CompletionPolicySweepPoint(BenchmarkModel):
    policy: PolicyConfiguration
    metrics: CompletionPolicyMetrics


class CompletionPredictionCoverage(BenchmarkModel):
    inventory_candidate_support: int = Field(ge=0)
    scored_candidate_support: int = Field(ge=0)
    missing_candidate_support: int = Field(ge=0)
    prediction_support: int = Field(ge=0)


class CompletionDiscriminationMetrics(BenchmarkModel):
    native_gate_seconds: tuple[float, ...]
    clean_support: int = Field(ge=0)
    scored_clean_support: int = Field(ge=0)
    missing_clean_support: int = Field(ge=0)
    hold_support: int = Field(ge=0)
    eot_support: int = Field(ge=0)
    auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    eot_average_precision: float | None = Field(default=None, ge=0.0, le=1.0)


class CompletionBreakdownMetrics(BenchmarkModel):
    dimension: str
    value: str
    metrics: CompletionPolicyMetrics


class CompletionAnalysisReport(BenchmarkModel):
    schema_version: Literal["voice-light-turn-completion-analysis-v2"] = (
        "voice-light-turn-completion-analysis-v2"
    )
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    causal_candidate_coverage_seconds: Literal[2.0] = 2.0
    assistant_active_anchors_excluded: Literal[True] = True
    insufficient_horizon_candidates_censored: Literal[True] = True
    detector: CompletionDetectorProvenance
    evaluation: CompletionEvaluationConfiguration
    prediction_coverage: CompletionPredictionCoverage
    discrimination: CompletionDiscriminationMetrics
    calibration: CalibrationMetrics | None
    sweep: tuple[CompletionPolicySweepPoint, ...]
    pareto_frontier: tuple[CompletionPolicySweepPoint, ...]
    selected_validation_point: CompletionPolicySweepPoint
    selected_breakdowns: tuple[CompletionBreakdownMetrics, ...]


def sweep_completion_policies(
    candidates: Iterable[TurnCompletionCandidate],
    predictions: Iterable[CompletionCandidatePrediction],
    thresholds: Iterable[float],
    action_delays_seconds: Iterable[float],
    timeouts_seconds: Iterable[float],
    evaluation: CompletionEvaluationConfiguration,
) -> tuple[CompletionPolicySweepPoint, ...]:
    candidate_rows = tuple(candidates)
    prediction_rows = tuple(predictions)
    validate_completion_predictions(candidate_rows, prediction_rows)
    grouped = _predictions_by_candidate(prediction_rows)
    policies = tuple(
        PolicyConfiguration(
            threshold=threshold,
            action_delay_seconds=action_delay,
            timeout_seconds=timeout,
        )
        for threshold, action_delay, timeout in itertools.product(
            sorted(set(thresholds)),
            sorted(set(action_delays_seconds)),
            sorted(set(timeouts_seconds)),
        )
        if action_delay <= timeout
    )
    if not policies:
        raise ValueError("Policy sweep requires at least one valid policy combination.")
    return tuple(
        CompletionPolicySweepPoint(
            policy=policy,
            metrics=_evaluate_policy(candidate_rows, grouped, policy, evaluation),
        )
        for policy in policies
    )


def completion_pareto_frontier(
    points: Iterable[CompletionPolicySweepPoint],
) -> tuple[CompletionPolicySweepPoint, ...]:
    eligible = tuple(point for point in points if point.metrics.mean_latency_seconds is not None)
    frontier = tuple(
        point
        for point in eligible
        if not any(_dominates(other, point) for other in eligible if other is not point)
    )
    return tuple(
        sorted(
            frontier,
            key=lambda point: (
                point.metrics.false_cutoff_rate,
                _required_latency(point),
                -point.metrics.eot_recall,
            ),
        )
    )


def select_completion_validation_point(
    points: Iterable[CompletionPolicySweepPoint],
    cutoff_budget: float = 0.05,
) -> CompletionPolicySweepPoint:
    point_rows = tuple(points)
    eligible = tuple(
        point
        for point in point_rows
        if point.metrics.mean_latency_seconds is not None
        and point.metrics.false_cutoff_rate <= cutoff_budget
    )
    if eligible:
        return min(
            eligible,
            key=lambda point: (
                -point.metrics.eot_recall,
                _required_latency(point),
                point.metrics.false_cutoff_rate,
            ),
        )
    frontier = completion_pareto_frontier(point_rows)
    if not frontier:
        raise ValueError("Policy sweep did not produce an eligible operating point.")
    return min(
        frontier,
        key=lambda point: (
            point.metrics.false_cutoff_rate,
            _required_latency(point),
        ),
    )


def validate_completion_predictions(
    candidates: Iterable[TurnCompletionCandidate],
    predictions: Iterable[CompletionCandidatePrediction],
) -> CompletionPredictionCoverage:
    candidate_rows = tuple(candidates)
    prediction_rows = tuple(predictions)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_rows}
    if len(candidate_by_id) != len(candidate_rows):
        raise ValueError("Completion inventory contains duplicate candidate IDs.")
    observed_keys: set[tuple[str, float]] = set()
    scored_candidate_ids: set[str] = set()
    for prediction in prediction_rows:
        candidate = candidate_by_id.get(prediction.candidate_id)
        if candidate is None:
            raise ValueError(f"Prediction references unknown candidate {prediction.candidate_id}.")
        key = (prediction.candidate_id, prediction.absolute_time_seconds)
        if key in observed_keys:
            raise ValueError("Completion predictions contain a duplicate candidate timestamp.")
        observed_keys.add(key)
        scored_candidate_ids.add(prediction.candidate_id)
        expected_time = candidate.anchor_seconds + prediction.elapsed_seconds
        if not math.isclose(
            prediction.absolute_time_seconds,
            expected_time,
            abs_tol=TIMESTAMP_TOLERANCE_SECONDS,
        ):
            raise ValueError("Completion prediction absolute and elapsed times are inconsistent.")
        if (
            prediction.absolute_time_seconds <= candidate.anchor_seconds
            or prediction.absolute_time_seconds
            > candidate.end_seconds + TIMESTAMP_TOLERANCE_SECONDS
        ):
            raise ValueError("Completion prediction falls outside its candidate opportunity.")
    return CompletionPredictionCoverage(
        inventory_candidate_support=len(candidate_rows),
        scored_candidate_support=len(scored_candidate_ids),
        missing_candidate_support=len(candidate_rows) - len(scored_candidate_ids),
        prediction_support=len(prediction_rows),
    )


def completion_discrimination_metrics(
    candidates: Iterable[TurnCompletionCandidate],
    predictions: Iterable[CompletionCandidatePrediction],
    evaluation: CompletionEvaluationConfiguration,
) -> CompletionDiscriminationMetrics:
    candidate_rows = tuple(candidates)
    prediction_rows = tuple(predictions)
    validate_completion_predictions(candidate_rows, prediction_rows)
    first_predictions = {
        candidate_id: values[0]
        for candidate_id, values in _predictions_by_candidate(prediction_rows).items()
    }
    clean_labels = {
        candidate.candidate_id: label
        for candidate in candidate_rows
        if (label := _clean_label(candidate, evaluation)) is not None
    }
    scored = tuple(
        (first_predictions[candidate_id].completion_probability, label)
        for candidate_id, label in clean_labels.items()
        if candidate_id in first_predictions
    )
    hold_support = sum(label == 0 for _, label in scored)
    eot_support = sum(label == 1 for _, label in scored)
    return CompletionDiscriminationMetrics(
        native_gate_seconds=tuple(
            sorted(
                {
                    prediction.elapsed_seconds
                    for candidate_id, prediction in first_predictions.items()
                    if candidate_id in clean_labels
                }
            )
        ),
        clean_support=len(clean_labels),
        scored_clean_support=len(scored),
        missing_clean_support=len(clean_labels) - len(scored),
        hold_support=hold_support,
        eot_support=eot_support,
        auroc=_auroc(scored),
        eot_average_precision=_average_precision(scored),
    )


def completion_breakdowns(
    candidates: Iterable[TurnCompletionCandidate],
    predictions: Iterable[CompletionCandidatePrediction],
    policy: PolicyConfiguration,
    evaluation: CompletionEvaluationConfiguration,
) -> tuple[CompletionBreakdownMetrics, ...]:
    candidate_rows = tuple(candidates)
    prediction_rows = tuple(predictions)
    grouped = _predictions_by_candidate(prediction_rows)
    slices: defaultdict[tuple[str, str], list[TurnCompletionCandidate]] = defaultdict(list)
    for candidate in candidate_rows:
        slices[("dataset", candidate.dataset_name)].append(candidate)
        for category in candidate.categories:
            slices[("category", category)].append(candidate)
    return tuple(
        CompletionBreakdownMetrics(
            dimension=dimension,
            value=value,
            metrics=_evaluate_policy(tuple(slice_candidates), grouped, policy, evaluation),
        )
        for (dimension, value), slice_candidates in sorted(slices.items())
    )


def completion_calibration_metrics(
    candidates: Iterable[TurnCompletionCandidate],
    predictions: Iterable[CompletionCandidatePrediction],
    bin_count: int = 10,
) -> CalibrationMetrics:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")
    candidate_rows = tuple(candidates)
    prediction_rows = tuple(predictions)
    validate_completion_predictions(candidate_rows, prediction_rows)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_rows}
    first_predictions = {
        candidate_id: values[0]
        for candidate_id, values in _predictions_by_candidate(prediction_rows).items()
    }
    pairs: list[tuple[float, float]] = []
    for candidate_id, prediction in first_predictions.items():
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Prediction references unknown candidate {candidate_id}.")
        pairs.append(
            (
                prediction.completion_probability,
                candidate.target_points[0].completion_probability,
            )
        )
    bins = _calibration_bins(pairs, bin_count)
    if not pairs:
        return CalibrationMetrics(
            support=0,
            binary_cross_entropy=None,
            brier_score=None,
            expected_calibration_error=None,
            bins=bins,
        )
    binary_cross_entropy = sum(_binary_cross_entropy(*pair) for pair in pairs) / len(pairs)
    brier_score = sum((prediction - target) ** 2 for prediction, target in pairs) / len(pairs)
    expected_calibration_error = sum(
        calibration_bin.support
        / len(pairs)
        * abs(
            _required_mean(calibration_bin.prediction_mean)
            - _required_mean(calibration_bin.target_mean)
        )
        for calibration_bin in bins
        if calibration_bin.support
    )
    return CalibrationMetrics(
        support=len(pairs),
        binary_cross_entropy=binary_cross_entropy,
        brier_score=brier_score,
        expected_calibration_error=expected_calibration_error,
        bins=bins,
    )


def _evaluate_policy(
    candidates: tuple[TurnCompletionCandidate, ...],
    predictions_by_candidate: dict[str, tuple[CompletionCandidatePrediction, ...]],
    policy: PolicyConfiguration,
    evaluation: CompletionEvaluationConfiguration,
) -> CompletionPolicyMetrics:
    hold_support = 0
    eot_support = 0
    ignored_support = 0
    conflicting_support = 0
    false_cutoff_count = 0
    detector_eot_count = 0
    latencies: list[float] = []
    for candidate in candidates:
        completion = candidate.target_points[0].completion_probability
        continuation = candidate.continuation_probability
        conflicting = (
            completion >= evaluation.eot_completion_minimum
            and continuation is not None
            and continuation >= evaluation.conflicting_continuation_minimum
        )
        clean_label = _clean_label(candidate, evaluation)
        if conflicting:
            conflicting_support += 1
        if clean_label is None:
            ignored_support += 1
            continue
        crossing = _first_crossing(
            predictions_by_candidate.get(candidate.candidate_id, ()),
            policy,
            evaluation.score_persistence,
        )
        opportunity_seconds = candidate.end_seconds - candidate.anchor_seconds
        action_seconds = min(policy.timeout_seconds, opportunity_seconds)
        detector_action = None
        if crossing is not None:
            detector_action = max(policy.action_delay_seconds, crossing.elapsed_seconds)
            action_seconds = min(action_seconds, detector_action)
        if clean_label == 0:
            hold_support += 1
            if action_seconds < opportunity_seconds - TIMESTAMP_TOLERANCE_SECONDS:
                false_cutoff_count += 1
        else:
            eot_support += 1
            if (
                detector_action is not None
                and detector_action <= policy.timeout_seconds
                and detector_action < opportunity_seconds - TIMESTAMP_TOLERANCE_SECONDS
            ):
                detector_eot_count += 1
            latencies.append(action_seconds)
    evaluated_support = hold_support + eot_support
    return CompletionPolicyMetrics(
        inventory_support=len(candidates),
        evaluated_support=evaluated_support,
        ignored_support=ignored_support,
        conflicting_support=conflicting_support,
        hold_support=hold_support,
        eot_support=eot_support,
        false_cutoff_count=false_cutoff_count,
        false_cutoff_rate=_safe_rate(false_cutoff_count, hold_support),
        detector_eot_count=detector_eot_count,
        eot_recall=_safe_rate(detector_eot_count, eot_support),
        mean_latency_seconds=sum(latencies) / len(latencies) if latencies else None,
        latency_p50_seconds=_percentile(latencies, 0.50),
        latency_p90_seconds=_percentile(latencies, 0.90),
        latency_p95_seconds=_percentile(latencies, 0.95),
        latency_p99_seconds=_percentile(latencies, 0.99),
    )


def _clean_label(
    candidate: TurnCompletionCandidate,
    evaluation: CompletionEvaluationConfiguration,
) -> int | None:
    completion = candidate.target_points[0].completion_probability
    continuation = candidate.continuation_probability
    conflicting = (
        completion >= evaluation.eot_completion_minimum
        and continuation is not None
        and continuation >= evaluation.conflicting_continuation_minimum
    )
    if (
        completion <= evaluation.hold_completion_maximum
        and continuation is not None
        and continuation >= evaluation.hold_continuation_minimum
    ):
        return 0
    if completion >= evaluation.eot_completion_minimum and not conflicting:
        return 1
    return None


def _predictions_by_candidate(
    predictions: Iterable[CompletionCandidatePrediction],
) -> dict[str, tuple[CompletionCandidatePrediction, ...]]:
    grouped: defaultdict[str, list[CompletionCandidatePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.candidate_id].append(prediction)
    return {
        candidate_id: tuple(sorted(values, key=lambda prediction: prediction.elapsed_seconds))
        for candidate_id, values in grouped.items()
    }


def _first_crossing(
    predictions: tuple[CompletionCandidatePrediction, ...],
    policy: PolicyConfiguration,
    persistence: ScorePersistence,
) -> CompletionCandidatePrediction | None:
    return next(
        (
            prediction
            for prediction in predictions
            if prediction.completion_probability >= policy.threshold
            and (
                persistence is ScorePersistence.LATCHED
                or prediction.elapsed_seconds + TIMESTAMP_TOLERANCE_SECONDS
                >= policy.action_delay_seconds
            )
        ),
        None,
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _binary_cross_entropy(prediction: float, target: float) -> float:
    probability = min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, prediction))
    return -(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability))


def _calibration_bins(
    pairs: list[tuple[float, float]], bin_count: int
) -> tuple[CalibrationBin, ...]:
    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for prediction, target in pairs:
        grouped[min(bin_count - 1, math.floor(prediction * bin_count))].append((prediction, target))
    return tuple(
        CalibrationBin(
            lower_bound=index / bin_count,
            upper_bound=(index + 1) / bin_count,
            support=len(values),
            prediction_mean=(
                sum(prediction for prediction, _ in values) / len(values) if values else None
            ),
            target_mean=(sum(target for _, target in values) / len(values) if values else None),
        )
        for index, values in enumerate(grouped)
    )


def _dominates(
    candidate: CompletionPolicySweepPoint,
    other: CompletionPolicySweepPoint,
) -> bool:
    candidate_latency = _required_latency(candidate)
    other_latency = _required_latency(other)
    no_worse = (
        candidate.metrics.false_cutoff_rate <= other.metrics.false_cutoff_rate
        and candidate_latency <= other_latency
        and candidate.metrics.eot_recall >= other.metrics.eot_recall
    )
    strictly_better = (
        candidate.metrics.false_cutoff_rate < other.metrics.false_cutoff_rate
        or candidate_latency < other_latency
        or candidate.metrics.eot_recall > other.metrics.eot_recall
    )
    return no_worse and strictly_better


def _required_latency(point: CompletionPolicySweepPoint) -> float:
    assert point.metrics.mean_latency_seconds is not None
    return point.metrics.mean_latency_seconds


def _required_mean(value: float | None) -> float:
    assert value is not None
    return value


def _auroc(scored_labels: tuple[tuple[float, int], ...]) -> float | None:
    positive_scores = tuple(score for score, label in scored_labels if label == 1)
    negative_scores = tuple(score for score, label in scored_labels if label == 0)
    if not positive_scores or not negative_scores:
        return None
    comparisons = tuple(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positive_scores
        for negative in negative_scores
    )
    return sum(comparisons) / len(comparisons)


def _average_precision(scored_labels: tuple[tuple[float, int], ...]) -> float | None:
    positive_support = sum(label == 1 for _, label in scored_labels)
    if positive_support == 0:
        return None
    grouped: defaultdict[float, list[int]] = defaultdict(list)
    for score, label in scored_labels:
        grouped[score].append(label)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(grouped, reverse=True):
        labels = grouped[score]
        true_positives += sum(labels)
        false_positives += len(labels) - sum(labels)
        recall = true_positives / positive_support
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area
