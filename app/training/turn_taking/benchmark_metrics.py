from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum

from pydantic import Field, model_validator

from app.training.turn_taking.benchmark_models import (
    BenchmarkModel,
    CandidatePrediction,
    CandidateTargetPoint,
    SilenceCandidate,
)

PROBABILITY_EPSILON = 1e-7
TIMESTAMP_TOLERANCE_SECONDS = 1e-6


class PolicyConfiguration(BenchmarkModel):
    threshold: float = Field(ge=0.0, le=1.0)
    action_delay_seconds: float = Field(ge=0.0)
    timeout_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_timing(self) -> PolicyConfiguration:
        if self.action_delay_seconds > self.timeout_seconds:
            raise ValueError("action_delay_seconds must not exceed timeout_seconds.")
        return self


class ScorePersistence(StrEnum):
    CURRENT = "current"
    LATCHED = "latched"


class EvaluationConfiguration(BenchmarkModel):
    target_score_point_seconds: float = Field(gt=0.0)
    target_yield_threshold: float = Field(ge=0.0, le=1.0)
    score_persistence: ScorePersistence


class PolicyMetrics(BenchmarkModel):
    candidate_support: int = Field(ge=0)
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


class PolicySweepPoint(BenchmarkModel):
    policy: PolicyConfiguration
    metrics: PolicyMetrics


class CalibrationBin(BenchmarkModel):
    lower_bound: float = Field(ge=0.0, le=1.0)
    upper_bound: float = Field(gt=0.0, le=1.0)
    support: int = Field(ge=0)
    prediction_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    target_mean: float | None = Field(default=None, ge=0.0, le=1.0)


class CalibrationMetrics(BenchmarkModel):
    support: int = Field(ge=0)
    binary_cross_entropy: float | None = Field(default=None, ge=0.0)
    brier_score: float | None = Field(default=None, ge=0.0)
    expected_calibration_error: float | None = Field(default=None, ge=0.0, le=1.0)
    bins: tuple[CalibrationBin, ...]


class BreakdownMetrics(BenchmarkModel):
    dimension: str
    value: str
    metrics: PolicyMetrics


def evaluate_policy(
    candidates: Iterable[SilenceCandidate],
    predictions: Iterable[CandidatePrediction],
    policy: PolicyConfiguration,
    evaluation: EvaluationConfiguration,
) -> PolicyMetrics:
    candidate_rows = tuple(candidates)
    predictions_by_candidate = _predictions_by_candidate(predictions)
    return _evaluate_policy_with_grouped_predictions(
        candidate_rows=candidate_rows,
        predictions_by_candidate=predictions_by_candidate,
        policy=policy,
        evaluation=evaluation,
    )


def _evaluate_policy_with_grouped_predictions(
    candidate_rows: tuple[SilenceCandidate, ...],
    predictions_by_candidate: dict[str, tuple[CandidatePrediction, ...]],
    policy: PolicyConfiguration,
    evaluation: EvaluationConfiguration,
) -> PolicyMetrics:
    hold_support = 0
    eot_support = 0
    false_cutoff_count = 0
    detector_eot_count = 0
    latencies: list[float] = []
    evaluated_count = 0
    for candidate in candidate_rows:
        target = _target_at_score_point(
            candidate=candidate,
            score_point_seconds=evaluation.target_score_point_seconds,
        )
        if target is None:
            continue
        evaluated_count += 1
        crossing = _first_threshold_crossing(
            predictions=predictions_by_candidate.get(candidate.candidate_id, ()),
            policy=policy,
            score_persistence=evaluation.score_persistence,
        )
        action_seconds = policy.timeout_seconds
        if crossing is not None:
            score_action_seconds = max(
                policy.action_delay_seconds,
                crossing.silence_duration_seconds,
            )
            action_seconds = min(action_seconds, score_action_seconds)
        is_eot = target.yield_probability >= evaluation.target_yield_threshold
        if is_eot:
            eot_support += 1
            if crossing is not None and score_action_seconds <= policy.timeout_seconds:
                detector_eot_count += 1
            latencies.append(action_seconds)
        else:
            hold_support += 1
            candidate_duration = candidate.end_seconds - candidate.start_seconds
            if action_seconds < candidate_duration - TIMESTAMP_TOLERANCE_SECONDS:
                false_cutoff_count += 1
    return PolicyMetrics(
        candidate_support=evaluated_count,
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


def sweep_policies(
    candidates: Iterable[SilenceCandidate],
    predictions: Iterable[CandidatePrediction],
    thresholds: Iterable[float],
    action_delays_seconds: Iterable[float],
    timeouts_seconds: Iterable[float],
    evaluation: EvaluationConfiguration,
) -> tuple[PolicySweepPoint, ...]:
    candidate_rows = tuple(candidates)
    prediction_rows = tuple(predictions)
    predictions_by_candidate = _predictions_by_candidate(prediction_rows)
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
        PolicySweepPoint(
            policy=policy,
            metrics=_evaluate_policy_with_grouped_predictions(
                candidate_rows=candidate_rows,
                predictions_by_candidate=predictions_by_candidate,
                policy=policy,
                evaluation=evaluation,
            ),
        )
        for policy in policies
    )


def pareto_frontier(points: Iterable[PolicySweepPoint]) -> tuple[PolicySweepPoint, ...]:
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


def best_at_latency_budget(
    points: Iterable[PolicySweepPoint], latency_budget_seconds: float
) -> PolicySweepPoint | None:
    eligible = tuple(
        point
        for point in points
        if point.metrics.mean_latency_seconds is not None
        and point.metrics.mean_latency_seconds <= latency_budget_seconds
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda point: (
            point.metrics.false_cutoff_rate,
            -point.metrics.eot_recall,
            _required_latency(point),
        ),
    )


def best_at_cutoff_budget(
    points: Iterable[PolicySweepPoint], cutoff_budget: float
) -> PolicySweepPoint | None:
    eligible = tuple(
        point
        for point in points
        if point.metrics.mean_latency_seconds is not None
        and point.metrics.false_cutoff_rate <= cutoff_budget
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda point: (
            _required_latency(point),
            -point.metrics.eot_recall,
            point.metrics.false_cutoff_rate,
        ),
    )


def calibration_metrics(
    candidates: Iterable[SilenceCandidate],
    predictions: Iterable[CandidatePrediction],
    bin_count: int = 10,
) -> CalibrationMetrics:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    pairs: list[tuple[float, float]] = []
    for prediction in predictions:
        candidate = candidate_by_id.get(prediction.candidate_id)
        if candidate is None:
            raise ValueError(f"Prediction references unknown candidate {prediction.candidate_id}.")
        target = _target_at_timestamp(
            target_points=candidate.target_points,
            absolute_time_seconds=prediction.absolute_time_seconds,
        )
        pairs.append((prediction.yield_probability, target.yield_probability))
    bins = _calibration_bins(pairs=pairs, bin_count=bin_count)
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


def evaluate_breakdowns(
    candidates: Iterable[SilenceCandidate],
    predictions: Iterable[CandidatePrediction],
    policy: PolicyConfiguration,
    evaluation: EvaluationConfiguration,
) -> tuple[BreakdownMetrics, ...]:
    candidate_rows = tuple(candidates)
    prediction_rows = tuple(predictions)
    slices: defaultdict[tuple[str, str], list[SilenceCandidate]] = defaultdict(list)
    for candidate in candidate_rows:
        slices[("dataset", candidate.dataset_name)].append(candidate)
        for category in candidate.categories:
            slices[("category", category)].append(candidate)
    return tuple(
        BreakdownMetrics(
            dimension=dimension,
            value=value,
            metrics=evaluate_policy(
                candidates=slice_candidates,
                predictions=prediction_rows,
                policy=policy,
                evaluation=evaluation,
            ),
        )
        for (dimension, value), slice_candidates in sorted(slices.items())
    )


def _predictions_by_candidate(
    predictions: Iterable[CandidatePrediction],
) -> dict[str, tuple[CandidatePrediction, ...]]:
    grouped: defaultdict[str, list[CandidatePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.candidate_id].append(prediction)
    return {
        candidate_id: tuple(
            sorted(values, key=lambda prediction: prediction.silence_duration_seconds)
        )
        for candidate_id, values in grouped.items()
    }


def _target_at_score_point(
    candidate: SilenceCandidate,
    score_point_seconds: float,
) -> CandidateTargetPoint | None:
    return next(
        (
            target
            for target in candidate.target_points
            if target.silence_duration_seconds + TIMESTAMP_TOLERANCE_SECONDS >= score_point_seconds
        ),
        None,
    )


def _target_at_timestamp(
    target_points: tuple[CandidateTargetPoint, ...],
    absolute_time_seconds: float,
) -> CandidateTargetPoint:
    matches = tuple(
        target
        for target in target_points
        if math.isclose(
            target.absolute_time_seconds,
            absolute_time_seconds,
            abs_tol=TIMESTAMP_TOLERANCE_SECONDS,
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Prediction timestamp {absolute_time_seconds} does not match one target point."
        )
    return matches[0]


def _first_threshold_crossing(
    predictions: tuple[CandidatePrediction, ...],
    policy: PolicyConfiguration,
    score_persistence: ScorePersistence,
) -> CandidatePrediction | None:
    return next(
        (
            prediction
            for prediction in predictions
            if prediction.yield_probability >= policy.threshold
            and (
                score_persistence is ScorePersistence.LATCHED
                or prediction.silence_duration_seconds + TIMESTAMP_TOLERANCE_SECONDS
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
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _binary_cross_entropy(prediction: float, target: float) -> float:
    probability = min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, prediction))
    return -(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability))


def _calibration_bins(
    pairs: list[tuple[float, float]], bin_count: int
) -> tuple[CalibrationBin, ...]:
    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for prediction, target in pairs:
        index = min(bin_count - 1, math.floor(prediction * bin_count))
        grouped[index].append((prediction, target))
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


def _dominates(candidate: PolicySweepPoint, other: PolicySweepPoint) -> bool:
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


def _required_latency(point: PolicySweepPoint) -> float:
    assert point.metrics.mean_latency_seconds is not None
    return point.metrics.mean_latency_seconds


def _required_mean(value: float | None) -> float:
    assert value is not None
    return value
