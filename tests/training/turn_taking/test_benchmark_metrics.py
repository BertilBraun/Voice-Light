import math

import pytest

from app.training.turn_taking.benchmark_metrics import (
    EvaluationConfiguration,
    PolicyConfiguration,
    ScorePersistence,
    best_at_cutoff_budget,
    best_at_latency_budget,
    calibration_metrics,
    evaluate_breakdowns,
    evaluate_policy,
    pareto_frontier,
    sweep_policies,
)
from app.training.turn_taking.benchmark_models import (
    CandidatePrediction,
    CandidateTargetPoint,
    SilenceCandidate,
)

EVALUATION = EvaluationConfiguration(
    target_score_point_seconds=0.2,
    target_yield_threshold=0.5,
    score_persistence=ScorePersistence.CURRENT,
)


def test_policy_reports_false_cutoffs_latency_recall_and_percentiles() -> None:
    hold = _candidate("1" * 64, 0.1, "hold_pause")
    eot = _candidate("2" * 64, 0.9, "turn_shift")
    predictions = (_prediction(hold, 0.8), _prediction(eot, 0.8))

    metrics = evaluate_policy(
        candidates=(hold, eot),
        predictions=predictions,
        policy=PolicyConfiguration(
            threshold=0.5,
            action_delay_seconds=0.2,
            timeout_seconds=0.5,
        ),
        evaluation=EVALUATION,
    )

    assert metrics.candidate_support == 2
    assert metrics.hold_support == 1
    assert metrics.eot_support == 1
    assert metrics.false_cutoff_rate == 1.0
    assert metrics.eot_recall == 1.0
    assert metrics.mean_latency_seconds == pytest.approx(0.2)
    assert metrics.latency_p99_seconds == pytest.approx(0.2)


def test_timeout_avoids_hold_cutoff_and_exposes_detector_miss() -> None:
    hold = _candidate("1" * 64, 0.1, "hold_pause")
    eot = _candidate("2" * 64, 0.9, "turn_shift")

    metrics = evaluate_policy(
        candidates=(hold, eot),
        predictions=(_prediction(hold, 0.8), _prediction(eot, 0.8)),
        policy=PolicyConfiguration(
            threshold=0.9,
            action_delay_seconds=0.2,
            timeout_seconds=0.8,
        ),
        evaluation=EVALUATION,
    )

    assert metrics.false_cutoff_rate == 0.0
    assert metrics.eot_recall == 0.0
    assert metrics.mean_latency_seconds == pytest.approx(0.8)


def test_calibration_uses_only_matching_native_candidate_timestamps() -> None:
    hold = _candidate("1" * 64, 0.1, "hold_pause")
    eot = _candidate("2" * 64, 0.9, "turn_shift")

    result = calibration_metrics(
        candidates=(hold, eot),
        predictions=(_prediction(hold, 0.8), _prediction(eot, 0.8)),
        bin_count=5,
    )

    expected_bce = (-0.1 * math.log(0.8) - 0.9 * math.log(0.2)) / 2
    expected_bce += (-0.9 * math.log(0.8) - 0.1 * math.log(0.2)) / 2
    assert result.support == 2
    assert result.binary_cross_entropy == pytest.approx(expected_bce)
    assert result.brier_score == pytest.approx(0.25)
    assert result.expected_calibration_error == pytest.approx(0.3)


def test_sweep_operating_points_pareto_and_breakdowns() -> None:
    hold = _candidate("1" * 64, 0.1, "hold_pause")
    eot = _candidate("2" * 64, 0.9, "turn_shift")
    predictions = (_prediction(hold, 0.8), _prediction(eot, 0.8))

    points = sweep_policies(
        candidates=(hold, eot),
        predictions=predictions,
        thresholds=(0.5, 0.9),
        action_delays_seconds=(0.2,),
        timeouts_seconds=(0.5, 0.8),
        evaluation=EVALUATION,
    )

    assert len(points) == 4
    assert pareto_frontier(points)
    assert best_at_latency_budget(points, 0.3) is not None
    assert best_at_cutoff_budget(points, 0.0) is not None
    breakdowns = evaluate_breakdowns(
        candidates=(hold, eot),
        predictions=predictions,
        policy=points[0].policy,
        evaluation=EVALUATION,
    )
    assert {(row.dimension, row.value) for row in breakdowns} == {
        ("category", "hold_pause"),
        ("category", "turn_shift"),
        ("dataset", "dataset"),
    }


def _candidate(candidate_id: str, target: float, category: str) -> SilenceCandidate:
    return SilenceCandidate(
        candidate_id=candidate_id,
        dataset_id="dataset-id",
        dataset_name="dataset",
        conversation_id=f"conversation-{candidate_id[0]}",
        external_id=f"external-{candidate_id[0]}",
        user_side="speaker1",
        user_audio_path=f"audio-{candidate_id[0]}.flac",
        preceding_speech_start_seconds=0.5,
        start_seconds=1.0,
        end_seconds=1.6,
        target_points=(
            CandidateTargetPoint(
                absolute_time_seconds=1.2,
                silence_duration_seconds=0.2,
                yield_probability=target,
            ),
        ),
        categories=(category,),
        source_window_ids=("3" * 64,),
    )


def _prediction(candidate: SilenceCandidate, probability: float) -> CandidatePrediction:
    target = candidate.target_points[0]
    return CandidatePrediction(
        candidate_id=candidate.candidate_id,
        absolute_time_seconds=target.absolute_time_seconds,
        silence_duration_seconds=target.silence_duration_seconds,
        yield_probability=probability,
        inference_duration_seconds=0.01,
    )
