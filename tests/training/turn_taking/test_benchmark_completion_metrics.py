import pytest

from app.training.turn_taking.benchmark_completion_metrics import (
    CompletionEvaluationConfiguration,
    completion_breakdowns,
    completion_calibration_metrics,
    completion_discrimination_metrics,
    select_completion_validation_point,
    sweep_completion_policies,
    validate_completion_predictions,
)
from app.training.turn_taking.benchmark_metrics import ScorePersistence
from app.training.turn_taking.benchmark_models import (
    CompletionBoundaryKind,
    CompletionCandidatePrediction,
    CompletionTargetPoint,
    TurnCompletionCandidate,
)


def test_completion_metrics_use_only_confident_nonconflicting_labels() -> None:
    candidates = (
        _candidate("1", completion=0.1, continuation=0.9, duration=0.4),
        _candidate("2", completion=0.9, continuation=0.1),
        _candidate("3", completion=0.5, continuation=0.5),
        _candidate("4", completion=0.9, continuation=0.9),
    )
    predictions = tuple(
        _prediction(candidate.candidate_id, probability=0.9) for candidate in candidates
    )

    point = sweep_completion_policies(
        candidates=candidates,
        predictions=predictions,
        thresholds=(0.5,),
        action_delays_seconds=(0.16,),
        timeouts_seconds=(0.8,),
        evaluation=_evaluation(),
    )[0]

    assert point.metrics.inventory_support == 4
    assert point.metrics.evaluated_support == 2
    assert point.metrics.ignored_support == 2
    assert point.metrics.conflicting_support == 1
    assert point.metrics.hold_support == 1
    assert point.metrics.eot_support == 1
    assert point.metrics.false_cutoff_rate == pytest.approx(1.0)
    assert point.metrics.eot_recall == pytest.approx(1.0)
    assert point.metrics.mean_latency_seconds == pytest.approx(0.16)


def test_completion_calibration_retains_soft_targets_once_per_candidate() -> None:
    candidates = (
        _candidate("1", completion=0.25, continuation=None),
        _candidate("2", completion=0.75, continuation=None),
    )
    predictions = (
        _prediction(candidates[0].candidate_id, probability=0.2, elapsed=0.08),
        _prediction(candidates[0].candidate_id, probability=0.8, elapsed=0.16),
        _prediction(candidates[1].candidate_id, probability=0.7, elapsed=0.08),
    )

    calibration = completion_calibration_metrics(candidates, predictions, bin_count=2)

    assert calibration.support == 2
    assert calibration.brier_score == pytest.approx(0.0025)


def test_discrimination_uses_one_native_score_and_reports_missing_clean_support() -> None:
    candidates = (
        _candidate("1", completion=0.1, continuation=0.9),
        _candidate("2", completion=0.9, continuation=0.1),
        _candidate("3", completion=0.9, continuation=None),
    )
    predictions = (
        _prediction(candidates[0].candidate_id, probability=0.1, elapsed=0.2),
        _prediction(candidates[0].candidate_id, probability=0.9, elapsed=0.3),
        _prediction(candidates[1].candidate_id, probability=0.8, elapsed=0.2),
    )

    metrics = completion_discrimination_metrics(candidates, predictions, _evaluation())

    assert metrics.native_gate_seconds == (0.2,)
    assert metrics.clean_support == 3
    assert metrics.scored_clean_support == 2
    assert metrics.missing_clean_support == 1
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.eot_average_precision == pytest.approx(1.0)


def test_prediction_preflight_rejects_duplicate_and_inconsistent_times() -> None:
    candidate = _candidate("1", completion=0.9, continuation=None)
    prediction = _prediction(candidate.candidate_id, probability=0.8)

    with pytest.raises(ValueError, match="duplicate candidate timestamp"):
        validate_completion_predictions((candidate,), (prediction, prediction))
    with pytest.raises(ValueError, match="inconsistent"):
        validate_completion_predictions(
            (candidate,),
            (prediction.model_copy(update={"absolute_time_seconds": 1.5}),),
        )
    with pytest.raises(ValueError, match="unknown candidate"):
        validate_completion_predictions(
            (candidate,),
            (prediction.model_copy(update={"candidate_id": "f" * 64}),),
        )


def test_action_at_candidate_end_is_not_detector_recall_and_latency_is_clamped() -> None:
    candidate = _candidate("1", completion=0.9, continuation=None, duration=0.08)
    point = sweep_completion_policies(
        candidates=(candidate,),
        predictions=(_prediction(candidate.candidate_id, probability=0.9),),
        thresholds=(0.5,),
        action_delays_seconds=(0.08,),
        timeouts_seconds=(0.8,),
        evaluation=_evaluation(),
    )[0]

    assert point.metrics.detector_eot_count == 0
    assert point.metrics.eot_recall == 0.0
    assert point.metrics.mean_latency_seconds == pytest.approx(0.08)


def test_selected_policy_breakdowns_include_dataset_and_category() -> None:
    candidate = _candidate("1", completion=0.9, continuation=None)
    prediction = _prediction(candidate.candidate_id, probability=0.9)
    policy = sweep_completion_policies(
        (candidate,),
        (prediction,),
        thresholds=(0.5,),
        action_delays_seconds=(0.08,),
        timeouts_seconds=(0.8,),
        evaluation=_evaluation(),
    )[0].policy

    breakdowns = completion_breakdowns((candidate,), (prediction,), policy, _evaluation())

    assert [(item.dimension, item.value) for item in breakdowns] == [
        ("category", "category"),
        ("dataset", "dataset"),
    ]


def test_cutoff_selection_prioritizes_recall_before_latency() -> None:
    candidates = (
        _candidate("1", completion=0.1, continuation=0.9),
        _candidate("2", completion=0.9, continuation=None),
    )
    predictions = (
        _prediction(candidates[0].candidate_id, probability=0.1),
        _prediction(candidates[1].candidate_id, probability=0.6),
    )
    sweep = sweep_completion_policies(
        candidates,
        predictions,
        thresholds=(0.5, 0.9),
        action_delays_seconds=(0.08,),
        timeouts_seconds=(0.3,),
        evaluation=_evaluation(),
    )

    selected = select_completion_validation_point(sweep)

    assert selected.policy.threshold == pytest.approx(0.5)
    assert selected.metrics.eot_recall == pytest.approx(1.0)


def _evaluation() -> CompletionEvaluationConfiguration:
    return CompletionEvaluationConfiguration(
        hold_completion_maximum=0.2,
        hold_continuation_minimum=0.8,
        eot_completion_minimum=0.8,
        conflicting_continuation_minimum=0.8,
        score_persistence=ScorePersistence.LATCHED,
    )


def _candidate(
    suffix: str,
    completion: float,
    continuation: float | None,
    duration: float = 2.0,
) -> TurnCompletionCandidate:
    return TurnCompletionCandidate(
        candidate_id=suffix * 64,
        dataset_id="dataset-id",
        dataset_name="dataset",
        conversation_id="conversation",
        external_id="external",
        user_side="speaker_1",
        user_audio_path="audio.flac",
        preceding_speech_start_seconds=0.0,
        anchor_seconds=1.0,
        end_seconds=1.0 + duration,
        boundary_kind=(
            CompletionBoundaryKind.CONTINUATION
            if continuation is not None and continuation >= 0.8
            else CompletionBoundaryKind.TERMINAL
        ),
        continuation_probability=continuation,
        target_points=(
            CompletionTargetPoint(
                absolute_time_seconds=1.08,
                elapsed_seconds=0.08,
                completion_probability=completion,
            ),
        ),
        categories=("category",),
        source_window_ids=("window",),
    )


def _prediction(
    candidate_id: str,
    probability: float,
    elapsed: float = 0.08,
) -> CompletionCandidatePrediction:
    return CompletionCandidatePrediction(
        candidate_id=candidate_id,
        absolute_time_seconds=1.0 + elapsed,
        elapsed_seconds=elapsed,
        completion_probability=probability,
        inference_duration_seconds=0.01,
    )
