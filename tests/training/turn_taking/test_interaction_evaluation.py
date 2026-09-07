from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.training.turn_taking.interaction_evaluation import (
    InteractionDecision,
    InteractionEventPrediction,
    InteractionKind,
    InteractionObservation,
    InteractionPredictionArtifact,
    InteractionPredictionManifest,
    MissingInteractionEvent,
    MissingInteractionReason,
    evaluate_interaction_predictions,
    interaction_predictions_sha256,
    output_anchor_index,
)


def test_interaction_metrics_expose_confusion_latency_and_missing_coverage() -> None:
    predictions = (
        _prediction("backchannel-correct", InteractionKind.BACKCHANNEL, ((0.0, 0.8, 0.1),)),
        _prediction(
            "backchannel-false-cancel",
            InteractionKind.BACKCHANNEL,
            ((0.0, 0.2, 0.1), (0.08, 0.3, 0.9)),
        ),
        _prediction(
            "interruption-correct",
            InteractionKind.INTERRUPTION,
            ((0.0, 0.1, 0.2), (0.16, 0.2, 0.7)),
        ),
        _prediction("interruption-missed", InteractionKind.INTERRUPTION, ((0.0, 0.2, 0.3),)),
    )
    missing = (
        MissingInteractionEvent(
            event_id="missing",
            interaction_kind=InteractionKind.INTERRUPTION,
            reason=MissingInteractionReason.ANCHOR_NOT_EMITTED,
        ),
    )
    artifact = InteractionPredictionArtifact(
        manifest=InteractionPredictionManifest(
            generated_at=datetime.now(UTC),
            checkpoint_sha256="a" * 64,
            optimizer_step=750,
            source_roots=("v4", "v5"),
            split="validation",
            frame_seconds=0.08,
            detection_horizon_seconds=0.8,
            eligible_event_count=5,
            prediction_count=4,
            missing_event_count=1,
            predictions_sha256=interaction_predictions_sha256(predictions),
        ),
        predictions=predictions,
        missing_events=missing,
    )

    report = evaluate_interaction_predictions(artifact, threshold=0.5)

    assert report.evaluated_event_count == 4
    assert report.missing_event_count == 1
    assert report.backchannel_false_cancel_rate == pytest.approx(0.5)
    assert report.backchannel_recall == pytest.approx(0.5)
    assert report.interruption_floor_take_recall == pytest.approx(0.5)
    assert report.interruption_floor_take_latency.median_seconds == pytest.approx(0.16)
    assert report.interruption_floor_take_latency.censored_count == 1
    assert (
        _confusion_count(
            report.confusion_matrix,
            InteractionKind.BACKCHANNEL,
            InteractionDecision.INTERRUPTION,
        )
        == 1
    )
    assert (
        _confusion_count(
            report.confusion_matrix,
            InteractionKind.INTERRUPTION,
            InteractionDecision.UNRESOLVED,
        )
        == 1
    )
    assert report.missing_reasons[0].count == 1


def test_output_anchor_index_reports_dropped_target_frame() -> None:
    assert output_anchor_index(3, source_frame_count=5, output_frame_count=4) == 2
    assert output_anchor_index(2, source_frame_count=5, output_frame_count=4) is None
    assert output_anchor_index(1, source_frame_count=5, output_frame_count=2) is None


def test_interaction_artifact_rejects_unaccounted_events() -> None:
    with pytest.raises(ValueError, match="account for every eligible event"):
        InteractionPredictionArtifact(
            manifest=InteractionPredictionManifest(
                generated_at=datetime.now(UTC),
                checkpoint_sha256="a" * 64,
                optimizer_step=1,
                source_roots=("v4",),
                split="validation",
                frame_seconds=0.08,
                detection_horizon_seconds=0.8,
                eligible_event_count=1,
                prediction_count=0,
                missing_event_count=0,
                predictions_sha256=interaction_predictions_sha256(()),
            ),
            predictions=(),
            missing_events=(),
        )


def _prediction(
    event_id: str,
    kind: InteractionKind,
    probabilities: tuple[tuple[float, float, float], ...],
) -> InteractionEventPrediction:
    return InteractionEventPrediction(
        event_id=event_id,
        interaction_kind=kind,
        observations=tuple(
            InteractionObservation(
                elapsed_seconds=elapsed,
                non_floor_feedback_probability=backchannel,
                floor_take_probability=interruption,
                backchannel_active=kind is InteractionKind.BACKCHANNEL,
            )
            for elapsed, backchannel, interruption in probabilities
        ),
    )


def _confusion_count(
    values: tuple,
    actual: InteractionKind,
    predicted: InteractionDecision,
) -> int:
    return next(
        value.count for value in values if value.actual is actual and value.predicted is predicted
    )
