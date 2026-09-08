from datetime import UTC, datetime

import pytest

from app.training.turn_taking.interaction_baselines import (
    InteractionBaselineArtifact,
    InteractionBaselineEventPrediction,
    InteractionBaselineManifest,
    InteractionBaselineObservation,
    SileroInteractionProvenance,
    baseline_predictions_sha256,
    evaluate_interaction_baseline,
    sweep_interaction_baseline,
)
from app.training.turn_taking.interaction_evaluation import InteractionKind


def test_evaluate_interaction_baseline_applies_sustained_action_requirement() -> None:
    predictions = (
        _prediction("backchannel", InteractionKind.BACKCHANNEL, (0.9,) * 5),
        _prediction("interruption", InteractionKind.INTERRUPTION, (0.2, 0.8, 0.8, 0.8)),
    )
    artifact = _artifact(predictions)

    report = evaluate_interaction_baseline(artifact, threshold=0.5)

    assert report.backchannel_false_action_rate == 1.0
    assert report.interruption_recall == 0.0
    assert report.backchannel_action_latency.observed_count == 1
    assert report.backchannel_action_latency.median_seconds == pytest.approx(0.0)


def test_interaction_baseline_artifact_rejects_changed_predictions() -> None:
    predictions = (_prediction("event", InteractionKind.INTERRUPTION, (0.8,) * 5),)
    artifact = _artifact(predictions)
    changed = artifact.model_copy(
        update={"predictions": (_prediction("event", InteractionKind.INTERRUPTION, (0.1,) * 5),)}
    )

    with pytest.raises(ValueError, match="manifest hash"):
        InteractionBaselineArtifact.model_validate_json(changed.model_dump_json())


def test_sweep_interaction_baseline_includes_both_endpoints() -> None:
    artifact = _artifact((_prediction("event", InteractionKind.INTERRUPTION, (0.8,) * 5),))

    sweep = sweep_interaction_baseline(artifact, threshold_step=0.25)

    assert tuple(report.threshold for report in sweep.reports) == (0.0, 0.25, 0.5, 0.75, 1.0)


def _artifact(
    predictions: tuple[InteractionBaselineEventPrediction, ...],
) -> InteractionBaselineArtifact:
    return InteractionBaselineArtifact(
        manifest=InteractionBaselineManifest(
            generated_at=datetime.now(UTC),
            detector=SileroInteractionProvenance(package_version="6.2.1"),
            source_roots=("synthetic",),
            detection_horizon_seconds=0.8,
            crop_random_seed=17,
            event_count=len(predictions),
            predictions_sha256=baseline_predictions_sha256(predictions),
        ),
        predictions=predictions,
    )


def _prediction(
    event_id: str,
    kind: InteractionKind,
    probabilities: tuple[float, ...],
) -> InteractionBaselineEventPrediction:
    return InteractionBaselineEventPrediction(
        event_id=event_id,
        interaction_kind=kind,
        observations=tuple(
            InteractionBaselineObservation(
                elapsed_seconds=index * 0.032,
                action_probability=probability,
            )
            for index, probability in enumerate(probabilities)
        ),
        inference_duration_seconds=0.0,
    )
