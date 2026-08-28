from __future__ import annotations

from collections.abc import Sequence

from app.training.turn_taking.benchmark_models import (
    CompletionPredictionArtifact,
    CompletionPredictionManifest,
    TurnCompletionInventory,
    TurnCompletionInventoryManifest,
    completion_prediction_rows_sha256,
    turn_completion_candidate_rows_sha256,
)


def merge_completion_inventories(
    inventories: Sequence[TurnCompletionInventory],
) -> TurnCompletionInventory:
    if not inventories:
        raise ValueError("At least one completion inventory is required.")
    reference = inventories[0].manifest
    for inventory in inventories[1:]:
        manifest = inventory.manifest
        if (
            manifest.corpus_repository != reference.corpus_repository
            or manifest.corpus_revision != reference.corpus_revision
            or manifest.split is not reference.split
            or manifest.frame_seconds != reference.frame_seconds
            or manifest.user_floor_threshold != reference.user_floor_threshold
            or manifest.assistant_active_threshold != reference.assistant_active_threshold
            or manifest.causal_horizon_seconds != reference.causal_horizon_seconds
        ):
            raise ValueError("Completion inventories use incompatible benchmark contracts.")
    candidates = tuple(
        sorted(
            (candidate for inventory in inventories for candidate in inventory.candidates),
            key=lambda candidate: candidate.candidate_id,
        )
    )
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("Completion inventories contain duplicate candidate IDs.")
    return TurnCompletionInventory(
        manifest=TurnCompletionInventoryManifest(
            corpus_repository=reference.corpus_repository,
            corpus_revision=reference.corpus_revision,
            split=reference.split,
            frame_seconds=reference.frame_seconds,
            user_floor_threshold=reference.user_floor_threshold,
            assistant_active_threshold=reference.assistant_active_threshold,
            causal_horizon_seconds=reference.causal_horizon_seconds,
            sample_window_count=sum(
                inventory.manifest.sample_window_count for inventory in inventories
            ),
            conversation_count=sum(
                inventory.manifest.conversation_count for inventory in inventories
            ),
            candidate_count=len(candidates),
            candidate_sha256=turn_completion_candidate_rows_sha256(candidates),
        ),
        candidates=candidates,
    )


def merge_completion_predictions(
    inventory: TurnCompletionInventory,
    artifacts: Sequence[CompletionPredictionArtifact],
) -> CompletionPredictionArtifact:
    if not artifacts:
        raise ValueError("At least one completion prediction artifact is required.")
    detector = artifacts[0].manifest.detector
    if any(artifact.manifest.detector != detector for artifact in artifacts[1:]):
        raise ValueError("Completion prediction artifacts use different detectors.")
    predictions = tuple(
        sorted(
            (prediction for artifact in artifacts for prediction in artifact.predictions),
            key=lambda prediction: (
                prediction.candidate_id,
                prediction.absolute_time_seconds,
            ),
        )
    )
    candidate_ids = {candidate.candidate_id for candidate in inventory.candidates}
    prediction_ids = {prediction.candidate_id for prediction in predictions}
    if prediction_ids != candidate_ids:
        raise ValueError("Merged predictions do not cover the merged inventory candidates.")
    return CompletionPredictionArtifact(
        manifest=CompletionPredictionManifest(
            inventory_sha256=inventory.manifest.candidate_sha256,
            split=inventory.manifest.split,
            detector=detector,
            prediction_count=len(predictions),
            predictions_sha256=completion_prediction_rows_sha256(predictions),
        ),
        predictions=predictions,
    )
