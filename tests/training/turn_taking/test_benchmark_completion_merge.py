import pytest

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_completion_merge import (
    merge_completion_inventories,
    merge_completion_predictions,
)
from app.training.turn_taking.benchmark_models import (
    CompletionBoundaryKind,
    CompletionCandidatePrediction,
    CompletionPredictionArtifact,
    CompletionPredictionManifest,
    CompletionTargetPoint,
    SileroCompletionDetectorProvenance,
    SileroDetectorConfiguration,
    TurnCompletionCandidate,
    TurnCompletionInventory,
    TurnCompletionInventoryManifest,
    completion_prediction_rows_sha256,
    turn_completion_candidate_rows_sha256,
)

CORPUS_REVISION = "1" * 40


def test_merges_inventories_and_predictions() -> None:
    first = _inventory("1")
    second = _inventory("2")

    inventory = merge_completion_inventories((second, first))
    artifact = merge_completion_predictions(
        inventory,
        (_predictions(second), _predictions(first)),
    )

    assert inventory.manifest.sample_window_count == 2
    assert inventory.manifest.conversation_count == 2
    assert [candidate.conversation_id for candidate in inventory.candidates] == ["1", "2"]
    assert artifact.manifest.inventory_sha256 == inventory.manifest.candidate_sha256
    assert artifact.manifest.prediction_count == 2


def test_rejects_duplicate_inventory_candidates() -> None:
    inventory = _inventory("1")

    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        merge_completion_inventories((inventory, inventory))


def _inventory(identifier: str) -> TurnCompletionInventory:
    candidate = TurnCompletionCandidate(
        candidate_id=identifier * 64,
        dataset_id=f"dataset-{identifier}",
        dataset_name="Synthetic",
        conversation_id=identifier,
        external_id=identifier,
        user_side="speaker1",
        user_audio_path=f"audio/{identifier}.flac",
        preceding_speech_start_seconds=0.0,
        anchor_seconds=1.0,
        end_seconds=1.08,
        boundary_kind=CompletionBoundaryKind.TERMINAL,
        continuation_probability=0.0,
        target_points=(
            CompletionTargetPoint(
                absolute_time_seconds=1.08,
                elapsed_seconds=0.08,
                completion_probability=1.0,
            ),
        ),
        categories=("eot",),
        source_window_ids=(identifier * 64,),
    )
    candidates = (candidate,)
    return TurnCompletionInventory(
        manifest=TurnCompletionInventoryManifest(
            corpus_repository="owner/synthetic",
            corpus_revision=CORPUS_REVISION,
            split=TrainingCorpusSplit.VALIDATION,
            frame_seconds=0.08,
            user_floor_threshold=0.5,
            assistant_active_threshold=0.5,
            causal_horizon_seconds=2.0,
            sample_window_count=1,
            conversation_count=1,
            candidate_count=1,
            candidate_sha256=turn_completion_candidate_rows_sha256(candidates),
        ),
        candidates=candidates,
    )


def _predictions(inventory: TurnCompletionInventory) -> CompletionPredictionArtifact:
    candidate = inventory.candidates[0]
    predictions = (
        CompletionCandidatePrediction(
            candidate_id=candidate.candidate_id,
            absolute_time_seconds=1.08,
            elapsed_seconds=0.08,
            completion_probability=0.9,
            inference_duration_seconds=0.01,
        ),
    )
    return CompletionPredictionArtifact(
        manifest=CompletionPredictionManifest(
            inventory_sha256=inventory.manifest.candidate_sha256,
            split=TrainingCorpusSplit.VALIDATION,
            detector=SileroCompletionDetectorProvenance(
                display_name="Silero",
                implementation_version="test",
                package_name="silero-vad",
                package_version="test",
                configuration=SileroDetectorConfiguration(
                    speech_threshold=0.5,
                    minimum_speech_seconds=0.1,
                    minimum_silence_seconds=0.032,
                    use_onnx=True,
                ),
            ),
            prediction_count=1,
            predictions_sha256=completion_prediction_rows_sha256(predictions),
        ),
        predictions=predictions,
    )
