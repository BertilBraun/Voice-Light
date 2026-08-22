import hashlib
from pathlib import Path

import pytest

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_completion_audit import (
    CompletionAuditConfiguration,
    CompletionAuditGroup,
    CompletionAuditSelectionReason,
    build_completion_audit_manifest,
)
from app.training.turn_taking.benchmark_models import (
    CompletionBoundaryKind,
    CompletionCandidatePrediction,
    CompletionPredictionArtifact,
    CompletionPredictionManifest,
    CompletionTargetPoint,
    LiveKitCompletionDetectorProvenance,
    LiveKitDetectorConfiguration,
    SmartTurnCompletionDetectorProvenance,
    SmartTurnDetectorConfiguration,
    TurnCompletionCandidate,
    TurnCompletionInventory,
    TurnCompletionInventoryManifest,
    VoiceLightCompletionDetectorConfiguration,
    VoiceLightCompletionDetectorProvenance,
    completion_prediction_rows_sha256,
    turn_completion_candidate_rows_sha256,
)


def test_completion_audit_selects_balanced_challenges_and_controls_deterministically() -> None:
    inventory = _inventory()
    voice_light = _artifact(inventory, "voice_light")
    smart_turn = _artifact(inventory, "smart_turn")
    livekit = _artifact(inventory, "livekit")
    configuration = CompletionAuditConfiguration(
        ambiguous_count=4,
        confident_hold_count=4,
        confident_eot_count=4,
        double_review_count=3,
    )

    first = build_completion_audit_manifest(
        inventory,
        voice_light,
        smart_turn,
        livekit,
        configuration,
    )
    second = build_completion_audit_manifest(
        inventory,
        voice_light,
        smart_turn,
        livekit,
        configuration,
    )

    assert first == second
    assert first.item_count == 12
    assert sum(item.double_review for item in first.items) == 3
    assert {
        group: sum(item.group is group for item in first.items) for group in CompletionAuditGroup
    } == {
        CompletionAuditGroup.AMBIGUOUS: 4,
        CompletionAuditGroup.CONFIDENT_HOLD: 4,
        CompletionAuditGroup.CONFIDENT_EOT: 4,
    }
    assert (
        sum(
            CompletionAuditSelectionReason.REPRESENTATIVE_CONTROL in item.selection_reasons
            for item in first.items
        )
        == 4
    )
    assert len({item.conversation_id for item in first.items}) == 2
    assert all(item.clip_path.startswith("clips/") for item in first.items)


def test_completion_audit_rejects_nonvalidation_inventory() -> None:
    inventory = _inventory().model_copy(
        update={
            "manifest": _inventory().manifest.model_copy(update={"split": TrainingCorpusSplit.TEST})
        }
    )

    with pytest.raises(ValueError, match="restricted to validation"):
        build_completion_audit_manifest(
            inventory,
            _artifact(_inventory(), "voice_light"),
            _artifact(_inventory(), "smart_turn"),
            _artifact(_inventory(), "livekit"),
            CompletionAuditConfiguration(
                ambiguous_count=1,
                confident_hold_count=1,
                confident_eot_count=1,
                double_review_count=1,
            ),
        )


def test_completion_audit_rejects_missing_prediction_coverage() -> None:
    inventory = _inventory()
    voice_light = _artifact(inventory, "voice_light")
    incomplete_predictions = voice_light.predictions[:-1]
    incomplete = voice_light.model_copy(
        update={
            "predictions": incomplete_predictions,
            "manifest": voice_light.manifest.model_copy(
                update={
                    "prediction_count": len(incomplete_predictions),
                    "predictions_sha256": completion_prediction_rows_sha256(incomplete_predictions),
                }
            ),
        }
    )

    with pytest.raises(ValueError, match="one score for every candidate"):
        build_completion_audit_manifest(
            inventory,
            incomplete,
            _artifact(inventory, "smart_turn"),
            _artifact(inventory, "livekit"),
            CompletionAuditConfiguration(
                ambiguous_count=1,
                confident_hold_count=1,
                confident_eot_count=1,
                double_review_count=1,
            ),
        )


def _inventory() -> TurnCompletionInventory:
    candidates = tuple(
        _candidate(index, completion, continuation)
        for index, (completion, continuation) in enumerate(
            (
                (0.05, 0.95),
                (0.10, 0.90),
                (0.15, 0.85),
                (0.20, 0.80),
                (0.00, 1.00),
                (0.10, 0.95),
                (0.85, 0.10),
                (0.90, 0.05),
                (0.95, 0.00),
                (1.00, 0.10),
                (0.90, None),
                (0.80, 0.00),
                (0.30, 0.70),
                (0.40, 0.60),
                (0.50, 0.50),
                (0.60, 0.40),
                (0.70, 0.30),
                (0.50, None),
            )
        )
    )
    return TurnCompletionInventory(
        manifest=TurnCompletionInventoryManifest(
            corpus_repository="owner/corpus",
            corpus_revision="1" * 40,
            split=TrainingCorpusSplit.VALIDATION,
            frame_seconds=0.08,
            user_floor_threshold=0.5,
            assistant_active_threshold=0.5,
            causal_horizon_seconds=2.0,
            sample_window_count=18,
            conversation_count=2,
            candidate_count=len(candidates),
            candidate_sha256=turn_completion_candidate_rows_sha256(candidates),
        ),
        candidates=candidates,
    )


def _candidate(
    index: int,
    completion: float,
    continuation: float | None,
) -> TurnCompletionCandidate:
    candidate_id = hashlib.sha256(f"candidate-{index}".encode()).hexdigest()
    anchor = 5.0 + index
    return TurnCompletionCandidate(
        candidate_id=candidate_id,
        dataset_id=f"dataset-{index % 2}",
        dataset_name=f"dataset-{index % 2}",
        conversation_id=f"conversation-{index % 2}",
        external_id=f"external-{index % 2}",
        user_side="speaker_1",
        user_audio_path=f"audio-{index % 2}.flac",
        preceding_speech_start_seconds=anchor - 1.0,
        anchor_seconds=anchor,
        end_seconds=anchor + 2.0,
        boundary_kind=CompletionBoundaryKind.TERMINAL,
        continuation_probability=continuation,
        target_points=(
            CompletionTargetPoint(
                absolute_time_seconds=anchor + 0.08,
                elapsed_seconds=0.08,
                completion_probability=completion,
            ),
        ),
        categories=("hold_pause" if index % 2 else "turn_shift",),
        source_window_ids=(hashlib.sha256(f"window-{index}".encode()).hexdigest(),),
    )


def _artifact(
    inventory: TurnCompletionInventory,
    role: str,
) -> CompletionPredictionArtifact:
    role_offset = {"voice_light": 0.0, "smart_turn": 0.1, "livekit": -0.1}[role]
    gate = {"voice_light": 0.08, "smart_turn": 0.24, "livekit": 0.32}[role]
    predictions = tuple(
        sorted(
            (
                CompletionCandidatePrediction(
                    candidate_id=candidate.candidate_id,
                    absolute_time_seconds=candidate.anchor_seconds + gate,
                    elapsed_seconds=gate,
                    completion_probability=min(
                        1.0,
                        max(
                            0.0,
                            candidate.target_points[0].completion_probability + role_offset,
                        ),
                    ),
                    inference_duration_seconds=0.01,
                )
                for candidate in inventory.candidates
            ),
            key=lambda prediction: prediction.candidate_id,
        )
    )
    detector = {
        "voice_light": VoiceLightCompletionDetectorProvenance(
            display_name="Voice Light",
            implementation_version="v2",
            model_repository="owner/model",
            model_revision="2" * 40,
            checkpoint_path=str(Path("adapter.pt")),
            checkpoint_sha256="3" * 64,
            configuration=VoiceLightCompletionDetectorConfiguration(
                model_identifier="owner/model",
                lookahead_tokens=1,
                encoder_frame_seconds=0.08,
                optimizer_step=3500,
            ),
        ),
        "smart_turn": SmartTurnCompletionDetectorProvenance(
            display_name="Smart Turn",
            implementation_version="v2",
            runtime_package_name="onnxruntime",
            runtime_package_version="1.0",
            model_repository="owner/smart-turn",
            model_revision="4" * 40,
            model_filename="model.onnx",
            model_sha256="5" * 64,
            configuration=SmartTurnDetectorConfiguration(
                sample_rate_hz=16_000,
                maximum_window_seconds=8.0,
                candidate_silence_seconds=0.2,
            ),
        ),
        "livekit": LiveKitCompletionDetectorProvenance(
            display_name="LiveKit",
            implementation_version="v2",
            package_name="livekit-local-inference",
            package_version="1.0",
            model_sha256=None,
            configuration=LiveKitDetectorConfiguration(
                sample_rate_hz=16_000,
                vad_speech_threshold=0.5,
                candidate_silence_seconds=0.3,
            ),
        ),
    }[role]
    return CompletionPredictionArtifact(
        manifest=CompletionPredictionManifest(
            inventory_sha256=inventory.manifest.candidate_sha256,
            split=TrainingCorpusSplit.VALIDATION,
            detector=detector,
            prediction_count=len(predictions),
            predictions_sha256=completion_prediction_rows_sha256(predictions),
        ),
        predictions=predictions,
    )
