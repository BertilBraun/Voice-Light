from pathlib import Path

import pytest

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_gate import (
    LockedDetectorPolicy,
    ValidationLockManifest,
    evaluate_locked_test_artifact,
)
from app.training.turn_taking.benchmark_metrics import (
    EvaluationConfiguration,
    PolicyConfiguration,
    PolicyMetrics,
    ScorePersistence,
)
from app.training.turn_taking.benchmark_models import (
    CandidateInventory,
    CandidateInventoryManifest,
    CandidatePrediction,
    CandidateTargetPoint,
    PredictionArtifact,
    PredictionManifest,
    SilenceCandidate,
    VoiceLightDetectorConfiguration,
    VoiceLightDetectorProvenance,
    candidate_rows_sha256,
    prediction_rows_sha256,
)


def test_final_gate_evaluates_only_matching_locked_policy() -> None:
    inventory, artifact, lock = _fixtures()

    report = evaluate_locked_test_artifact(lock, inventory, artifact, "7" * 64)

    assert report.result.policy.threshold == 0.6
    assert report.result.metrics.eot_recall == 1.0
    assert report.result.calibration is not None


def test_final_gate_rejects_unlocked_detector() -> None:
    inventory, artifact, lock = _fixtures()
    changed_detector = artifact.manifest.detector.model_copy(
        update={"checkpoint_path": str(Path("different.pt"))}
    )
    changed_artifact = artifact.model_copy(
        update={"manifest": artifact.manifest.model_copy(update={"detector": changed_detector})}
    )

    with pytest.raises(ValueError, match="not present"):
        evaluate_locked_test_artifact(lock, inventory, changed_artifact, "7" * 64)


def _fixtures() -> tuple[CandidateInventory, PredictionArtifact, ValidationLockManifest]:
    candidate = SilenceCandidate(
        candidate_id="1" * 64,
        dataset_id="dataset-id",
        dataset_name="dataset",
        conversation_id="conversation",
        external_id="external",
        user_side="speaker1",
        user_audio_path="audio.flac",
        preceding_speech_start_seconds=0.5,
        start_seconds=1.0,
        end_seconds=1.6,
        target_points=(
            CandidateTargetPoint(
                absolute_time_seconds=1.2,
                silence_duration_seconds=0.2,
                yield_probability=0.9,
            ),
        ),
        categories=("turn_shift",),
        source_window_ids=("2" * 64,),
    )
    candidates = (candidate,)
    inventory = CandidateInventory(
        manifest=CandidateInventoryManifest(
            corpus_repository="repository",
            corpus_revision="3" * 40,
            split=TrainingCorpusSplit.TEST,
            frame_seconds=0.08,
            silence_floor_threshold=0.5,
            minimum_silence_seconds=0.08,
            sample_window_count=1,
            conversation_count=1,
            candidate_count=1,
            candidate_sha256=candidate_rows_sha256(candidates),
        ),
        candidates=candidates,
    )
    detector = VoiceLightDetectorProvenance(
        display_name="Voice Light",
        implementation_version="test",
        model_repository="model",
        model_revision="4" * 40,
        checkpoint_path="checkpoint.pt",
        checkpoint_sha256="5" * 64,
        configuration=VoiceLightDetectorConfiguration(
            model_identifier="model",
            lookahead_tokens=0,
            encoder_frame_seconds=0.08,
            optimizer_step=3500,
        ),
    )
    prediction = CandidatePrediction(
        candidate_id=candidate.candidate_id,
        absolute_time_seconds=1.2,
        silence_duration_seconds=0.2,
        yield_probability=0.8,
        inference_duration_seconds=0.01,
    )
    predictions = (prediction,)
    artifact = PredictionArtifact(
        manifest=PredictionManifest(
            inventory_sha256=inventory.manifest.candidate_sha256,
            split=TrainingCorpusSplit.TEST,
            detector=detector,
            prediction_count=1,
            predictions_sha256=prediction_rows_sha256(predictions),
        ),
        predictions=predictions,
    )
    evaluation = EvaluationConfiguration(
        target_score_point_seconds=0.2,
        target_yield_threshold=0.5,
        score_persistence=ScorePersistence.CURRENT,
    )
    policy = PolicyConfiguration(
        threshold=0.6,
        action_delay_seconds=0.2,
        timeout_seconds=0.8,
    )
    metrics = PolicyMetrics(
        candidate_support=1,
        hold_support=0,
        eot_support=1,
        false_cutoff_count=0,
        false_cutoff_rate=0.0,
        detector_eot_count=1,
        eot_recall=1.0,
        mean_latency_seconds=0.2,
        latency_p50_seconds=0.2,
        latency_p90_seconds=0.2,
        latency_p95_seconds=0.2,
        latency_p99_seconds=0.2,
    )
    lock = ValidationLockManifest(
        corpus_repository="repository",
        corpus_revision="3" * 40,
        validation_inventory_sha256="6" * 64,
        primary_checkpoint_sha256=detector.checkpoint_sha256,
        smart_turn_overlap_audit_sha256="8" * 64,
        smart_turn_clean_comparative_claim_permitted=False,
        policies=(
            LockedDetectorPolicy(
                detector=detector,
                validation_predictions_sha256="9" * 64,
                evaluation=evaluation,
                policy=policy,
                validation_metrics=metrics,
            ),
        ),
    )
    return inventory, artifact, lock
