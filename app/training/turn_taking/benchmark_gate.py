from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field, model_validator

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_metrics import (
    BreakdownMetrics,
    CalibrationMetrics,
    EvaluationConfiguration,
    PolicyConfiguration,
    PolicyMetrics,
    calibration_metrics,
    evaluate_breakdowns,
    evaluate_policy,
)
from app.training.turn_taking.benchmark_models import (
    BenchmarkModel,
    CandidateInventory,
    DetectorKind,
    DetectorProvenance,
    OverlapAuditReport,
    PredictionArtifact,
    VoiceLightDetectorProvenance,
)
from app.training.turn_taking.benchmark_report import BenchmarkAnalysisReport


class LockedDetectorPolicy(BenchmarkModel):
    detector: DetectorProvenance
    validation_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation: EvaluationConfiguration
    policy: PolicyConfiguration
    validation_metrics: PolicyMetrics


class ValidationLockManifest(BenchmarkModel):
    schema_version: str = "voice-light-causal-validation-lock-v1"
    corpus_repository: str
    corpus_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    validation_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    smart_turn_overlap_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    smart_turn_clean_comparative_claim_permitted: bool
    policies: tuple[LockedDetectorPolicy, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_primary_checkpoint(self) -> ValidationLockManifest:
        voice_light_checkpoints = {
            policy.detector.checkpoint_sha256
            for policy in self.policies
            if isinstance(policy.detector, VoiceLightDetectorProvenance)
        }
        if self.primary_checkpoint_sha256 not in voice_light_checkpoints:
            raise ValueError("Primary checkpoint is absent from the locked policies.")
        return self


class TestGateResult(BenchmarkModel):
    detector: DetectorProvenance
    test_predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation: EvaluationConfiguration
    policy: PolicyConfiguration
    metrics: PolicyMetrics
    calibration: CalibrationMetrics | None
    breakdowns: tuple[BreakdownMetrics, ...]


class TestGateReport(BenchmarkModel):
    schema_version: str = "voice-light-causal-test-gate-v1"
    validation_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: TestGateResult


def create_validation_lock(
    inventory: CandidateInventory,
    reports: tuple[BenchmarkAnalysisReport, ...],
    primary_checkpoint_sha256: str,
    overlap_report: OverlapAuditReport,
    overlap_report_sha256: str,
) -> ValidationLockManifest:
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("A validation lock requires a validation inventory.")
    if not reports:
        raise ValueError("A validation lock requires at least one analysis report.")
    for report in reports:
        if report.inventory_sha256 != inventory.manifest.candidate_sha256:
            raise ValueError("Analysis report does not match the validation inventory.")
    policies = tuple(
        LockedDetectorPolicy(
            detector=report.detector,
            validation_predictions_sha256=report.predictions_sha256,
            evaluation=report.evaluation,
            policy=report.selected_validation_point.policy,
            validation_metrics=report.selected_validation_point.metrics,
        )
        for report in reports
    )
    return ValidationLockManifest(
        corpus_repository=inventory.manifest.corpus_repository,
        corpus_revision=inventory.manifest.corpus_revision,
        validation_inventory_sha256=inventory.manifest.candidate_sha256,
        primary_checkpoint_sha256=primary_checkpoint_sha256,
        smart_turn_overlap_audit_sha256=overlap_report_sha256,
        smart_turn_clean_comparative_claim_permitted=(
            overlap_report.clean_comparative_claim_permitted
        ),
        policies=policies,
    )


def evaluate_locked_test_artifact(
    lock: ValidationLockManifest,
    inventory: CandidateInventory,
    artifact: PredictionArtifact,
    lock_sha256: str,
) -> TestGateReport:
    if inventory.manifest.split is not TrainingCorpusSplit.TEST:
        raise ValueError("The final gate requires a test inventory.")
    if artifact.manifest.split is not TrainingCorpusSplit.TEST:
        raise ValueError("The final gate requires test predictions.")
    if artifact.manifest.inventory_sha256 != inventory.manifest.candidate_sha256:
        raise ValueError("Test predictions do not match the test inventory.")
    policy = next(
        (row for row in lock.policies if row.detector == artifact.manifest.detector),
        None,
    )
    if policy is None:
        raise ValueError("Detector provenance is not present in the validation lock.")
    calibration = (
        None
        if artifact.manifest.detector.detector_kind is DetectorKind.SILERO_TIMEOUT
        else calibration_metrics(inventory.candidates, artifact.predictions)
    )
    return TestGateReport(
        validation_lock_sha256=lock_sha256,
        test_inventory_sha256=inventory.manifest.candidate_sha256,
        result=TestGateResult(
            detector=artifact.manifest.detector,
            test_predictions_sha256=artifact.manifest.predictions_sha256,
            evaluation=policy.evaluation,
            policy=policy.policy,
            metrics=evaluate_policy(
                inventory.candidates,
                artifact.predictions,
                policy.policy,
                policy.evaluation,
            ),
            calibration=calibration,
            breakdowns=evaluate_breakdowns(
                inventory.candidates,
                artifact.predictions,
                policy.policy,
                policy.evaluation,
            ),
        ),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_validation_lock(path: Path) -> ValidationLockManifest:
    return ValidationLockManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_validation_lock(path: Path, lock: ValidationLockManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lock.model_dump_json(indent=2), encoding="utf-8")


def read_analysis_report(path: Path) -> BenchmarkAnalysisReport:
    return BenchmarkAnalysisReport.model_validate_json(path.read_text(encoding="utf-8"))
