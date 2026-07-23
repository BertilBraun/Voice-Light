from __future__ import annotations

import pytest

from app.local.synchronization_review.calibration import ReviewedAlignment
from app.local.synchronization_review.evaluation import (
    BASELINE_CONFIGURATION,
    EvidenceScope,
    OffsetEstimatorConfiguration,
    OffsetEvidence,
    OffsetEvidenceRecord,
    OffsetLabel,
    OffsetWindowEvidence,
    ReviewedOffsetExample,
    ReviewProvenance,
    WeakEvidenceFallback,
    evaluate_offset_estimator,
    predict_offset,
    reviewed_offset_labels,
)
from app.local.synchronization_review.models import SynchronizationEvidenceSource


def test_database_review_overrides_static_calibration_provenance() -> None:
    labels = reviewed_offset_labels(
        stored_alignments=(
            ReviewedAlignment(external_id="sample_284", speaker2_shift_seconds=-5.4),
        )
    )

    label = next(label for label in labels if label.external_id == "sample_284")

    assert label.speaker2_shift_seconds == pytest.approx(-5.4)
    assert label.provenance is ReviewProvenance.DATABASE


def test_zero_fallback_rejects_catastrophic_weak_evidence_shift() -> None:
    evidence = _evidence_record(
        external_id="sample_001",
        source_shifts=(-12.0, -11.8, 8.0),
        improvement=0.005,
        joint_reduction=0.001,
    )
    conservative_configuration = BASELINE_CONFIGURATION.model_copy(
        update={"weak_evidence_fallback": WeakEvidenceFallback.ZERO_SHIFT}
    )

    assert predict_offset(evidence=evidence, configuration=BASELINE_CONFIGURATION) == pytest.approx(
        -11.8
    )
    assert predict_offset(
        evidence=evidence,
        configuration=conservative_configuration,
    ) == pytest.approx(0.0)


def test_static_prediction_is_independent_of_variable_windows() -> None:
    evidence = _evidence_record(
        external_id="sample_001",
        source_shifts=(-3.0, -3.0, -3.0),
        improvement=0.1,
        joint_reduction=0.05,
    )
    evidence = OffsetEvidenceRecord(
        external_id=evidence.external_id,
        scope=evidence.scope,
        sources=evidence.sources,
        windows=(
            OffsetWindowEvidence(
                source=SynchronizationEvidenceSource.PARAKEET,
                start_seconds=0.0,
                end_seconds=180.0,
                estimated_shift_seconds=8.0,
                bad_state_improvement=0.1,
                overlap_reduction=0.05,
                silence_reduction=0.05,
            ),
            OffsetWindowEvidence(
                source=SynchronizationEvidenceSource.PARAKEET,
                start_seconds=180.0,
                end_seconds=360.0,
                estimated_shift_seconds=-8.0,
                bad_state_improvement=0.1,
                overlap_reduction=0.05,
                silence_reduction=0.05,
            ),
        ),
    )

    assert predict_offset(
        evidence=evidence,
        configuration=BASELINE_CONFIGURATION,
    ) == pytest.approx(-3.0)


def test_evaluation_reports_deterministic_out_of_fold_predictions() -> None:
    examples = tuple(
        _reviewed_example(
            index=index,
            reviewed_shift=0.0 if index % 2 == 0 else -2.0,
            source_shift=-10.0 if index % 2 == 0 else -2.0,
            improvement=0.005 if index % 2 == 0 else 0.2,
            joint_reduction=0.001 if index % 2 == 0 else 0.1,
            provenance=(
                ReviewProvenance.DATABASE if index < 6 else ReviewProvenance.STATIC_CALIBRATION
            ),
        )
        for index in range(10)
    )
    configurations = (
        BASELINE_CONFIGURATION,
        BASELINE_CONFIGURATION.model_copy(
            update={"weak_evidence_fallback": WeakEvidenceFallback.ZERO_SHIFT}
        ),
    )

    first_report = evaluate_offset_estimator(
        examples=examples,
        configurations=configurations,
        fold_count=5,
        split_seed="test-seed",
    )
    second_report = evaluate_offset_estimator(
        examples=examples,
        configurations=configurations,
        fold_count=5,
        split_seed="test-seed",
    )

    assert first_report == second_report
    assert first_report.combined_reviewed.baseline_metrics.label_count == 10
    assert first_report.database_reviews.baseline_metrics.label_count == 6
    assert first_report.combined_reviewed.tuned_out_of_fold_metrics.mean_absolute_error_seconds == 0
    assert first_report.combined_reviewed.baseline_metrics.mean_absolute_error_seconds == 5
    assert {sample.fold_index for sample in first_report.samples} == {0, 1, 2, 3, 4}
    assert all(fold.training_label_count == 8 for fold in first_report.folds)
    assert all(fold.validation_label_count == 2 for fold in first_report.folds)


def test_evaluation_rejects_mixed_evidence_scopes() -> None:
    first = _reviewed_example(
        index=1,
        reviewed_shift=0.0,
        source_shift=0.0,
        improvement=0.1,
        joint_reduction=0.1,
        provenance=ReviewProvenance.DATABASE,
    )
    second = _reviewed_example(
        index=2,
        reviewed_shift=0.0,
        source_shift=0.0,
        improvement=0.1,
        joint_reduction=0.1,
        provenance=ReviewProvenance.DATABASE,
        scope=EvidenceScope.FULL_RECORDING,
    )

    with pytest.raises(ValueError, match="same scope"):
        evaluate_offset_estimator(
            examples=(first, second),
            configurations=(BASELINE_CONFIGURATION,),
            fold_count=2,
        )


def test_estimator_configuration_requires_nonnegative_thresholds() -> None:
    with pytest.raises(ValueError):
        OffsetEstimatorConfiguration(
            minimum_meaningful_lag_seconds=-0.1,
            minimum_meaningful_improvement=0.03,
            minimum_joint_reduction=0.008,
            minimum_meaningful_source_count=1,
            weak_evidence_fallback=WeakEvidenceFallback.ZERO_SHIFT,
        )


def _reviewed_example(
    index: int,
    reviewed_shift: float,
    source_shift: float,
    improvement: float,
    joint_reduction: float,
    provenance: ReviewProvenance,
    scope: EvidenceScope = EvidenceScope.INITIAL_180_SECONDS,
) -> ReviewedOffsetExample:
    external_id = f"sample_{index:03d}"
    return ReviewedOffsetExample(
        label=OffsetLabel(
            external_id=external_id,
            speaker2_shift_seconds=reviewed_shift,
            provenance=provenance,
        ),
        evidence=_evidence_record(
            external_id=external_id,
            source_shifts=(source_shift, source_shift, source_shift),
            improvement=improvement,
            joint_reduction=joint_reduction,
            scope=scope,
        ),
    )


def _evidence_record(
    external_id: str,
    source_shifts: tuple[float, float, float],
    improvement: float,
    joint_reduction: float,
    scope: EvidenceScope = EvidenceScope.INITIAL_180_SECONDS,
) -> OffsetEvidenceRecord:
    sources = tuple(
        OffsetEvidence(
            source=source,
            estimated_shift_seconds=shift,
            bad_state_improvement=improvement,
            overlap_reduction=joint_reduction,
            silence_reduction=joint_reduction,
        )
        for source, shift in zip(
            (
                SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
                SynchronizationEvidenceSource.PARAKEET,
                SynchronizationEvidenceSource.CANARY,
            ),
            source_shifts,
            strict=True,
        )
    )
    return OffsetEvidenceRecord(
        external_id=external_id,
        scope=scope,
        sources=sources,
        windows=(
            OffsetWindowEvidence(
                source=SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
                start_seconds=0.0,
                end_seconds=60.0,
                estimated_shift_seconds=source_shifts[0],
                bad_state_improvement=improvement,
                overlap_reduction=joint_reduction,
                silence_reduction=joint_reduction,
            ),
        ),
    )
