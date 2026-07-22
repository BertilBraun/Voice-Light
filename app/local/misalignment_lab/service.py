from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.ingestion.local_audio import materialize_sample_track
from app.local.misalignment_lab.models import (
    AlignmentReviewCategory,
    InteractionWindowMetrics,
    MisalignmentCandidatePreview,
    MisalignmentCandidateSummary,
    MisalignmentGlobalCountercheckCandidate,
    MisalignmentGlobalCountercheckJudgment,
    MisalignmentGlobalCountercheckProgress,
    MisalignmentGlobalCountercheckQueueResponse,
    MisalignmentGlobalCountercheckStored,
    MisalignmentJudgment,
    MisalignmentLabProgress,
    MisalignmentOffsetRecommendation,
    MisalignmentQueueResponse,
    MisalignmentRepairCandidate,
    MisalignmentRepairEstimate,
    MisalignmentRepairJudgment,
    MisalignmentRepairProgress,
    MisalignmentRepairQueueResponse,
    MisalignmentRepairScope,
    MisalignmentRepairStoredJudgment,
    MisalignmentStoredJudgment,
    MisalignmentTransitionExclusionPolicy,
    MisalignmentTransitionLocationSource,
    MisalignmentTransitionPreview,
    MisalignmentWindowAnnotation,
)
from app.local.synchronization_review.models import (
    SynchronizationAuditKind,
    SynchronizationAuditReport,
    SynchronizationAuditResult,
    SynchronizationAuditWindow,
)
from app.local.training_samples.models import PreviewWaveformPoint
from app.local.training_samples.service import waveform_window
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationPoint,
    AnnotationSpan,
    ConversationAnnotation,
    QualityResult,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)

CLIP_DURATION_SECONDS = 20.0
BEGINNING_REVIEW_END_SECONDS = 180.0
MANUAL_TRANSITION_EXCLUSION_MARGIN_SECONDS = 10.0
AUTOMATIC_TRANSITION_EXCLUSION_MARGIN_SECONDS = 120.0
LATE_REGION_SECONDS = 360.0
LATE_REGION_START_RATIO = 0.6
WINDOW_STEP_SECONDS = 5.0
RAPID_BOUNDARY_SECONDS = 1.5
MAXIMUM_ALTERNATION_GAP_SECONDS = 6.0
WAVEFORM_POINT_COUNT = 1000
TRANSITION_PREVIEW_DURATION_SECONDS = 120.0
TRANSITION_WAVEFORM_POINT_COUNT = 1200
DEFAULT_QUEUE_SEED = "misalignment-lab-v1"
DEFAULT_QUEUE_SIZE = 50
REPAIR_ESTIMATOR_VERSION = "piecewise-stable-suffix-v1"
CONSTANT_OFFSET_ESTIMATOR_VERSION = "stable-global-offset-v1"
REPAIR_MINIMUM_WINDOW_CONFIDENCE = 0.65
REPAIR_MINIMUM_PERSISTENCE_WINDOWS = 3
REPAIR_MAXIMUM_NEIGHBOR_GAP_SECONDS = 180.0
REPAIR_SHIFT_TOLERANCE_SECONDS = 1.25
REPAIR_MINIMUM_SUFFIX_WINDOWS = 4
REPAIR_MINIMUM_SUFFIX_DURATION_SECONDS = 300.0
REPAIR_BASELINE_SHIFT_TOLERANCE_SECONDS = 0.75
REPAIR_MINIMUM_PREFIX_WINDOWS = 3
REPAIR_MINIMUM_SHIFT_CHANGE_SECONDS = 1.0
CONSTANT_OFFSET_MAXIMUM_SPREAD_SECONDS = 1.25
LIKELY_ALIGNED_MAXIMUM_SHIFT_SECONDS = 1.0
MINIMUM_RECOMMENDED_SHIFT_SECONDS = 0.1
REVIEW_CATEGORY_ORDER = {
    AlignmentReviewCategory.LIKELY_ALIGNED: 0,
    AlignmentReviewCategory.LIKELY_CONSTANT_OFFSET: 1,
    AlignmentReviewCategory.NON_CONSTANT_OR_UNCERTAIN: 2,
}


@dataclass(frozen=True)
class AnnotatedSample:
    dashboard_sample: DashboardSample
    annotation: ConversationAnnotation
    duration_seconds: float
    duration_mismatch_seconds: float | None
    speaker1_audio_sha256: str
    speaker2_audio_sha256: str


@dataclass(frozen=True)
class TimedSpeakerSegment:
    side: SpeakerSide
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class AlignmentAssessment:
    category: AlignmentReviewCategory
    likelihood_score: float
    summary: str
    recommendation: MisalignmentOffsetRecommendation | None


@dataclass(frozen=True)
class TransitionExclusionWindow:
    start_seconds: float
    end_seconds: float


def transition_exclusion_policy() -> MisalignmentTransitionExclusionPolicy:
    return MisalignmentTransitionExclusionPolicy(
        manual_margin_seconds=MANUAL_TRANSITION_EXCLUSION_MARGIN_SECONDS,
        automatic_margin_seconds=AUTOMATIC_TRANSITION_EXCLUSION_MARGIN_SECONDS,
    )


def transition_exclusion_window(
    change_point_seconds: float,
    duration_seconds: float,
    location_source: MisalignmentTransitionLocationSource,
) -> TransitionExclusionWindow:
    if not 0.0 <= change_point_seconds <= duration_seconds:
        raise ValueError("Transition point must lie inside the recording duration.")
    margin_seconds = (
        MANUAL_TRANSITION_EXCLUSION_MARGIN_SECONDS
        if location_source is MisalignmentTransitionLocationSource.MANUAL
        else AUTOMATIC_TRANSITION_EXCLUSION_MARGIN_SECONDS
    )
    return TransitionExclusionWindow(
        start_seconds=max(0.0, change_point_seconds - margin_seconds),
        end_seconds=min(duration_seconds, change_point_seconds + margin_seconds),
    )


def build_global_countercheck_queue(
    dashboard_samples: Sequence[DashboardSample],
    audit_report: SynchronizationAuditReport | None,
    repair_judgments: Sequence[MisalignmentRepairStoredJudgment],
    counterchecks: Sequence[MisalignmentGlobalCountercheckStored],
) -> MisalignmentGlobalCountercheckQueueResponse:
    provisional_repairs = tuple(
        judgment
        for judgment in repair_judgments
        if judgment.repair_scope is MisalignmentRepairScope.GLOBAL_OFFSET
        and judgment.judgment is MisalignmentRepairJudgment.PLAUSIBLE
    )
    dashboard_by_sample_id = {sample.sample.id: sample for sample in dashboard_samples}
    audit_by_sample_id = (
        {result.sample_id: result for result in audit_report.results}
        if audit_report is not None
        else {}
    )
    countercheck_by_sample_id = {
        countercheck.sample_id: countercheck for countercheck in counterchecks
    }
    candidates: list[MisalignmentGlobalCountercheckCandidate] = []
    for provisional_repair in provisional_repairs:
        dashboard_sample = dashboard_by_sample_id.get(provisional_repair.sample_id)
        if dashboard_sample is None:
            continue
        annotated_sample = _annotated_sample(dashboard_sample)
        if annotated_sample is None:
            continue
        beginning, ending = _global_countercheck_windows(
            annotated_sample=annotated_sample,
            audit_result=audit_by_sample_id.get(provisional_repair.sample_id),
            provisional_repair=provisional_repair,
        )
        candidates.append(
            MisalignmentGlobalCountercheckCandidate(
                beginning=beginning,
                ending=ending,
                provisional_repair=provisional_repair,
                stored_judgment=countercheck_by_sample_id.get(provisional_repair.sample_id),
            )
        )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.stored_judgment is not None,
                candidate.beginning.external_id,
            ),
        )
    )
    return MisalignmentGlobalCountercheckQueueResponse(
        candidates=ordered,
        progress=global_countercheck_progress(
            candidate_count=len(ordered),
            counterchecks=counterchecks,
        ),
        exclusion_policy=transition_exclusion_policy(),
    )


def build_misalignment_queue(
    dashboard_samples: Sequence[DashboardSample],
    audit_report: SynchronizationAuditReport | None,
    judgments: Sequence[MisalignmentStoredJudgment],
    seed: str,
    limit: int,
) -> MisalignmentQueueResponse:
    if not seed or len(seed) > 100:
        raise ValueError("Queue seed must contain between 1 and 100 characters.")
    if limit < 1:
        raise ValueError("Queue limit must be positive.")
    annotations = tuple(
        annotated
        for dashboard_sample in dashboard_samples
        if (annotated := _annotated_sample(dashboard_sample)) is not None
    )
    audit_by_sample_id = (
        {result.sample_id: result for result in audit_report.results}
        if audit_report is not None
        else {}
    )
    reviewed_candidate_ids = {judgment.candidate_id for judgment in judgments}
    reviewed_windows_by_sample_id: dict[UUID, tuple[MisalignmentStoredJudgment, ...]] = {
        sample_id: tuple(judgment for judgment in judgments if judgment.sample_id == sample_id)
        for sample_id in {judgment.sample_id for judgment in judgments}
    }
    unsure_sample_ids = {
        judgment.sample_id
        for judgment in judgments
        if judgment.judgment is MisalignmentJudgment.UNSURE
    }
    decided_sample_ids = {
        judgment.sample_id
        for judgment in judgments
        if judgment.judgment
        in (
            MisalignmentJudgment.PLAUSIBLY_ALIGNED,
            MisalignmentJudgment.LIKELY_MISALIGNED,
        )
    }
    candidates: list[MisalignmentCandidateSummary] = []
    for annotated_sample in annotations:
        if annotated_sample.dashboard_sample.sample.id in decided_sample_ids:
            continue
        ranked_windows = _ranked_windows(
            annotated_sample=annotated_sample,
            audit_result=audit_by_sample_id.get(annotated_sample.dashboard_sample.sample.id),
        )
        unreviewed = next(
            (
                candidate
                for candidate in ranked_windows
                if candidate.candidate_id not in reviewed_candidate_ids
                and not _overlaps_reviewed_window(
                    candidate=candidate,
                    reviewed_windows=reviewed_windows_by_sample_id.get(
                        candidate.sample_id,
                        (),
                    ),
                )
            ),
            None,
        )
        if unreviewed is not None:
            candidates.append(unreviewed)
    ordered = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.sample_id in unsure_sample_ids,
                REVIEW_CATEGORY_ORDER[candidate.review_category],
                -candidate.alignment_likelihood_score,
                candidate.external_id,
            ),
        )[:limit]
    )
    return MisalignmentQueueResponse(
        seed=seed,
        requested_count=limit,
        candidates=ordered,
        progress=misalignment_progress(
            judgments=judgments,
            eligible_session_count=len(annotations),
        ),
    )


def _overlaps_reviewed_window(
    candidate: MisalignmentCandidateSummary,
    reviewed_windows: Sequence[MisalignmentStoredJudgment],
) -> bool:
    return any(
        candidate.window_start_seconds < reviewed.window_end_seconds
        and candidate.window_end_seconds > reviewed.window_start_seconds
        for reviewed in reviewed_windows
    )


def build_misalignment_repair_queue(
    dashboard_samples: Sequence[DashboardSample],
    audit_report: SynchronizationAuditReport | None,
    judgments: Sequence[MisalignmentStoredJudgment],
    repair_judgments: Sequence[MisalignmentRepairStoredJudgment],
) -> MisalignmentRepairQueueResponse:
    quarantined_by_sample_id = {
        judgment.sample_id: judgment
        for judgment in judgments
        if judgment.judgment is MisalignmentJudgment.LIKELY_MISALIGNED
    }
    audit_by_sample_id = (
        {result.sample_id: result for result in audit_report.results}
        if audit_report is not None
        else {}
    )
    dashboard_by_sample_id = {sample.sample.id: sample for sample in dashboard_samples}
    repair_judgment_by_sample_id = {judgment.sample_id: judgment for judgment in repair_judgments}
    candidates: list[MisalignmentRepairCandidate] = []
    for sample_id, quarantine_judgment in quarantined_by_sample_id.items():
        dashboard_sample = dashboard_by_sample_id.get(sample_id)
        audit_result = audit_by_sample_id.get(sample_id)
        if dashboard_sample is None or audit_result is None:
            continue
        annotated_sample = _annotated_sample(dashboard_sample)
        if annotated_sample is None:
            continue
        estimate = estimate_piecewise_repair(audit_result=audit_result)
        if estimate is None:
            continue
        candidate = _candidate_summary(
            annotated_sample=annotated_sample,
            audit_result=audit_result,
            start_seconds=quarantine_judgment.window_start_seconds,
        )
        if candidate.candidate_id != quarantine_judgment.candidate_id:
            continue
        candidates.append(
            MisalignmentRepairCandidate(
                candidate=candidate,
                repair_estimate=estimate,
                stored_judgment=repair_judgment_by_sample_id.get(sample_id),
            )
        )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda repair_candidate: (
                repair_candidate.repair_estimate.confidence_score,
                repair_candidate.repair_estimate.stable_second_part_duration_seconds,
                abs(repair_candidate.repair_estimate.shift_change_seconds),
                repair_candidate.candidate.external_id,
            ),
            reverse=True,
        )
    )
    return MisalignmentRepairQueueResponse(
        candidates=ordered,
        progress=misalignment_repair_progress(
            quarantined_session_count=len(quarantined_by_sample_id),
            repair_candidate_count=len(ordered),
            repair_judgments=repair_judgments,
        ),
        exclusion_policy=transition_exclusion_policy(),
    )


def estimate_piecewise_repair(
    audit_result: SynchronizationAuditResult,
) -> MisalignmentRepairEstimate | None:
    reliable_windows = tuple(
        window
        for window in audit_result.windows
        if window.accepted
        and not window.maximum_lag_boundary
        and window.confidence_score >= REPAIR_MINIMUM_WINDOW_CONFIDENCE
        and window.persistence_window_count >= REPAIR_MINIMUM_PERSISTENCE_WINDOWS
    )
    if not reliable_windows:
        return None
    suffix = _stable_final_suffix(windows=reliable_windows)
    if (
        len(suffix) < REPAIR_MINIMUM_SUFFIX_WINDOWS
        or suffix[-1].end_seconds - suffix[0].start_seconds < REPAIR_MINIMUM_SUFFIX_DURATION_SECONDS
    ):
        return None
    suffix_shift = statistics.median(window.estimated_b_shift_seconds for window in suffix)
    prefix_candidates = tuple(
        window
        for window in audit_result.windows
        if not window.maximum_lag_boundary
        and window.end_seconds <= suffix[0].start_seconds
        and abs(window.estimated_b_shift_seconds) <= REPAIR_BASELINE_SHIFT_TOLERANCE_SECONDS
    )
    prefix = _stable_final_prefix(windows=prefix_candidates)
    if len(prefix) < REPAIR_MINIMUM_PREFIX_WINDOWS:
        return None
    prefix_shift = statistics.median(window.estimated_b_shift_seconds for window in prefix)
    shift_change = suffix_shift - prefix_shift
    if abs(shift_change) < REPAIR_MINIMUM_SHIFT_CHANGE_SECONDS:
        return None
    suffix_shifts = tuple(window.estimated_b_shift_seconds for window in suffix)
    shift_spread = max(suffix_shifts) - min(suffix_shifts)
    confidence = _repair_confidence(
        suffix=suffix,
        shift_spread_seconds=shift_spread,
    )
    prefix_midpoint = (prefix[-1].start_seconds + prefix[-1].end_seconds) / 2.0
    suffix_midpoint = (suffix[0].start_seconds + suffix[0].end_seconds) / 2.0
    return MisalignmentRepairEstimate(
        estimator_version=REPAIR_ESTIMATOR_VERSION,
        first_part_shift_seconds=round(prefix_shift, 2),
        predicted_second_part_shift_seconds=round(suffix_shift, 2),
        shift_change_seconds=round(shift_change, 2),
        change_interval_start_seconds=round(prefix_midpoint, 3),
        change_interval_end_seconds=round(suffix_midpoint, 3),
        conservative_first_part_end_seconds=prefix[-1].start_seconds,
        conservative_second_part_start_seconds=suffix[0].end_seconds,
        stable_second_part_start_seconds=suffix[0].start_seconds,
        stable_second_part_end_seconds=suffix[-1].end_seconds,
        stable_second_part_duration_seconds=round(
            suffix[-1].end_seconds - suffix[0].start_seconds,
            3,
        ),
        supporting_window_count=len(suffix),
        shift_spread_seconds=round(shift_spread, 2),
        confidence_score=round(confidence, 3),
    )


def alignment_assessment(
    audit_result: SynchronizationAuditResult | None,
) -> AlignmentAssessment:
    if audit_result is None:
        return AlignmentAssessment(
            category=AlignmentReviewCategory.NON_CONSTANT_OR_UNCERTAIN,
            likelihood_score=0.0,
            summary="No synchronization audit is available; no single shift is safe to recommend.",
            recommendation=None,
        )
    repair_estimate = estimate_piecewise_repair(audit_result=audit_result)
    if repair_estimate is not None:
        return AlignmentAssessment(
            category=AlignmentReviewCategory.LIKELY_CONSTANT_OFFSET,
            likelihood_score=repair_estimate.confidence_score,
            summary=(
                "The later part has one stable offset relative to the aligned beginning. "
                "Toggle the recommended late-track shift before deciding."
            ),
            recommendation=MisalignmentOffsetRecommendation(
                estimator_version=repair_estimate.estimator_version,
                repair_scope=MisalignmentRepairScope.AFTER_CHANGE_POINT,
                shift_seconds=repair_estimate.predicted_second_part_shift_seconds,
                confidence_score=repair_estimate.confidence_score,
                supporting_window_count=repair_estimate.supporting_window_count,
                summary=(
                    f"{repair_estimate.supporting_window_count} stable late windows over "
                    f"{repair_estimate.stable_second_part_duration_seconds / 60.0:.1f} min"
                ),
            ),
        )
    if audit_result.kind is SynchronizationAuditKind.STABLE_OFFSET:
        stable_recommendation = _stable_offset_recommendation(audit_result)
        if stable_recommendation is not None:
            shift_magnitude = abs(stable_recommendation.shift_seconds)
            if shift_magnitude <= LIKELY_ALIGNED_MAXIMUM_SHIFT_SECONDS:
                likelihood = stable_recommendation.confidence_score * (
                    1.0 - shift_magnitude / (2.0 * LIKELY_ALIGNED_MAXIMUM_SHIFT_SECONDS)
                )
                return AlignmentAssessment(
                    category=AlignmentReviewCategory.LIKELY_ALIGNED,
                    likelihood_score=round(likelihood, 3),
                    summary=(
                        "Audit windows stay stable and close to the original timeline. "
                        "Listen to the raw tracks first."
                    ),
                    recommendation=(
                        stable_recommendation
                        if shift_magnitude >= MINIMUM_RECOMMENDED_SHIFT_SECONDS
                        else None
                    ),
                )
            return AlignmentAssessment(
                category=AlignmentReviewCategory.LIKELY_CONSTANT_OFFSET,
                likelihood_score=stable_recommendation.confidence_score,
                summary=(
                    "Audit windows agree on one stable offset. Toggle the recommended shift "
                    "before deciding."
                ),
                recommendation=stable_recommendation,
            )
    return AlignmentAssessment(
        category=AlignmentReviewCategory.NON_CONSTANT_OR_UNCERTAIN,
        likelihood_score=round(max(0.0, 1.0 - audit_result.anomaly_score), 3),
        summary=(
            "The audit does not support one safe constant shift. This recording is intentionally "
            "placed after the aligned and one-offset candidates."
        ),
        recommendation=None,
    )


def _stable_offset_recommendation(
    audit_result: SynchronizationAuditResult,
) -> MisalignmentOffsetRecommendation | None:
    reliable_windows = tuple(
        window
        for window in audit_result.windows
        if window.accepted
        and not window.maximum_lag_boundary
        and window.confidence_score >= REPAIR_MINIMUM_WINDOW_CONFIDENCE
        and window.persistence_window_count >= REPAIR_MINIMUM_PERSISTENCE_WINDOWS
    )
    if len(reliable_windows) < REPAIR_MINIMUM_PREFIX_WINDOWS:
        return None
    shifts = tuple(window.estimated_b_shift_seconds for window in reliable_windows)
    spread = max(shifts) - min(shifts)
    if spread > CONSTANT_OFFSET_MAXIMUM_SPREAD_SECONDS:
        return None
    shift = statistics.median(shifts)
    stability = 1.0 - spread / CONSTANT_OFFSET_MAXIMUM_SPREAD_SECONDS
    evidence_confidence = statistics.mean(window.confidence_score for window in reliable_windows)
    confidence = 0.6 * evidence_confidence + 0.4 * stability
    return MisalignmentOffsetRecommendation(
        estimator_version=CONSTANT_OFFSET_ESTIMATOR_VERSION,
        repair_scope=MisalignmentRepairScope.GLOBAL_OFFSET,
        shift_seconds=round(shift, 2),
        confidence_score=round(confidence, 3),
        supporting_window_count=len(reliable_windows),
        summary=(f"{len(reliable_windows)} agreeing windows; {spread:.2f} s total shift spread"),
    )


def misalignment_repair_progress(
    quarantined_session_count: int,
    repair_candidate_count: int,
    repair_judgments: Sequence[MisalignmentRepairStoredJudgment],
) -> MisalignmentRepairProgress:
    reviewed_sample_ids = {judgment.sample_id for judgment in repair_judgments}
    return MisalignmentRepairProgress(
        quarantined_session_count=quarantined_session_count,
        repair_candidate_count=repair_candidate_count,
        reviewed_repair_count=len(reviewed_sample_ids),
        plausible_repair_count=sum(
            judgment.judgment is MisalignmentRepairJudgment.PLAUSIBLE
            for judgment in repair_judgments
        ),
        rejected_repair_count=sum(
            judgment.judgment is MisalignmentRepairJudgment.NOT_PLAUSIBLE
            for judgment in repair_judgments
        ),
        unsure_repair_count=sum(
            judgment.judgment is MisalignmentRepairJudgment.UNSURE for judgment in repair_judgments
        ),
    )


def global_countercheck_progress(
    candidate_count: int,
    counterchecks: Sequence[MisalignmentGlobalCountercheckStored],
) -> MisalignmentGlobalCountercheckProgress:
    return MisalignmentGlobalCountercheckProgress(
        candidate_count=candidate_count,
        reviewed_count=len({countercheck.sample_id for countercheck in counterchecks}),
        needs_transition_count=sum(
            countercheck.judgment is MisalignmentGlobalCountercheckJudgment.NEEDS_TRANSITION
            for countercheck in counterchecks
        ),
        global_offset_confirmed_count=sum(
            countercheck.judgment is MisalignmentGlobalCountercheckJudgment.GLOBAL_OFFSET_CONFIRMED
            for countercheck in counterchecks
        ),
        not_repairable_count=sum(
            countercheck.judgment is MisalignmentGlobalCountercheckJudgment.NOT_REPAIRABLE
            for countercheck in counterchecks
        ),
        unsure_count=sum(
            countercheck.judgment is MisalignmentGlobalCountercheckJudgment.UNSURE
            for countercheck in counterchecks
        ),
    )


def build_global_countercheck_preview(
    dashboard_sample: DashboardSample,
    audit_report: SynchronizationAuditReport | None,
    provisional_repair: MisalignmentRepairStoredJudgment,
    candidate_id: UUID,
) -> MisalignmentCandidatePreview:
    annotated_sample = _annotated_sample(dashboard_sample)
    if annotated_sample is None:
        raise ValueError("The selected sample has no full conversation annotation.")
    audit_result = _audit_result_for_sample(
        audit_report=audit_report,
        sample_id=dashboard_sample.sample.id,
    )
    candidates = _global_countercheck_windows(
        annotated_sample=annotated_sample,
        audit_result=audit_result,
        provisional_repair=provisional_repair,
    )
    candidate = next(
        (candidate for candidate in candidates if candidate.candidate_id == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError("Countercheck window does not belong to this provisional repair.")
    return _build_candidate_preview(
        annotated_sample=annotated_sample,
        candidate=candidate,
        audit_result=audit_result,
    )


def build_misalignment_preview(
    dashboard_sample: DashboardSample,
    audit_report: SynchronizationAuditReport | None,
    candidate_id: UUID,
) -> MisalignmentCandidatePreview:
    annotated_sample = _annotated_sample(dashboard_sample)
    if annotated_sample is None:
        raise ValueError("The selected sample has no full conversation annotation.")
    audit_result = _audit_result_for_sample(
        audit_report=audit_report,
        sample_id=dashboard_sample.sample.id,
    )
    candidate = next(
        (
            ranked
            for ranked in _ranked_windows(
                annotated_sample=annotated_sample,
                audit_result=audit_result,
            )
            if ranked.candidate_id == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("Candidate does not belong to the selected sample annotation.")
    return _build_candidate_preview(
        annotated_sample=annotated_sample,
        candidate=candidate,
        audit_result=audit_result,
    )


def _build_candidate_preview(
    annotated_sample: AnnotatedSample,
    candidate: MisalignmentCandidateSummary,
    audit_result: SynchronizationAuditResult | None,
) -> MisalignmentCandidatePreview:
    dashboard_sample = annotated_sample.dashboard_sample
    speaker1_path = _track_path(dashboard_sample=dashboard_sample, side=TrackSide.SPEAKER1)
    speaker2_path = _track_path(dashboard_sample=dashboard_sample, side=TrackSide.SPEAKER2)
    _, speaker1_waveform = waveform_window(
        path=speaker1_path,
        start_seconds=candidate.window_start_seconds,
        end_seconds=candidate.window_end_seconds,
        point_count=WAVEFORM_POINT_COUNT,
    )
    _, speaker2_waveform = waveform_window(
        path=speaker2_path,
        start_seconds=candidate.window_start_seconds,
        end_seconds=candidate.window_end_seconds,
        point_count=WAVEFORM_POINT_COUNT,
    )
    repair_estimate = (
        estimate_piecewise_repair(audit_result=audit_result) if audit_result is not None else None
    )
    predicted_speaker2_waveform: tuple[PreviewWaveformPoint, ...] | None = None
    predicted_speaker2: MisalignmentWindowAnnotation | None = None
    if candidate.offset_recommendation is not None:
        predicted_shift = candidate.offset_recommendation.shift_seconds
        predicted_source_start = candidate.window_start_seconds - predicted_shift
        predicted_source_end = candidate.window_end_seconds - predicted_shift
        if predicted_source_start >= 0.0:
            _, predicted_speaker2_waveform = waveform_window(
                path=speaker2_path,
                start_seconds=predicted_source_start,
                end_seconds=predicted_source_end,
                point_count=WAVEFORM_POINT_COUNT,
            )
            predicted_speaker2 = _shift_window_annotation(
                annotation=_window_annotation(
                    annotation=annotated_sample.annotation.speaker2,
                    side=SpeakerSide.SPEAKER2,
                    start_seconds=predicted_source_start,
                    end_seconds=predicted_source_end,
                ),
                shift_seconds=predicted_shift,
            )
    return MisalignmentCandidatePreview(
        candidate=candidate,
        annotation_version=annotated_sample.annotation.annotation_version,
        speaker1_waveform=speaker1_waveform,
        speaker2_waveform=speaker2_waveform,
        predicted_speaker2_waveform=predicted_speaker2_waveform,
        speaker1=_window_annotation(
            annotation=annotated_sample.annotation.speaker1,
            side=SpeakerSide.SPEAKER1,
            start_seconds=candidate.window_start_seconds,
            end_seconds=candidate.window_end_seconds,
        ),
        speaker2=_window_annotation(
            annotation=annotated_sample.annotation.speaker2,
            side=SpeakerSide.SPEAKER2,
            start_seconds=candidate.window_start_seconds,
            end_seconds=candidate.window_end_seconds,
        ),
        predicted_speaker2=predicted_speaker2,
        repair_estimate=repair_estimate,
    )


def build_transition_preview(
    dashboard_sample: DashboardSample,
    audit_report: SynchronizationAuditReport | None,
    candidate_id: UUID,
    center_seconds: float | None,
) -> MisalignmentTransitionPreview:
    annotated_sample = _annotated_sample(dashboard_sample)
    if annotated_sample is None:
        raise ValueError("The selected sample has no full conversation annotation.")
    audit_result = _audit_result_for_sample(
        audit_report=audit_report,
        sample_id=dashboard_sample.sample.id,
    )
    if audit_result is None:
        raise ValueError("No synchronization audit exists for this sample.")
    repair_estimate = estimate_piecewise_repair(audit_result=audit_result)
    if repair_estimate is None:
        raise ValueError("No single piecewise transition can be estimated for this sample.")
    candidate = next(
        (
            ranked
            for ranked in _ranked_windows(
                annotated_sample=annotated_sample,
                audit_result=audit_result,
            )
            if ranked.candidate_id == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("Candidate does not belong to the selected sample annotation.")
    estimated_change_point = (
        repair_estimate.change_interval_start_seconds + repair_estimate.change_interval_end_seconds
    ) / 2.0
    selected_center = estimated_change_point if center_seconds is None else center_seconds
    if not (
        repair_estimate.conservative_first_part_end_seconds
        <= selected_center
        <= repair_estimate.stable_second_part_end_seconds
    ):
        raise ValueError(
            "Transition preview center must stay inside the supported review interval."
        )
    window_start, window_end = _transition_window(
        duration_seconds=annotated_sample.duration_seconds,
        center_seconds=selected_center,
        first_shift_seconds=repair_estimate.first_part_shift_seconds,
        second_shift_seconds=repair_estimate.predicted_second_part_shift_seconds,
    )
    speaker1_path = _track_path(dashboard_sample=dashboard_sample, side=TrackSide.SPEAKER1)
    speaker2_path = _track_path(dashboard_sample=dashboard_sample, side=TrackSide.SPEAKER2)
    _, speaker1_waveform = waveform_window(
        path=speaker1_path,
        start_seconds=window_start,
        end_seconds=window_end,
        point_count=TRANSITION_WAVEFORM_POINT_COUNT,
    )
    _, speaker2_raw_waveform = waveform_window(
        path=speaker2_path,
        start_seconds=window_start,
        end_seconds=window_end,
        point_count=TRANSITION_WAVEFORM_POINT_COUNT,
    )
    first_source_start = window_start - repair_estimate.first_part_shift_seconds
    first_source_end = window_end - repair_estimate.first_part_shift_seconds
    _, speaker2_first_waveform = waveform_window(
        path=speaker2_path,
        start_seconds=first_source_start,
        end_seconds=first_source_end,
        point_count=TRANSITION_WAVEFORM_POINT_COUNT,
    )
    second_source_start = window_start - repair_estimate.predicted_second_part_shift_seconds
    second_source_end = window_end - repair_estimate.predicted_second_part_shift_seconds
    _, speaker2_second_waveform = waveform_window(
        path=speaker2_path,
        start_seconds=second_source_start,
        end_seconds=second_source_end,
        point_count=TRANSITION_WAVEFORM_POINT_COUNT,
    )
    return MisalignmentTransitionPreview(
        sample_id=dashboard_sample.sample.id,
        candidate_id=candidate.candidate_id,
        external_id=dashboard_sample.sample.external_id,
        window_start_seconds=round(window_start, 3),
        window_end_seconds=round(window_end, 3),
        estimated_change_point_seconds=round(estimated_change_point, 3),
        change_interval_start_seconds=repair_estimate.change_interval_start_seconds,
        change_interval_end_seconds=repair_estimate.change_interval_end_seconds,
        search_start_seconds=repair_estimate.conservative_first_part_end_seconds,
        search_end_seconds=repair_estimate.stable_second_part_end_seconds,
        first_part_shift_seconds=repair_estimate.first_part_shift_seconds,
        second_part_shift_seconds=repair_estimate.predicted_second_part_shift_seconds,
        speaker1_waveform=speaker1_waveform,
        speaker2_raw_waveform=speaker2_raw_waveform,
        speaker2_first_alignment_waveform=speaker2_first_waveform,
        speaker2_second_alignment_waveform=speaker2_second_waveform,
        speaker1=_window_annotation(
            annotation=annotated_sample.annotation.speaker1,
            side=SpeakerSide.SPEAKER1,
            start_seconds=window_start,
            end_seconds=window_end,
        ),
        speaker2_raw=_window_annotation(
            annotation=annotated_sample.annotation.speaker2,
            side=SpeakerSide.SPEAKER2,
            start_seconds=window_start,
            end_seconds=window_end,
        ),
        speaker2_first_alignment=_shift_window_annotation(
            annotation=_window_annotation(
                annotation=annotated_sample.annotation.speaker2,
                side=SpeakerSide.SPEAKER2,
                start_seconds=first_source_start,
                end_seconds=first_source_end,
            ),
            shift_seconds=repair_estimate.first_part_shift_seconds,
        ),
        speaker2_second_alignment=_shift_window_annotation(
            annotation=_window_annotation(
                annotation=annotated_sample.annotation.speaker2,
                side=SpeakerSide.SPEAKER2,
                start_seconds=second_source_start,
                end_seconds=second_source_end,
            ),
            shift_seconds=repair_estimate.predicted_second_part_shift_seconds,
        ),
    )


def _transition_window(
    duration_seconds: float,
    center_seconds: float,
    first_shift_seconds: float,
    second_shift_seconds: float,
) -> tuple[float, float]:
    earliest_output_seconds = max(0.0, first_shift_seconds, second_shift_seconds)
    latest_output_seconds = min(
        duration_seconds,
        duration_seconds + first_shift_seconds,
        duration_seconds + second_shift_seconds,
    )
    available_duration = latest_output_seconds - earliest_output_seconds
    if available_duration <= 0.0:
        raise ValueError("No common audio range exists for the estimated shifts.")
    preview_duration = min(TRANSITION_PREVIEW_DURATION_SECONDS, available_duration)
    window_start = min(
        max(center_seconds - preview_duration / 2.0, earliest_output_seconds),
        latest_output_seconds - preview_duration,
    )
    return window_start, window_start + preview_duration


def validate_judgment_candidate(
    dashboard_sample: DashboardSample,
    audit_report: SynchronizationAuditReport | None,
    candidate_id: UUID,
    start_seconds: float,
    end_seconds: float,
) -> MisalignmentCandidateSummary:
    annotated_sample = _annotated_sample(dashboard_sample)
    if annotated_sample is None:
        raise ValueError("The selected sample has no full conversation annotation.")
    audit_result = _audit_result_for_sample(
        audit_report=audit_report,
        sample_id=dashboard_sample.sample.id,
    )
    candidate = next(
        (
            ranked
            for ranked in _ranked_windows(
                annotated_sample=annotated_sample,
                audit_result=audit_result,
            )
            if ranked.candidate_id == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("Candidate does not belong to the selected sample annotation.")
    if (
        abs(candidate.window_start_seconds - start_seconds) > 1e-6
        or abs(candidate.window_end_seconds - end_seconds) > 1e-6
    ):
        raise ValueError("Judgment window does not match the generated candidate.")
    return candidate


def _audit_result_for_sample(
    audit_report: SynchronizationAuditReport | None,
    sample_id: UUID,
) -> SynchronizationAuditResult | None:
    if audit_report is None:
        return None
    return next(
        (result for result in audit_report.results if result.sample_id == sample_id),
        None,
    )


def misalignment_progress(
    judgments: Sequence[MisalignmentStoredJudgment],
    eligible_session_count: int,
) -> MisalignmentLabProgress:
    return MisalignmentLabProgress(
        reviewed_snippet_count=len(judgments),
        plausibly_aligned_count=len(
            {
                judgment.sample_id
                for judgment in judgments
                if judgment.judgment is MisalignmentJudgment.PLAUSIBLY_ALIGNED
            }
        ),
        likely_misaligned_count=len(
            {
                judgment.sample_id
                for judgment in judgments
                if judgment.judgment is MisalignmentJudgment.LIKELY_MISALIGNED
            }
        ),
        unsure_count=sum(
            judgment.judgment is MisalignmentJudgment.UNSURE for judgment in judgments
        ),
        quarantined_session_count=len(
            {
                judgment.sample_id
                for judgment in judgments
                if judgment.judgment is MisalignmentJudgment.LIKELY_MISALIGNED
            }
        ),
        eligible_session_count=eligible_session_count,
    )


def eligible_annotated_session_count(
    dashboard_samples: Sequence[DashboardSample],
) -> int:
    return sum(_annotated_sample(sample) is not None for sample in dashboard_samples)


def _annotated_sample(dashboard_sample: DashboardSample) -> AnnotatedSample | None:
    quality_record = dashboard_sample.latest_quality
    if quality_record is None:
        return None
    quality_result = QualityResult.model_validate(quality_record.payload)
    annotation = quality_result.conversation_annotation
    if annotation is None or annotation.analyzed_duration_seconds < CLIP_DURATION_SECONDS:
        return None
    represented_duration = (
        dashboard_sample.sample.duration_seconds
        if dashboard_sample.sample.duration_seconds is not None
        else annotation.analyzed_duration_seconds
    )
    duration_seconds = min(represented_duration, annotation.analyzed_duration_seconds)
    if duration_seconds < CLIP_DURATION_SECONDS:
        return None
    return AnnotatedSample(
        dashboard_sample=dashboard_sample,
        annotation=annotation,
        duration_seconds=duration_seconds,
        duration_mismatch_seconds=(
            quality_result.audio_quality.duration_gap_seconds
            if quality_result.audio_quality is not None
            else None
        ),
        speaker1_audio_sha256=_track_record(
            dashboard_sample=dashboard_sample,
            side=TrackSide.SPEAKER1,
        ).audio_sha256,
        speaker2_audio_sha256=_track_record(
            dashboard_sample=dashboard_sample,
            side=TrackSide.SPEAKER2,
        ).audio_sha256,
    )


def _global_countercheck_windows(
    annotated_sample: AnnotatedSample,
    audit_result: SynchronizationAuditResult | None,
    provisional_repair: MisalignmentRepairStoredJudgment,
) -> tuple[MisalignmentCandidateSummary, MisalignmentCandidateSummary]:
    if provisional_repair.sample_id != annotated_sample.dashboard_sample.sample.id:
        raise ValueError("Provisional repair does not belong to the selected sample.")
    if (
        provisional_repair.repair_scope is not MisalignmentRepairScope.GLOBAL_OFFSET
        or provisional_repair.judgment is not MisalignmentRepairJudgment.PLAUSIBLE
    ):
        raise ValueError("Counterchecks require a plausible provisional global repair.")
    current_assessment = alignment_assessment(audit_result=audit_result)
    current_recommendation = current_assessment.recommendation
    confidence_score = (
        current_recommendation.confidence_score
        if current_recommendation is not None
        and current_recommendation.estimator_version == provisional_repair.estimator_version
        and abs(current_recommendation.shift_seconds - provisional_repair.predicted_shift_seconds)
        <= 1e-6
        else 0.0
    )
    supporting_window_count = (
        current_recommendation.supporting_window_count if current_recommendation is not None else 0
    )
    alignment = AlignmentAssessment(
        category=AlignmentReviewCategory.LIKELY_CONSTANT_OFFSET,
        likelihood_score=confidence_score,
        summary=(
            "Provisional late-clip approval only. Compare the manually aligned beginning with "
            "the end before assigning a repair scope."
        ),
        recommendation=MisalignmentOffsetRecommendation(
            estimator_version=provisional_repair.estimator_version,
            repair_scope=MisalignmentRepairScope.GLOBAL_OFFSET,
            shift_seconds=provisional_repair.predicted_shift_seconds,
            confidence_score=confidence_score,
            supporting_window_count=supporting_window_count,
            summary="Previously approved from one late clip; beginning was not checked.",
        ),
    )
    minimum_start = max(0.0, provisional_repair.predicted_shift_seconds)
    maximum_start = min(
        annotated_sample.duration_seconds - CLIP_DURATION_SECONDS,
        annotated_sample.duration_seconds
        + provisional_repair.predicted_shift_seconds
        - CLIP_DURATION_SECONDS,
    )
    beginning_maximum = min(
        maximum_start,
        BEGINNING_REVIEW_END_SECONDS - CLIP_DURATION_SECONDS,
    )
    ending_minimum = max(minimum_start, _late_region_start(annotated_sample.duration_seconds))
    if beginning_maximum < minimum_start or maximum_start < ending_minimum:
        raise ValueError("No valid beginning and ending windows exist for this offset.")
    beginning = _highest_interaction_candidate(
        annotated_sample=annotated_sample,
        audit_result=audit_result,
        alignment=alignment,
        minimum_start=minimum_start,
        maximum_start=beginning_maximum,
    )
    ending = _highest_interaction_candidate(
        annotated_sample=annotated_sample,
        audit_result=audit_result,
        alignment=alignment,
        minimum_start=ending_minimum,
        maximum_start=maximum_start,
    )
    return beginning, ending


def _highest_interaction_candidate(
    annotated_sample: AnnotatedSample,
    audit_result: SynchronizationAuditResult | None,
    alignment: AlignmentAssessment,
    minimum_start: float,
    maximum_start: float,
) -> MisalignmentCandidateSummary:
    starts = _candidate_starts_between(
        annotation=annotated_sample.annotation,
        minimum_start=minimum_start,
        maximum_start=maximum_start,
    )
    candidates = tuple(
        _candidate_summary(
            annotated_sample=annotated_sample,
            audit_result=audit_result,
            start_seconds=start_seconds,
            alignment=alignment,
        )
        for start_seconds in starts
    )
    return max(
        candidates,
        key=lambda candidate: (
            candidate.interaction.interaction_score,
            candidate.window_start_seconds,
        ),
    )


def _candidate_starts_between(
    annotation: ConversationAnnotation,
    minimum_start: float,
    maximum_start: float,
) -> tuple[float, ...]:
    grid_start = math.ceil(minimum_start / WINDOW_STEP_SECONDS) * WINDOW_STEP_SECONDS
    starts = {
        round(start, 3) for start in _float_range(grid_start, maximum_start, WINDOW_STEP_SECONDS)
    }
    event_times = (
        *(point.time_seconds for point in annotation.speaker1.turns),
        *(point.time_seconds for point in annotation.speaker2.turns),
        *(span.start_seconds for span in annotation.speaker1.backchannels),
        *(span.start_seconds for span in annotation.speaker2.backchannels),
        *(point.time_seconds for point in annotation.speaker1.interruptions),
        *(point.time_seconds for point in annotation.speaker2.interruptions),
    )
    for event_time in event_times:
        centered_start = event_time - CLIP_DURATION_SECONDS / 2.0
        if minimum_start <= centered_start <= maximum_start:
            starts.add(round(centered_start, 3))
    starts.add(round(minimum_start, 3))
    starts.add(round(maximum_start, 3))
    return tuple(sorted(starts))


def _ranked_windows(
    annotated_sample: AnnotatedSample,
    audit_result: SynchronizationAuditResult | None,
) -> tuple[MisalignmentCandidateSummary, ...]:
    alignment = alignment_assessment(audit_result=audit_result)
    interaction_sample = replace(
        annotated_sample,
        annotation=_interaction_region_annotation(
            annotation=annotated_sample.annotation,
            start_seconds=_late_region_start(annotated_sample.duration_seconds),
            end_seconds=annotated_sample.duration_seconds,
        ),
    )
    starts = _candidate_starts(
        annotation=interaction_sample.annotation,
        duration_seconds=annotated_sample.duration_seconds,
    )
    candidates = tuple(
        _candidate_summary(
            annotated_sample=interaction_sample,
            audit_result=audit_result,
            start_seconds=start_seconds,
            alignment=alignment,
        )
        for start_seconds in starts
    )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.interaction.interaction_score,
                -candidate.seconds_from_recording_end,
                candidate.window_start_seconds,
            ),
            reverse=True,
        )
    )


def _candidate_starts(
    annotation: ConversationAnnotation,
    duration_seconds: float,
) -> tuple[float, ...]:
    maximum_start = duration_seconds - CLIP_DURATION_SECONDS
    late_start = _late_region_start(duration_seconds)
    grid_start = math.ceil(late_start / WINDOW_STEP_SECONDS) * WINDOW_STEP_SECONDS
    starts = {
        round(start, 3) for start in _float_range(grid_start, maximum_start, WINDOW_STEP_SECONDS)
    }
    event_times = (
        *(point.time_seconds for point in annotation.speaker1.turns),
        *(point.time_seconds for point in annotation.speaker2.turns),
        *(span.start_seconds for span in annotation.speaker1.backchannels),
        *(span.start_seconds for span in annotation.speaker2.backchannels),
        *(point.time_seconds for point in annotation.speaker1.interruptions),
        *(point.time_seconds for point in annotation.speaker2.interruptions),
    )
    for event_time in event_times:
        centered_start = min(
            maximum_start, max(late_start, event_time - CLIP_DURATION_SECONDS / 2.0)
        )
        starts.add(round(centered_start, 3))
    starts.add(round(maximum_start, 3))
    return tuple(sorted(starts))


def _late_region_start(duration_seconds: float) -> float:
    return max(
        0.0,
        duration_seconds - LATE_REGION_SECONDS,
        duration_seconds * LATE_REGION_START_RATIO,
    )


def _interaction_region_annotation(
    annotation: ConversationAnnotation,
    start_seconds: float,
    end_seconds: float,
) -> ConversationAnnotation:
    return annotation.model_copy(
        update={
            "speaker1": _interaction_region_speaker(
                annotation=annotation.speaker1,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            ),
            "speaker2": _interaction_region_speaker(
                annotation=annotation.speaker2,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            ),
        }
    )


def _interaction_region_speaker(
    annotation: SpeakerConversationAnnotation,
    start_seconds: float,
    end_seconds: float,
) -> SpeakerConversationAnnotation:
    return annotation.model_copy(
        update={
            "speech_segments": _clip_spans(
                annotation.speech_segments,
                start_seconds,
                end_seconds,
            ),
            "backchannels": _clip_spans(
                annotation.backchannels,
                start_seconds,
                end_seconds,
            ),
            "turns": _clip_points(annotation.turns, start_seconds, end_seconds),
            "interruptions": _clip_points(
                annotation.interruptions,
                start_seconds,
                end_seconds,
            ),
            "segment_targets": tuple(
                target
                for target in annotation.segment_targets
                if target.end_seconds >= start_seconds and target.start_seconds <= end_seconds
            ),
        }
    )


def _candidate_summary(
    annotated_sample: AnnotatedSample,
    audit_result: SynchronizationAuditResult | None,
    start_seconds: float,
    alignment: AlignmentAssessment | None = None,
) -> MisalignmentCandidateSummary:
    end_seconds = start_seconds + CLIP_DURATION_SECONDS
    interaction = interaction_window_metrics(
        annotation=annotated_sample.annotation,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    suspicion_score, audit_anomaly_score, audit_late_shift_seconds = _suspicion(
        audit_result=audit_result,
        duration_seconds=annotated_sample.duration_seconds,
        duration_mismatch_seconds=annotated_sample.duration_mismatch_seconds,
    )
    interaction_strength = 1.0 - math.exp(-interaction.interaction_score / 10.0)
    sampling_weight = 0.15 + 3.0 * suspicion_score + 0.75 * interaction_strength
    resolved_alignment = alignment or alignment_assessment(audit_result=audit_result)
    sample = annotated_sample.dashboard_sample.sample
    candidate_id = uuid5(
        NAMESPACE_URL,
        (
            f"voice-light:misalignment:{sample.id}:"
            f"{start_seconds:.3f}:{end_seconds:.3f}:"
            f"{annotated_sample.speaker1_audio_sha256}:"
            f"{annotated_sample.speaker2_audio_sha256}:"
            f"{annotated_sample.annotation.annotation_version}"
        ),
    )
    return MisalignmentCandidateSummary(
        candidate_id=candidate_id,
        sample_id=sample.id,
        external_id=sample.external_id,
        window_start_seconds=start_seconds,
        window_end_seconds=end_seconds,
        seconds_from_recording_end=annotated_sample.duration_seconds - end_seconds,
        interaction=interaction,
        suspicion_score=suspicion_score,
        audit_anomaly_score=audit_anomaly_score,
        audit_late_shift_seconds=audit_late_shift_seconds,
        duration_mismatch_seconds=annotated_sample.duration_mismatch_seconds,
        sampling_weight=sampling_weight,
        review_category=resolved_alignment.category,
        alignment_likelihood_score=resolved_alignment.likelihood_score,
        review_category_summary=resolved_alignment.summary,
        offset_recommendation=resolved_alignment.recommendation,
    )


def interaction_window_metrics(
    annotation: ConversationAnnotation,
    start_seconds: float,
    end_seconds: float,
) -> InteractionWindowMetrics:
    speaker_segments = tuple(
        sorted(
            (
                *_timed_segments(
                    side=SpeakerSide.SPEAKER1,
                    targets=annotation.speaker1.segment_targets,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                ),
                *_timed_segments(
                    side=SpeakerSide.SPEAKER2,
                    targets=annotation.speaker2.segment_targets,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                ),
            ),
            key=lambda segment: (segment.start_seconds, segment.end_seconds, segment.side.value),
        )
    )
    alternating_boundaries = 0
    rapid_boundaries = 0
    for earlier, later in zip(speaker_segments, speaker_segments[1:], strict=False):
        if earlier.side is later.side:
            continue
        gap_seconds = later.start_seconds - earlier.end_seconds
        if gap_seconds > MAXIMUM_ALTERNATION_GAP_SECONDS:
            continue
        alternating_boundaries += 1
        if gap_seconds <= RAPID_BOUNDARY_SECONDS:
            rapid_boundaries += 1
    speaker1_spans = _merged_clipped_spans(
        spans=annotation.speaker1.speech_segments,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    speaker2_spans = _merged_clipped_spans(
        spans=annotation.speaker2.speech_segments,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    both_speakers_active_seconds = _intersection_duration(speaker1_spans, speaker2_spans)
    turn_count = _point_count(
        points=(*annotation.speaker1.turns, *annotation.speaker2.turns),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    backchannel_count = _span_count(
        spans=(*annotation.speaker1.backchannels, *annotation.speaker2.backchannels),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    interruption_count = _point_count(
        points=(*annotation.speaker1.interruptions, *annotation.speaker2.interruptions),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    interaction_score = (
        2.0 * alternating_boundaries
        + 1.5 * rapid_boundaries
        + 0.8 * turn_count
        + 1.8 * backchannel_count
        + 2.0 * interruption_count
        + 0.5 * min(6.0, both_speakers_active_seconds)
    )
    return InteractionWindowMetrics(
        alternating_speaker_boundaries=alternating_boundaries,
        rapid_speaker_boundaries=rapid_boundaries,
        turn_count=turn_count,
        backchannel_count=backchannel_count,
        interruption_count=interruption_count,
        both_speakers_active_seconds=round(both_speakers_active_seconds, 3),
        interaction_score=round(interaction_score, 3),
    )


def _suspicion(
    audit_result: SynchronizationAuditResult | None,
    duration_seconds: float,
    duration_mismatch_seconds: float | None,
) -> tuple[float, float | None, float | None]:
    audit_signal = 0.0
    late_shift: float | None = None
    anomaly_score: float | None = None
    if audit_result is not None:
        anomaly_score = audit_result.anomaly_score
        late_windows = tuple(
            window
            for window in audit_result.windows
            if window.end_seconds >= duration_seconds - LATE_REGION_SECONDS
            and window.accepted
            and not window.maximum_lag_boundary
            and abs(window.estimated_b_shift_seconds) > 1.0
        )
        strongest_late_window = max(
            late_windows,
            key=lambda window: (
                len(window.agreeing_transcript_sources),
                window.persistence_window_count,
                window.confidence_score,
                abs(window.estimated_b_shift_seconds),
            ),
            default=None,
        )
        if strongest_late_window is not None:
            late_shift = strongest_late_window.estimated_b_shift_seconds
            transcript_factor = min(
                1.0,
                len(strongest_late_window.agreeing_transcript_sources) / 2.0,
            )
            persistence_factor = min(
                1.0,
                strongest_late_window.persistence_window_count / 3.0,
            )
            audit_signal = audit_result.anomaly_score * (
                0.35 + 0.35 * transcript_factor + 0.30 * persistence_factor
            )
        else:
            audit_signal = 0.2 * audit_result.anomaly_score
    duration_signal = min(
        1.0,
        (duration_mismatch_seconds or 0.0) / 6.0,
    )
    suspicion = 1.0 - (1.0 - audit_signal) * (1.0 - 0.65 * duration_signal)
    return max(0.03, min(1.0, suspicion)), anomaly_score, late_shift


def _stable_final_suffix(
    windows: Sequence[SynchronizationAuditWindow],
) -> tuple[SynchronizationAuditWindow, ...]:
    suffix = [windows[-1]]
    for window in reversed(windows[:-1]):
        if suffix[0].start_seconds - window.start_seconds > REPAIR_MAXIMUM_NEIGHBOR_GAP_SECONDS:
            break
        suffix_median = statistics.median(item.estimated_b_shift_seconds for item in suffix)
        if abs(window.estimated_b_shift_seconds - suffix_median) > REPAIR_SHIFT_TOLERANCE_SECONDS:
            break
        suffix.insert(0, window)
    return tuple(suffix)


def _stable_final_prefix(
    windows: Sequence[SynchronizationAuditWindow],
) -> tuple[SynchronizationAuditWindow, ...]:
    if not windows:
        return ()
    prefix = [windows[-1]]
    for window in reversed(windows[:-1]):
        if prefix[0].start_seconds - window.start_seconds > REPAIR_MAXIMUM_NEIGHBOR_GAP_SECONDS:
            break
        prefix.insert(0, window)
    return tuple(prefix)


def _repair_confidence(
    suffix: Sequence[SynchronizationAuditWindow],
    shift_spread_seconds: float,
) -> float:
    stability = 1.0 - min(1.0, shift_spread_seconds / 1.5)
    coverage_seconds = suffix[-1].end_seconds - suffix[0].start_seconds
    coverage = min(1.0, coverage_seconds / 900.0)
    transcript_agreement = statistics.mean(
        len(window.agreeing_transcript_sources) / 2.0 for window in suffix
    )
    support = min(1.0, len(suffix) / 8.0)
    return 0.30 * stability + 0.25 * coverage + 0.25 * transcript_agreement + 0.20 * support


def _timed_segments(
    side: SpeakerSide,
    targets: Sequence[SegmentAnnotationTarget],
    start_seconds: float,
    end_seconds: float,
) -> tuple[TimedSpeakerSegment, ...]:
    return tuple(
        TimedSpeakerSegment(
            side=side,
            start_seconds=target.start_seconds,
            end_seconds=target.end_seconds,
        )
        for target in targets
        if target.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
        and target.end_seconds >= start_seconds
        and target.start_seconds <= end_seconds
    )


def _merged_clipped_spans(
    spans: Sequence[AnnotationSpan],
    start_seconds: float,
    end_seconds: float,
) -> tuple[tuple[float, float], ...]:
    clipped = sorted(
        (
            max(start_seconds, span.start_seconds),
            min(end_seconds, span.end_seconds),
        )
        for span in spans
        if span.end_seconds > start_seconds and span.start_seconds < end_seconds
    )
    merged: list[tuple[float, float]] = []
    for span_start, span_end in clipped:
        if not merged or span_start > merged[-1][1]:
            merged.append((span_start, span_end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], span_end))
    return tuple(merged)


def _intersection_duration(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
) -> float:
    first_index = 0
    second_index = 0
    duration = 0.0
    while first_index < len(first) and second_index < len(second):
        first_start, first_end = first[first_index]
        second_start, second_end = second[second_index]
        duration += max(0.0, min(first_end, second_end) - max(first_start, second_start))
        if first_end <= second_end:
            first_index += 1
        else:
            second_index += 1
    return duration


def _point_count(
    points: Iterable[AnnotationPoint],
    start_seconds: float,
    end_seconds: float,
) -> int:
    return sum(start_seconds <= point.time_seconds <= end_seconds for point in points)


def _span_count(
    spans: Iterable[AnnotationSpan],
    start_seconds: float,
    end_seconds: float,
) -> int:
    return sum(
        span.end_seconds >= start_seconds and span.start_seconds <= end_seconds for span in spans
    )


def _float_range(start: float, stop: float, step: float) -> Iterable[float]:
    current = start
    while current <= stop + 1e-9:
        yield current
        current += step


def _window_annotation(
    annotation: SpeakerConversationAnnotation,
    side: SpeakerSide,
    start_seconds: float,
    end_seconds: float,
) -> MisalignmentWindowAnnotation:
    return MisalignmentWindowAnnotation(
        side=side,
        speech_segments=_clip_spans(annotation.speech_segments, start_seconds, end_seconds),
        pauses=_clip_spans(annotation.pauses, start_seconds, end_seconds),
        backchannels=_clip_spans(annotation.backchannels, start_seconds, end_seconds),
        turns=_clip_points(annotation.turns, start_seconds, end_seconds),
        interruptions=_clip_points(annotation.interruptions, start_seconds, end_seconds),
        segment_targets=tuple(
            target
            for target in annotation.segment_targets
            if target.end_seconds >= start_seconds and target.start_seconds <= end_seconds
        ),
        connection_targets=tuple(
            target
            for target in annotation.connection_targets
            if target.later_start_seconds >= start_seconds
            and target.earlier_end_seconds <= end_seconds
        ),
    )


def _shift_window_annotation(
    annotation: MisalignmentWindowAnnotation,
    shift_seconds: float,
) -> MisalignmentWindowAnnotation:
    return annotation.model_copy(
        update={
            "speech_segments": tuple(
                span.model_copy(
                    update={
                        "start_seconds": span.start_seconds + shift_seconds,
                        "end_seconds": span.end_seconds + shift_seconds,
                    }
                )
                for span in annotation.speech_segments
            ),
            "pauses": tuple(
                span.model_copy(
                    update={
                        "start_seconds": span.start_seconds + shift_seconds,
                        "end_seconds": span.end_seconds + shift_seconds,
                    }
                )
                for span in annotation.pauses
            ),
            "backchannels": tuple(
                span.model_copy(
                    update={
                        "start_seconds": span.start_seconds + shift_seconds,
                        "end_seconds": span.end_seconds + shift_seconds,
                    }
                )
                for span in annotation.backchannels
            ),
            "turns": tuple(
                point.model_copy(update={"time_seconds": point.time_seconds + shift_seconds})
                for point in annotation.turns
            ),
            "interruptions": tuple(
                point.model_copy(update={"time_seconds": point.time_seconds + shift_seconds})
                for point in annotation.interruptions
            ),
            "segment_targets": tuple(
                target.model_copy(
                    update={
                        "start_seconds": target.start_seconds + shift_seconds,
                        "end_seconds": target.end_seconds + shift_seconds,
                    }
                )
                for target in annotation.segment_targets
            ),
            "connection_targets": tuple(
                target.model_copy(
                    update={
                        "earlier_end_seconds": (target.earlier_end_seconds + shift_seconds),
                        "later_start_seconds": (target.later_start_seconds + shift_seconds),
                    }
                )
                for target in annotation.connection_targets
            ),
        }
    )


def _clip_spans(
    spans: Sequence[AnnotationSpan],
    start_seconds: float,
    end_seconds: float,
) -> tuple[AnnotationSpan, ...]:
    return tuple(
        span
        for span in spans
        if span.end_seconds >= start_seconds and span.start_seconds <= end_seconds
    )


def _clip_points(
    points: Sequence[AnnotationPoint],
    start_seconds: float,
    end_seconds: float,
) -> tuple[AnnotationPoint, ...]:
    return tuple(point for point in points if start_seconds <= point.time_seconds <= end_seconds)


def _track_path(dashboard_sample: DashboardSample, side: TrackSide) -> Path:
    track = _track_record(dashboard_sample=dashboard_sample, side=side)
    return materialize_sample_track(track)


def _track_record(
    dashboard_sample: DashboardSample,
    side: TrackSide,
) -> SampleTrackRecord:
    track = next(
        (candidate for candidate in dashboard_sample.tracks if candidate.side is side),
        None,
    )
    if track is None:
        raise ValueError(f"Sample has no {side.value} track.")
    return track
