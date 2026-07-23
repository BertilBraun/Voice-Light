from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.local.config import DATA_ROOT, WEB_ROOT
from app.local.synchronization_review.models import (
    SynchronizationAuditKind,
    SynchronizationAuditReport,
    SynchronizationAuditResult,
    SynchronizationAuditWindow,
    SynchronizationCandidate,
    SynchronizationEvidenceSource,
    SynchronizationWindowEstimate,
)
from app.local.synchronization_review.optimized_alignment import (
    ESTIMATOR_VERSION,
    MAXIMUM_SHIFT_SECONDS,
    WINDOW_DURATION_SECONDS,
    WINDOW_HOP_SECONDS,
    ActivityWindowAnalysis,
    full_recording_audio_activity_masks,
    overlapping_window_analyses,
)

AUDIT_ESTIMATOR_VERSION = f"{ESTIMATOR_VERSION}-overlap-audit-v1"
SYNCHRONIZATION_AUDIT_PATH = DATA_ROOT / "dataset_1" / "synchronization-audit.json"
SYNCHRONIZATION_AUDIT_STATIC_PATH = (
    WEB_ROOT / "pages" / "synchronization-review" / "synchronization-audit.generated.json"
)
BOUNDARY_MARGIN_SECONDS = 0.3
PERSISTENCE_SHIFT_TOLERANCE_SECONDS = 0.8
TRANSCRIPT_AGREEMENT_TOLERANCE_SECONDS = 0.8
MINIMUM_ANOMALY_SHIFT_SECONDS = 0.3
MINIMUM_CHANGE_SECONDS = 1.0


def build_synchronization_audit_report(
    candidates: tuple[SynchronizationCandidate, ...],
    track_paths: tuple[tuple[Path, Path], ...],
) -> SynchronizationAuditReport:
    if len(candidates) != len(track_paths):
        raise ValueError("Each synchronization candidate requires a pair of track paths.")
    results = tuple(
        audit_synchronization_candidate(
            candidate=candidate,
            speaker1_path=speaker1_path,
            speaker2_path=speaker2_path,
        )
        for candidate, (speaker1_path, speaker2_path) in zip(
            candidates,
            track_paths,
            strict=True,
        )
    )
    return SynchronizationAuditReport(
        estimator_version=AUDIT_ESTIMATOR_VERSION,
        window_duration_seconds=WINDOW_DURATION_SECONDS,
        window_hop_seconds=WINDOW_HOP_SECONDS,
        analyzed_session_count=len(results),
        generated_at=datetime.now(timezone.utc).isoformat(),
        results=tuple(
            sorted(
                results,
                key=lambda result: (
                    result.anomaly_score,
                    result.temporal_shift_range_seconds,
                    abs(result.strongest_shift_seconds),
                ),
                reverse=True,
            )
        ),
    )


def audit_synchronization_candidate(
    candidate: SynchronizationCandidate,
    speaker1_path: Path,
    speaker2_path: Path,
) -> SynchronizationAuditResult:
    speaker1_mask, speaker2_mask = full_recording_audio_activity_masks(
        speaker1_path=speaker1_path,
        speaker2_path=speaker2_path,
    )
    activity_windows = overlapping_window_analyses(
        speaker1_mask=speaker1_mask,
        speaker2_mask=speaker2_mask,
    )
    transcript_windows = tuple(
        window
        for window in candidate.window_estimates
        if window.source
        in (SynchronizationEvidenceSource.PARAKEET, SynchronizationEvidenceSource.CANARY)
        and window.meaningful
    )
    audit_windows = tuple(
        _audit_window(
            window=window,
            all_activity_windows=activity_windows,
            transcript_windows=transcript_windows,
        )
        for window in activity_windows
    )
    return _audit_result(
        sample_id=candidate.sample_id,
        external_id=candidate.external_id,
        audit_windows=audit_windows,
    )


def rescore_synchronization_audit_report(
    report: SynchronizationAuditReport,
) -> SynchronizationAuditReport:
    results = tuple(
        _audit_result(
            sample_id=result.sample_id,
            external_id=result.external_id,
            audit_windows=result.windows,
        )
        for result in report.results
    )
    return report.model_copy(
        update={
            "results": tuple(
                sorted(
                    results,
                    key=lambda result: (
                        result.anomaly_score,
                        result.temporal_shift_range_seconds,
                        abs(result.strongest_shift_seconds),
                    ),
                    reverse=True,
                )
            )
        }
    )


def _audit_result(
    sample_id: UUID,
    external_id: str,
    audit_windows: tuple[SynchronizationAuditWindow, ...],
) -> SynchronizationAuditResult:
    reliable_windows = tuple(
        window for window in audit_windows if window.accepted and not window.maximum_lag_boundary
    )
    strongest_window = max(
        audit_windows,
        key=lambda window: (
            window.confidence_score * min(1.0, abs(window.estimated_b_shift_seconds) / 2.0),
            abs(window.estimated_b_shift_seconds),
        ),
        default=_empty_audit_window(),
    )
    temporal_shift_range = _persistent_shift_range(windows=reliable_windows)
    kind = _audit_kind(
        strongest_window=strongest_window,
        temporal_shift_range_seconds=temporal_shift_range,
    )
    anomaly_score = _anomaly_score(
        kind=kind,
        strongest_window=strongest_window,
        temporal_shift_range_seconds=temporal_shift_range,
    )
    return SynchronizationAuditResult(
        sample_id=sample_id,
        external_id=external_id,
        kind=kind,
        anomaly_score=anomaly_score,
        strongest_window_start_seconds=strongest_window.start_seconds,
        strongest_window_end_seconds=strongest_window.end_seconds,
        strongest_shift_seconds=strongest_window.estimated_b_shift_seconds,
        temporal_shift_range_seconds=temporal_shift_range,
        summary=_summary(
            kind=kind,
            strongest_window=strongest_window,
            temporal_shift_range_seconds=temporal_shift_range,
        ),
        windows=audit_windows,
    )


def _audit_window(
    window: ActivityWindowAnalysis,
    all_activity_windows: tuple[ActivityWindowAnalysis, ...],
    transcript_windows: tuple[SynchronizationWindowEstimate, ...],
) -> SynchronizationAuditWindow:
    estimate = window.analysis.estimate
    boundary = abs(estimate.shift_seconds) >= MAXIMUM_SHIFT_SECONDS - BOUNDARY_MARGIN_SECONDS
    persistent_windows = tuple(
        candidate
        for candidate in all_activity_windows
        if candidate.analysis.accepted
        and abs(candidate.start_seconds - window.start_seconds) <= 2.0 * WINDOW_HOP_SECONDS
        and abs(candidate.analysis.estimate.shift_seconds - estimate.shift_seconds)
        <= PERSISTENCE_SHIFT_TOLERANCE_SECONDS
    )
    agreeing_sources = tuple(
        source
        for source in (
            SynchronizationEvidenceSource.PARAKEET,
            SynchronizationEvidenceSource.CANARY,
        )
        if _source_agrees(
            source=source,
            window=window,
            shift_seconds=estimate.shift_seconds,
            transcript_windows=transcript_windows,
        )
    )
    confidence = _window_confidence(
        window=window,
        persistence_window_count=len(persistent_windows),
        agreeing_source_count=len(agreeing_sources),
        boundary=boundary,
    )
    return SynchronizationAuditWindow(
        start_seconds=window.start_seconds,
        end_seconds=window.end_seconds,
        estimated_b_shift_seconds=estimate.shift_seconds,
        confidence_score=confidence,
        bad_state_improvement=estimate.improvement_over_zero,
        competing_margin=estimate.competing_margin,
        basin_width_seconds=estimate.basin_width_seconds,
        persistence_window_count=len(persistent_windows),
        agreeing_transcript_sources=agreeing_sources,
        accepted=window.analysis.accepted,
        maximum_lag_boundary=boundary,
    )


def _source_agrees(
    source: SynchronizationEvidenceSource,
    window: ActivityWindowAnalysis,
    shift_seconds: float,
    transcript_windows: tuple[SynchronizationWindowEstimate, ...],
) -> bool:
    midpoint_seconds = (window.start_seconds + window.end_seconds) / 2.0
    matches = tuple(
        estimate
        for estimate in transcript_windows
        if estimate.source is source
        and estimate.start_seconds <= midpoint_seconds < estimate.end_seconds
    )
    return (
        bool(matches)
        and min(abs(estimate.estimated_b_shift_seconds - shift_seconds) for estimate in matches)
        <= TRANSCRIPT_AGREEMENT_TOLERANCE_SECONDS
    )


def _window_confidence(
    window: ActivityWindowAnalysis,
    persistence_window_count: int,
    agreeing_source_count: int,
    boundary: bool,
) -> float:
    if not window.analysis.accepted or boundary:
        return 0.0
    estimate = window.analysis.estimate
    improvement = min(1.0, estimate.improvement_over_zero / 0.01)
    competing_margin = min(1.0, estimate.competing_margin / 0.002)
    persistence = min(1.0, max(0, persistence_window_count - 1) / 2.0)
    transcript_agreement = agreeing_source_count / 2.0
    score = (
        0.30 * math.sqrt(improvement * competing_margin)
        + 0.35 * persistence
        + 0.35 * transcript_agreement
    )
    return min(1.0, max(0.0, score))


def _persistent_shift_range(windows: tuple[SynchronizationAuditWindow, ...]) -> float:
    persistent = tuple(
        window
        for window in windows
        if window.persistence_window_count >= 3 and window.confidence_score >= 0.65
    )
    if len(persistent) < 2:
        return 0.0
    shifts = tuple(window.estimated_b_shift_seconds for window in persistent)
    return max(shifts) - min(shifts)


def _audit_kind(
    strongest_window: SynchronizationAuditWindow,
    temporal_shift_range_seconds: float,
) -> SynchronizationAuditKind:
    if temporal_shift_range_seconds >= MINIMUM_CHANGE_SECONDS:
        return SynchronizationAuditKind.TEMPORAL_CHANGE
    if (
        strongest_window.persistence_window_count >= 2
        and abs(strongest_window.estimated_b_shift_seconds) >= MINIMUM_ANOMALY_SHIFT_SECONDS
    ):
        return SynchronizationAuditKind.STABLE_OFFSET
    return SynchronizationAuditKind.UNCERTAIN


def _anomaly_score(
    kind: SynchronizationAuditKind,
    strongest_window: SynchronizationAuditWindow,
    temporal_shift_range_seconds: float,
) -> float:
    match kind:
        case SynchronizationAuditKind.TEMPORAL_CHANGE:
            magnitude = 1.0 - math.exp(-temporal_shift_range_seconds / 4.0)
        case SynchronizationAuditKind.STABLE_OFFSET:
            magnitude = 1.0 - math.exp(-abs(strongest_window.estimated_b_shift_seconds) / 4.0)
        case SynchronizationAuditKind.UNCERTAIN:
            return min(0.19, strongest_window.confidence_score * 0.19)
    return min(1.0, strongest_window.confidence_score * (0.50 + 0.50 * magnitude))


def _summary(
    kind: SynchronizationAuditKind,
    strongest_window: SynchronizationAuditWindow,
    temporal_shift_range_seconds: float,
) -> str:
    source_count = len(strongest_window.agreeing_transcript_sources)
    match kind:
        case SynchronizationAuditKind.TEMPORAL_CHANGE:
            return (
                f"Persistent local estimates span {temporal_shift_range_seconds:.2f} s; "
                f"{source_count}/2 transcript sources agree at the strongest window."
            )
        case SynchronizationAuditKind.STABLE_OFFSET:
            return (
                f"{strongest_window.persistence_window_count} neighboring windows support "
                f"{strongest_window.estimated_b_shift_seconds:+.2f} s; "
                f"{source_count}/2 transcript sources agree."
            )
        case SynchronizationAuditKind.UNCERTAIN:
            return "No persistent, non-boundary synchronization anomaly is established."


def _empty_audit_window() -> SynchronizationAuditWindow:
    return SynchronizationAuditWindow(
        start_seconds=0.0,
        end_seconds=0.0,
        estimated_b_shift_seconds=0.0,
        confidence_score=0.0,
        bad_state_improvement=0.0,
        competing_margin=0.0,
        basin_width_seconds=0.0,
        persistence_window_count=0,
        agreeing_transcript_sources=(),
        accepted=False,
        maximum_lag_boundary=False,
    )
