from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.misalignment_lab.models import (
    InteractionWindowMetrics,
    MisalignmentCandidatePreview,
    MisalignmentCandidateSummary,
    MisalignmentJudgment,
    MisalignmentLabProgress,
    MisalignmentQueueResponse,
    MisalignmentStoredJudgment,
    MisalignmentWindowAnnotation,
)
from app.local.synchronization_review.models import (
    SynchronizationAuditReport,
    SynchronizationAuditResult,
)
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
LATE_REGION_SECONDS = 360.0
LATE_REGION_START_RATIO = 0.6
WINDOW_STEP_SECONDS = 5.0
RAPID_BOUNDARY_SECONDS = 1.5
MAXIMUM_ALTERNATION_GAP_SECONDS = 6.0
WAVEFORM_POINT_COUNT = 1000
DEFAULT_QUEUE_SEED = "misalignment-lab-v1"
DEFAULT_QUEUE_SIZE = 50


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
    quarantined_sample_ids = {
        judgment.sample_id
        for judgment in judgments
        if judgment.judgment is MisalignmentJudgment.LIKELY_MISALIGNED
    }
    candidates: list[MisalignmentCandidateSummary] = []
    for annotated_sample in annotations:
        if annotated_sample.dashboard_sample.sample.id in quarantined_sample_ids:
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
            ),
            None,
        )
        if unreviewed is not None:
            candidates.append(unreviewed)
    ordered = tuple(
        candidate
        for _, candidate in sorted(
            (
                (_weighted_shuffle_key(candidate=candidate, seed=seed), candidate)
                for candidate in candidates
            ),
            key=lambda item: (item[0], item[1].external_id),
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


def build_misalignment_preview(
    dashboard_sample: DashboardSample,
    audit_report: SynchronizationAuditReport | None,
    candidate_id: UUID,
) -> MisalignmentCandidatePreview:
    annotated_sample = _annotated_sample(dashboard_sample)
    if annotated_sample is None:
        raise ValueError("The selected sample has no full conversation annotation.")
    audit_result = (
        next(
            (
                result
                for result in audit_report.results
                if result.sample_id == dashboard_sample.sample.id
            ),
            None,
        )
        if audit_report is not None
        else None
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
    return MisalignmentCandidatePreview(
        candidate=candidate,
        annotation_version=annotated_sample.annotation.annotation_version,
        speaker1_waveform=speaker1_waveform,
        speaker2_waveform=speaker2_waveform,
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
    )


def validate_judgment_candidate(
    dashboard_sample: DashboardSample,
    audit_report: SynchronizationAuditReport | None,
    candidate_id: UUID,
    start_seconds: float,
    end_seconds: float,
) -> None:
    annotated_sample = _annotated_sample(dashboard_sample)
    if annotated_sample is None:
        raise ValueError("The selected sample has no full conversation annotation.")
    audit_result = (
        next(
            (
                result
                for result in audit_report.results
                if result.sample_id == dashboard_sample.sample.id
            ),
            None,
        )
        if audit_report is not None
        else None
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


def misalignment_progress(
    judgments: Sequence[MisalignmentStoredJudgment],
    eligible_session_count: int,
) -> MisalignmentLabProgress:
    return MisalignmentLabProgress(
        reviewed_snippet_count=len(judgments),
        plausibly_aligned_count=sum(
            judgment.judgment is MisalignmentJudgment.PLAUSIBLY_ALIGNED for judgment in judgments
        ),
        likely_misaligned_count=sum(
            judgment.judgment is MisalignmentJudgment.LIKELY_MISALIGNED for judgment in judgments
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


def _ranked_windows(
    annotated_sample: AnnotatedSample,
    audit_result: SynchronizationAuditResult | None,
) -> tuple[MisalignmentCandidateSummary, ...]:
    starts = _candidate_starts(
        annotation=annotated_sample.annotation,
        duration_seconds=annotated_sample.duration_seconds,
    )
    candidates = tuple(
        _candidate_summary(
            annotated_sample=annotated_sample,
            audit_result=audit_result,
            start_seconds=start_seconds,
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
    late_start = max(
        0.0,
        duration_seconds - LATE_REGION_SECONDS,
        duration_seconds * LATE_REGION_START_RATIO,
    )
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


def _candidate_summary(
    annotated_sample: AnnotatedSample,
    audit_result: SynchronizationAuditResult | None,
    start_seconds: float,
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


def _weighted_shuffle_key(
    candidate: MisalignmentCandidateSummary,
    seed: str,
) -> float:
    digest = hashlib.sha256(f"{seed}:{candidate.candidate_id}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], byteorder="big"))
    uniform = max(generator.random(), 1e-12)
    return -math.log(uniform) / candidate.sampling_weight


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
    path = Path(track.access_uri).resolve()
    if not path.is_file():
        raise ValueError(f"Track audio does not exist: {path}")
    return path


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
