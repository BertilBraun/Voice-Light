from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeVar
from uuid import UUID

from app.local.conversation_regions.models import ConversationRegionAnalysis
from app.local.corpus_audit.models import (
    CorpusAuditCategorySummary,
    CorpusAuditConversationSummary,
    CorpusAuditDatasetSummary,
    CorpusAuditPhysicalEventCounts,
    CorpusAuditPilotMetric,
    CorpusAuditRejectionReason,
    CorpusAuditRejectionSummary,
    CorpusAuditReport,
    CorpusAuditRequest,
)
from app.local.corpus_audit.repository import CorpusAuditEvidence
from app.local.db.models import TrackSide
from app.local.training_samples.models import TrainingSamplePropositionKind
from app.local.training_samples.service import (
    BURN_IN_SECONDS,
    FRAME_SECONDS,
    FUTURE_ACTIVITY_WINDOWS_MILLISECONDS,
    INPUT_DURATION_SECONDS,
    MAXIMUM_TURN_SHIFT_GAP_SECONDS,
)
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationSpan,
    ConversationAnnotation,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
)

PILOT_HOURS = 100.0
PILOT_TURN_SHIFTS = 20_000
PILOT_HOLD_PAUSES = 20_000
PILOT_BACKCHANNELS = 5_000
PILOT_INTERRUPTION_OVERLAPS = 2_000
CountKey = TypeVar("CountKey", bound=Hashable)


@dataclass(frozen=True)
class AuditEvent:
    kind: TrainingSamplePropositionKind
    time_seconds: float


@dataclass(frozen=True)
class IntervalIndex:
    starts: tuple[float, ...]
    ends: tuple[float, ...]

    def contains(self, time_seconds: float) -> bool:
        index = bisect_right(self.starts, time_seconds) - 1
        return index >= 0 and time_seconds < self.ends[index]


@dataclass(frozen=True)
class FloorValidityIndex:
    transcript_segments: IntervalIndex
    valid_connections: IntervalIndex
    invalid_connections: IntervalIndex
    audio_activity: IntervalIndex

    def is_valid(self, time_seconds: float) -> bool:
        if self.transcript_segments.contains(time_seconds):
            return True
        if self.valid_connections.contains(time_seconds):
            return True
        if self.invalid_connections.contains(time_seconds):
            return False
        return not self.audio_activity.contains(time_seconds)


@dataclass(frozen=True)
class WindowAudit:
    accepted: bool
    input_duration_seconds: float
    supervised_duration_seconds: float
    effective_supervised_duration_seconds: float
    masked_duration_seconds: float
    category: TrainingSamplePropositionKind
    covered_events: tuple[AuditEvent, ...]
    rejection_reasons: tuple[CorpusAuditRejectionReason, ...]


@dataclass
class MutableAuditSummary:
    conversation_count: int = 0
    accepted_conversation_count: int = 0
    source_duration_seconds: float = 0.0
    usable_source_duration_seconds: float = 0.0
    candidate_window_count: int = 0
    accepted_window_count: int = 0
    input_duration_seconds: float = 0.0
    supervised_duration_seconds: float = 0.0
    effective_supervised_duration_seconds: float = 0.0
    masked_duration_seconds: float = 0.0
    turn_shift_count: int = 0
    pause_count: int = 0
    backchannel_count: int = 0
    interruption_count: int = 0
    overlap_count: int = 0
    category_windows: dict[TrainingSamplePropositionKind, int] = field(default_factory=dict)
    available_events: dict[TrainingSamplePropositionKind, int] = field(default_factory=dict)
    covered_events: dict[TrainingSamplePropositionKind, int] = field(default_factory=dict)
    rejections: dict[CorpusAuditRejectionReason, int] = field(default_factory=dict)


def generate_corpus_audit(
    evidence: Sequence[CorpusAuditEvidence],
    request: CorpusAuditRequest,
) -> CorpusAuditReport:
    _validate_window_config(request)
    totals = MutableAuditSummary()
    dataset_totals: dict[UUID, MutableAuditSummary] = {}
    dataset_names: dict[UUID, str] = {}
    conversations: list[CorpusAuditConversationSummary] = []
    for conversation in evidence:
        dataset_names[conversation.dataset_id] = conversation.dataset_name
        dataset_summary = dataset_totals.setdefault(
            conversation.dataset_id,
            MutableAuditSummary(),
        )
        conversation_summary, conversation_totals = _audit_conversation(
            conversation=conversation,
            request=request,
        )
        conversations.append(conversation_summary)
        _merge_summary(target=dataset_summary, source=conversation_totals)
        _merge_summary(target=totals, source=conversation_totals)
    dataset_summaries = tuple(
        _dataset_summary(
            dataset_id=dataset_id,
            dataset_name=dataset_names[dataset_id],
            summary=dataset_totals[dataset_id],
        )
        for dataset_id in sorted(dataset_totals, key=lambda item: dataset_names[item])
    )
    physical_events = _physical_events(totals)
    return CorpusAuditReport(
        generated_at=datetime.now(UTC),
        config=request,
        dataset_summaries=dataset_summaries,
        conversations=tuple(conversations),
        categories=tuple(
            CorpusAuditCategorySummary(
                kind=kind,
                accepted_window_count=totals.category_windows.get(kind, 0),
                available_oriented_event_count=totals.available_events.get(kind, 0),
                covered_oriented_event_count=totals.covered_events.get(kind, 0),
            )
            for kind in TrainingSamplePropositionKind
        ),
        rejections=tuple(
            CorpusAuditRejectionSummary(
                reason=reason,
                window_count=totals.rejections.get(reason, 0),
            )
            for reason in CorpusAuditRejectionReason
        ),
        pilot_metrics=_pilot_metrics(summary=totals, physical_events=physical_events),
        conversation_count=totals.conversation_count,
        accepted_conversation_count=totals.accepted_conversation_count,
        candidate_window_count=totals.candidate_window_count,
        accepted_window_count=totals.accepted_window_count,
        input_duration_seconds=totals.input_duration_seconds,
        supervised_duration_seconds=totals.supervised_duration_seconds,
        effective_supervised_duration_seconds=totals.effective_supervised_duration_seconds,
        masked_duration_seconds=totals.masked_duration_seconds,
        physical_events=physical_events,
    )


def _validate_window_config(request: CorpusAuditRequest) -> None:
    if request.input_duration_seconds != INPUT_DURATION_SECONDS:
        raise ValueError(f"input_duration_seconds must be {INPUT_DURATION_SECONDS}.")
    if request.burn_in_seconds != BURN_IN_SECONDS:
        raise ValueError(f"burn_in_seconds must be {BURN_IN_SECONDS}.")
    if request.burn_in_seconds >= request.input_duration_seconds:
        raise ValueError("burn_in_seconds must be shorter than input_duration_seconds.")


def _audit_conversation(
    conversation: CorpusAuditEvidence,
    request: CorpusAuditRequest,
) -> tuple[CorpusAuditConversationSummary, MutableAuditSummary]:
    duration_seconds = conversation.represented_duration_seconds
    summary = MutableAuditSummary(
        conversation_count=1,
        source_duration_seconds=duration_seconds,
        usable_source_duration_seconds=(
            conversation.conversation_regions.usable_duration_seconds
            if conversation.conversation_regions is not None
            else 0.0
        ),
        turn_shift_count=conversation.annotation.turn_taking_count,
        pause_count=conversation.annotation.pause_count,
        backchannel_count=conversation.annotation.backchannel_count,
        interruption_count=conversation.annotation.interruption_count,
        overlap_count=len(
            _overlap_onsets(
                conversation.annotation.speaker1.speech_segments,
                conversation.annotation.speaker2.speech_segments,
            )
        ),
    )
    for user_side in (TrackSide.SPEAKER1, TrackSide.SPEAKER2):
        user, assistant = _oriented_speakers(conversation.annotation, user_side)
        events = _audit_events(user=user, assistant=assistant)
        floor_validity = _floor_validity_index(user)
        for event in events:
            _increment(summary.available_events, event.kind)
        for start_seconds in _window_starts(duration_seconds=duration_seconds, request=request):
            summary.candidate_window_count += 1
            window = _audit_window(
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                floor_validity=floor_validity,
                events=events,
                conversation_regions=conversation.conversation_regions,
                request=request,
            )
            if not window.accepted:
                for reason in window.rejection_reasons:
                    _increment(summary.rejections, reason)
                continue
            summary.accepted_window_count += 1
            summary.input_duration_seconds += window.input_duration_seconds
            summary.supervised_duration_seconds += window.supervised_duration_seconds
            summary.effective_supervised_duration_seconds += (
                window.effective_supervised_duration_seconds
            )
            summary.masked_duration_seconds += window.masked_duration_seconds
            _increment(summary.category_windows, window.category)
            for event in window.covered_events:
                _increment(summary.covered_events, event.kind)
    summary.accepted_conversation_count = int(summary.accepted_window_count > 0)
    conversation_summary = CorpusAuditConversationSummary(
        dataset_id=conversation.dataset_id,
        dataset_name=conversation.dataset_name,
        sample_id=conversation.sample_id,
        external_id=conversation.external_id,
        quality_score=conversation.quality_score,
        source_duration_seconds=duration_seconds,
        usable_source_duration_seconds=(
            conversation.conversation_regions.usable_duration_seconds
            if conversation.conversation_regions is not None
            else 0.0
        ),
        candidate_window_count=summary.candidate_window_count,
        accepted_window_count=summary.accepted_window_count,
        effective_supervised_duration_seconds=summary.effective_supervised_duration_seconds,
        masked_duration_seconds=summary.masked_duration_seconds,
    )
    return conversation_summary, summary


def _merge_summary(
    target: MutableAuditSummary,
    source: MutableAuditSummary,
) -> None:
    target.conversation_count += source.conversation_count
    target.accepted_conversation_count += source.accepted_conversation_count
    target.source_duration_seconds += source.source_duration_seconds
    target.usable_source_duration_seconds += source.usable_source_duration_seconds
    target.candidate_window_count += source.candidate_window_count
    target.accepted_window_count += source.accepted_window_count
    target.input_duration_seconds += source.input_duration_seconds
    target.supervised_duration_seconds += source.supervised_duration_seconds
    target.effective_supervised_duration_seconds += source.effective_supervised_duration_seconds
    target.masked_duration_seconds += source.masked_duration_seconds
    target.turn_shift_count += source.turn_shift_count
    target.pause_count += source.pause_count
    target.backchannel_count += source.backchannel_count
    target.interruption_count += source.interruption_count
    target.overlap_count += source.overlap_count
    for key, count in source.category_windows.items():
        target.category_windows[key] = target.category_windows.get(key, 0) + count
    for key, count in source.available_events.items():
        target.available_events[key] = target.available_events.get(key, 0) + count
    for key, count in source.covered_events.items():
        target.covered_events[key] = target.covered_events.get(key, 0) + count
    for key, count in source.rejections.items():
        target.rejections[key] = target.rejections.get(key, 0) + count


def _audit_window(
    start_seconds: float,
    duration_seconds: float,
    floor_validity: FloorValidityIndex,
    events: tuple[AuditEvent, ...],
    conversation_regions: ConversationRegionAnalysis | None,
    request: CorpusAuditRequest,
) -> WindowAudit:
    end_seconds = min(duration_seconds, start_seconds + request.input_duration_seconds)
    supervision_start_seconds = min(end_seconds, start_seconds + request.burn_in_seconds)
    supervised_duration_seconds = max(0.0, end_seconds - supervision_start_seconds)
    covered_events = tuple(
        event for event in events if supervision_start_seconds <= event.time_seconds < end_seconds
    )
    category = _window_category(covered_events)
    if conversation_regions is None:
        return WindowAudit(
            accepted=False,
            input_duration_seconds=end_seconds - start_seconds,
            supervised_duration_seconds=supervised_duration_seconds,
            effective_supervised_duration_seconds=0.0,
            masked_duration_seconds=0.0,
            category=category,
            covered_events=(),
            rejection_reasons=(CorpusAuditRejectionReason.MISSING_REGION_ANALYSIS,),
        )
    primary_ratio, future_ratio = _supervision_coverage(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        annotation_end_seconds=duration_seconds,
        floor_validity=floor_validity,
    )
    masked_duration_seconds = _masked_duration(
        start_seconds=supervision_start_seconds,
        end_seconds=end_seconds,
        conversation_regions=conversation_regions,
    )
    masked_ratio = _ratio(masked_duration_seconds, supervised_duration_seconds)
    rejection_reasons: list[CorpusAuditRejectionReason] = []
    if supervised_duration_seconds < request.minimum_supervised_seconds:
        rejection_reasons.append(CorpusAuditRejectionReason.INSUFFICIENT_SUPERVISED_DURATION)
    if primary_ratio < request.minimum_primary_supervision_ratio:
        rejection_reasons.append(CorpusAuditRejectionReason.INSUFFICIENT_PRIMARY_SUPERVISION)
    if future_ratio < request.minimum_future_supervision_ratio:
        rejection_reasons.append(CorpusAuditRejectionReason.INSUFFICIENT_FUTURE_SUPERVISION)
    if masked_ratio > request.maximum_masked_ratio:
        rejection_reasons.append(CorpusAuditRejectionReason.EXCESSIVE_MASKING)
    return WindowAudit(
        accepted=not rejection_reasons,
        input_duration_seconds=end_seconds - start_seconds,
        supervised_duration_seconds=supervised_duration_seconds,
        effective_supervised_duration_seconds=max(
            0.0,
            supervised_duration_seconds * primary_ratio - masked_duration_seconds,
        ),
        masked_duration_seconds=masked_duration_seconds,
        category=category,
        covered_events=covered_events if not rejection_reasons else (),
        rejection_reasons=tuple(rejection_reasons),
    )


def _supervision_coverage(
    start_seconds: float,
    end_seconds: float,
    annotation_end_seconds: float,
    floor_validity: FloorValidityIndex,
) -> tuple[float, float]:
    frame_count = max(1, math.ceil((end_seconds - start_seconds) / FRAME_SECONDS))
    supervised_times = tuple(
        min(end_seconds, start_seconds + (frame_index + 0.5) * FRAME_SECONDS)
        for frame_index in range(frame_count)
        if start_seconds + (frame_index + 0.5) * FRAME_SECONDS >= start_seconds + BURN_IN_SECONDS
    )
    primary_valid_count = sum(
        floor_validity.is_valid(time_seconds) for time_seconds in supervised_times
    )
    future_valid_count = sum(
        time_seconds + end_milliseconds / 1000.0 <= annotation_end_seconds
        for time_seconds in supervised_times
        for _, end_milliseconds in FUTURE_ACTIVITY_WINDOWS_MILLISECONDS
    )
    return (
        _ratio(primary_valid_count, len(supervised_times)),
        _ratio(
            future_valid_count,
            len(supervised_times) * len(FUTURE_ACTIVITY_WINDOWS_MILLISECONDS),
        ),
    )


def _floor_validity_index(speaker: SpeakerConversationAnnotation) -> FloorValidityIndex:
    transcript_segments = tuple(
        (segment.start_seconds, segment.end_seconds)
        for segment in speaker.segment_targets
        if segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
    )
    audio_activity = tuple(
        (segment.start_seconds, segment.end_seconds)
        for segment in speaker.segment_targets
        if segment.evidence_source is AnnotationEvidenceSource.AUDIO_ACTIVITY
    )
    valid_connections: list[tuple[float, float]] = []
    invalid_connections: list[tuple[float, float]] = []
    transcript_starts = tuple(start_seconds for start_seconds, _ in transcript_segments)
    transcript_ends = tuple(end_seconds for _, end_seconds in transcript_segments)
    for connection in speaker.connection_targets:
        interval = (connection.earlier_end_seconds, connection.later_start_seconds)
        boundaries_are_valid = any(
            abs(end_seconds - connection.earlier_end_seconds) <= FRAME_SECONDS
            for end_seconds in transcript_ends
        ) and any(
            abs(start_seconds - connection.later_start_seconds) <= FRAME_SECONDS
            for start_seconds in transcript_starts
        )
        if boundaries_are_valid:
            valid_connections.append(interval)
        else:
            invalid_connections.append(interval)
    return FloorValidityIndex(
        transcript_segments=_interval_index(transcript_segments),
        valid_connections=_interval_index(valid_connections),
        invalid_connections=_interval_index(invalid_connections),
        audio_activity=_interval_index(audio_activity),
    )


def _interval_index(intervals: Sequence[tuple[float, float]]) -> IntervalIndex:
    merged: list[tuple[float, float]] = []
    for start_seconds, end_seconds in sorted(intervals):
        if start_seconds >= end_seconds:
            continue
        if merged and start_seconds <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_seconds))
        else:
            merged.append((start_seconds, end_seconds))
    return IntervalIndex(
        starts=tuple(start_seconds for start_seconds, _ in merged),
        ends=tuple(end_seconds for _, end_seconds in merged),
    )


def _window_starts(
    duration_seconds: float,
    request: CorpusAuditRequest,
) -> tuple[float, ...]:
    starts: list[float] = []
    start_seconds = 0.0
    while start_seconds + request.burn_in_seconds < duration_seconds:
        starts.append(start_seconds)
        start_seconds += request.stride_seconds
    return tuple(starts)


def _audit_events(
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
) -> tuple[AuditEvent, ...]:
    return tuple(
        sorted(
            (
                *(
                    AuditEvent(TrainingSamplePropositionKind.TURN_SHIFT, time_seconds)
                    for time_seconds in _tight_shift_times(user, assistant)
                ),
                *(
                    AuditEvent(
                        TrainingSamplePropositionKind.HOLD_PAUSE,
                        (connection.earlier_end_seconds + connection.later_start_seconds) / 2.0,
                    )
                    for connection in user.connection_targets
                    if connection.gap_seconds >= 0.25 and connection.merge_confidence >= 0.5
                ),
                *(
                    AuditEvent(
                        TrainingSamplePropositionKind.NON_FLOOR_FEEDBACK,
                        (span.start_seconds + span.end_seconds) / 2.0,
                    )
                    for span in user.backchannels
                    if _overlap_seconds(span, assistant.speech_segments) > 0.0
                ),
                *(
                    AuditEvent(
                        TrainingSamplePropositionKind.OVERLAP_INTERRUPTION,
                        point.time_seconds,
                    )
                    for point in user.interruptions
                ),
                *(
                    AuditEvent(
                        TrainingSamplePropositionKind.OVERLAP_INTERRUPTION,
                        time_seconds,
                    )
                    for time_seconds in _overlap_onsets(
                        user.speech_segments,
                        assistant.speech_segments,
                    )
                ),
            ),
            key=lambda event: (event.time_seconds, event.kind.value),
        )
    )


def _tight_shift_times(
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
) -> tuple[float, ...]:
    return (
        *_speaker_shift_times(user.segment_targets, assistant.segment_targets),
        *_speaker_shift_times(assistant.segment_targets, user.segment_targets),
    )


def _speaker_shift_times(
    releasing_segments: Sequence[SegmentAnnotationTarget],
    responding_segments: Sequence[SegmentAnnotationTarget],
) -> tuple[float, ...]:
    responses = tuple(
        segment
        for segment in responding_segments
        if segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
    )
    times: list[float] = []
    for releasing_segment in releasing_segments:
        if releasing_segment.evidence_source is not AnnotationEvidenceSource.TRANSCRIPT:
            continue
        response = next(
            (
                segment
                for segment in responses
                if segment.start_seconds >= releasing_segment.end_seconds
            ),
            None,
        )
        if (
            response is not None
            and response.start_seconds - releasing_segment.end_seconds
            <= MAXIMUM_TURN_SHIFT_GAP_SECONDS
        ):
            times.append(releasing_segment.end_seconds)
    return tuple(times)


def _window_category(events: Sequence[AuditEvent]) -> TrainingSamplePropositionKind:
    kinds = {event.kind for event in events}
    for kind in (
        TrainingSamplePropositionKind.OVERLAP_INTERRUPTION,
        TrainingSamplePropositionKind.NON_FLOOR_FEEDBACK,
        TrainingSamplePropositionKind.HOLD_PAUSE,
        TrainingSamplePropositionKind.TURN_SHIFT,
    ):
        if kind in kinds:
            return kind
    return TrainingSamplePropositionKind.BACKGROUND


def _masked_duration(
    start_seconds: float,
    end_seconds: float,
    conversation_regions: ConversationRegionAnalysis,
) -> float:
    return sum(
        max(
            0.0,
            min(end_seconds, region.end_seconds) - max(start_seconds, region.start_seconds),
        )
        for region in conversation_regions.unusable_regions
    )


def _oriented_speakers(
    annotation: ConversationAnnotation,
    user_side: TrackSide,
) -> tuple[SpeakerConversationAnnotation, SpeakerConversationAnnotation]:
    match user_side:
        case TrackSide.SPEAKER1:
            return annotation.speaker1, annotation.speaker2
        case TrackSide.SPEAKER2:
            return annotation.speaker2, annotation.speaker1


def _overlap_seconds(span: AnnotationSpan, candidates: Sequence[AnnotationSpan]) -> float:
    return sum(
        max(
            0.0,
            min(span.end_seconds, candidate.end_seconds)
            - max(span.start_seconds, candidate.start_seconds),
        )
        for candidate in candidates
    )


def _overlap_onsets(
    first_spans: Sequence[AnnotationSpan],
    second_spans: Sequence[AnnotationSpan],
) -> tuple[float, ...]:
    onsets: list[float] = []
    first_index = 0
    second_index = 0
    while first_index < len(first_spans) and second_index < len(second_spans):
        first = first_spans[first_index]
        second = second_spans[second_index]
        overlap_start = max(first.start_seconds, second.start_seconds)
        overlap_end = min(first.end_seconds, second.end_seconds)
        if overlap_end > overlap_start:
            onsets.append(overlap_start)
        if first.end_seconds <= second.end_seconds:
            first_index += 1
        else:
            second_index += 1
    return tuple(onsets)


def _dataset_summary(
    dataset_id: UUID,
    dataset_name: str,
    summary: MutableAuditSummary,
) -> CorpusAuditDatasetSummary:
    return CorpusAuditDatasetSummary(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        conversation_count=summary.conversation_count,
        accepted_conversation_count=summary.accepted_conversation_count,
        source_duration_seconds=summary.source_duration_seconds,
        usable_source_duration_seconds=summary.usable_source_duration_seconds,
        candidate_window_count=summary.candidate_window_count,
        accepted_window_count=summary.accepted_window_count,
        input_duration_seconds=summary.input_duration_seconds,
        supervised_duration_seconds=summary.supervised_duration_seconds,
        effective_supervised_duration_seconds=summary.effective_supervised_duration_seconds,
        masked_duration_seconds=summary.masked_duration_seconds,
        physical_events=_physical_events(summary),
    )


def _physical_events(summary: MutableAuditSummary) -> CorpusAuditPhysicalEventCounts:
    return CorpusAuditPhysicalEventCounts(
        turn_shift_count=summary.turn_shift_count,
        pause_count=summary.pause_count,
        backchannel_count=summary.backchannel_count,
        interruption_count=summary.interruption_count,
        overlap_count=summary.overlap_count,
    )


def _pilot_metrics(
    summary: MutableAuditSummary,
    physical_events: CorpusAuditPhysicalEventCounts,
) -> tuple[CorpusAuditPilotMetric, ...]:
    usable_hours = summary.usable_source_duration_seconds / 3600.0
    interruption_overlaps = physical_events.interruption_count + physical_events.overlap_count
    values = (
        ("Usable source audio", usable_hours, PILOT_HOURS, "hours"),
        ("Turn shifts", physical_events.turn_shift_count, PILOT_TURN_SHIFTS, "events"),
        ("Hold pauses", physical_events.pause_count, PILOT_HOLD_PAUSES, "events"),
        ("Backchannels", physical_events.backchannel_count, PILOT_BACKCHANNELS, "events"),
        (
            "Interruptions and overlaps",
            interruption_overlaps,
            PILOT_INTERRUPTION_OVERLAPS,
            "events",
        ),
    )
    return tuple(
        CorpusAuditPilotMetric(
            label=label,
            current=float(current),
            target=float(target),
            unit=unit,
            ready=current >= target,
        )
        for label, current, target, unit in values
    )


def _increment(counts: dict[CountKey, int], key: CountKey) -> None:
    counts[key] = counts.get(key, 0) + 1


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
