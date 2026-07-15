from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from app.local.analyses.end_of_turn.conversation_scoring import ConversationScoringConfig
from app.local.backchannel_review.models import (
    BackchannelReviewCandidate,
    ReviewWindowAnnotation,
)
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationPoint,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    ConversationAnnotation,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
)

CLIP_DURATION_SECONDS = 10.0
TIMESTAMP_TOLERANCE_SECONDS = 0.001
MERGE_THRESHOLD = ConversationScoringConfig().merge_threshold
MAXIMUM_RESPONSE_DURATION_SECONDS = 2.0
MAXIMUM_RESPONSE_WORDS = 4


def find_ambiguous_backchannel_candidates(
    sample_id: UUID,
    external_id: str,
    annotation: ConversationAnnotation,
) -> tuple[BackchannelReviewCandidate, ...]:
    candidates = [
        *_candidates_for_floor_holder(
            sample_id=sample_id,
            external_id=external_id,
            floor_holder=annotation.speaker1,
            possible_backchannel_speaker=annotation.speaker2,
            annotation_duration_seconds=annotation.analyzed_duration_seconds,
            speaker1=annotation.speaker1,
            speaker2=annotation.speaker2,
        ),
        *_candidates_for_floor_holder(
            sample_id=sample_id,
            external_id=external_id,
            floor_holder=annotation.speaker2,
            possible_backchannel_speaker=annotation.speaker1,
            annotation_duration_seconds=annotation.analyzed_duration_seconds,
            speaker1=annotation.speaker1,
            speaker2=annotation.speaker2,
        ),
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.possible_backchannel.start_seconds,
                candidate.floor_holder_side.value,
            ),
        )
    )


def _candidates_for_floor_holder(
    sample_id: UUID,
    external_id: str,
    floor_holder: SpeakerConversationAnnotation,
    possible_backchannel_speaker: SpeakerConversationAnnotation,
    annotation_duration_seconds: float,
    speaker1: SpeakerConversationAnnotation,
    speaker2: SpeakerConversationAnnotation,
) -> list[BackchannelReviewCandidate]:
    transcript_segments = tuple(
        segment
        for segment in floor_holder.segment_targets
        if segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
    )
    possible_backchannels = tuple(
        segment
        for segment in possible_backchannel_speaker.segment_targets
        if segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
    )
    candidates: list[BackchannelReviewCandidate] = []
    for connection in floor_holder.connection_targets:
        if connection.merge_confidence < MERGE_THRESHOLD:
            continue
        before = _segment_ending_at(connection.earlier_end_seconds, transcript_segments)
        after = _segment_starting_at(connection.later_start_seconds, transcript_segments)
        if before is None or after is None:
            continue
        for possible_backchannel in possible_backchannels:
            if not _overlaps_connection(possible_backchannel, connection) or not _is_short_response(
                possible_backchannel
            ):
                continue
            window_start_seconds, window_end_seconds = _clip_window(
                focus_start_seconds=possible_backchannel.start_seconds,
                focus_end_seconds=possible_backchannel.end_seconds,
                annotation_duration_seconds=annotation_duration_seconds,
            )
            candidates.append(
                BackchannelReviewCandidate(
                    sample_id=sample_id,
                    external_id=external_id,
                    floor_holder_side=floor_holder.side,
                    possible_backchannel_side=possible_backchannel_speaker.side,
                    window_start_seconds=window_start_seconds,
                    window_end_seconds=window_end_seconds,
                    floor_holder_before=before,
                    possible_backchannel=possible_backchannel,
                    floor_holder_after=after,
                    floor_holder_connection=connection,
                    speaker1=_window_annotation(speaker1, window_start_seconds, window_end_seconds),
                    speaker2=_window_annotation(speaker2, window_start_seconds, window_end_seconds),
                )
            )
    return candidates


def _segment_ending_at(
    time_seconds: float,
    segments: Sequence[SegmentAnnotationTarget],
) -> SegmentAnnotationTarget | None:
    return next(
        (
            segment
            for segment in segments
            if abs(segment.end_seconds - time_seconds) <= TIMESTAMP_TOLERANCE_SECONDS
        ),
        None,
    )


def _segment_starting_at(
    time_seconds: float,
    segments: Sequence[SegmentAnnotationTarget],
) -> SegmentAnnotationTarget | None:
    return next(
        (
            segment
            for segment in segments
            if abs(segment.start_seconds - time_seconds) <= TIMESTAMP_TOLERANCE_SECONDS
        ),
        None,
    )


def _overlaps_connection(
    segment: SegmentAnnotationTarget,
    connection: ConnectionAnnotationTarget,
) -> bool:
    return min(segment.end_seconds, connection.later_start_seconds) > max(
        segment.start_seconds, connection.earlier_end_seconds
    )


def _is_short_response(segment: SegmentAnnotationTarget) -> bool:
    duration_seconds = segment.end_seconds - segment.start_seconds
    word_count = len(segment.text.split())
    return (
        duration_seconds <= MAXIMUM_RESPONSE_DURATION_SECONDS
        and word_count <= MAXIMUM_RESPONSE_WORDS
    )


def _clip_window(
    focus_start_seconds: float,
    focus_end_seconds: float,
    annotation_duration_seconds: float,
) -> tuple[float, float]:
    focus_center_seconds = (focus_start_seconds + focus_end_seconds) / 2.0
    start_seconds = max(0.0, focus_center_seconds - CLIP_DURATION_SECONDS / 2.0)
    end_seconds = min(annotation_duration_seconds, start_seconds + CLIP_DURATION_SECONDS)
    start_seconds = max(0.0, end_seconds - CLIP_DURATION_SECONDS)
    return start_seconds, end_seconds


def _window_annotation(
    annotation: SpeakerConversationAnnotation,
    start_seconds: float,
    end_seconds: float,
) -> ReviewWindowAnnotation:
    return ReviewWindowAnnotation(
        side=annotation.side,
        speech_segments=_overlapping_spans(annotation.speech_segments, start_seconds, end_seconds),
        pauses=_overlapping_spans(annotation.pauses, start_seconds, end_seconds),
        backchannels=_overlapping_spans(annotation.backchannels, start_seconds, end_seconds),
        turns=_points_in_window(annotation.turns, start_seconds, end_seconds),
        interruptions=_points_in_window(annotation.interruptions, start_seconds, end_seconds),
        segment_targets=tuple(
            segment
            for segment in annotation.segment_targets
            if segment.end_seconds >= start_seconds and segment.start_seconds <= end_seconds
        ),
        connection_targets=tuple(
            connection
            for connection in annotation.connection_targets
            if connection.later_start_seconds >= start_seconds
            and connection.earlier_end_seconds <= end_seconds
        ),
    )


def _overlapping_spans(
    spans: Sequence[AnnotationSpan],
    start_seconds: float,
    end_seconds: float,
) -> tuple[AnnotationSpan, ...]:
    return tuple(
        span
        for span in spans
        if span.end_seconds >= start_seconds and span.start_seconds <= end_seconds
    )


def _points_in_window(
    points: Sequence[AnnotationPoint],
    start_seconds: float,
    end_seconds: float,
) -> tuple[AnnotationPoint, ...]:
    return tuple(point for point in points if start_seconds <= point.time_seconds <= end_seconds)
