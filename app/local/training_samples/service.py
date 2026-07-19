from __future__ import annotations

import math
import random
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.training_samples.models import (
    CandidateSource,
    EventTargetDistribution,
    FutureActivityTarget,
    PreviewEventType,
    PreviewPoint,
    PreviewSpan,
    PreviewWaveformPoint,
    SupervisionMaskReason,
    TrainingFramePreview,
    TrainingSamplePreview,
    TrainingSampleQuality,
    TrainingSampleSelectionMode,
)
from app.shared.audio.wav import mono_samples
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationPoint,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    ConversationAnnotation,
    QualityResult,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)

INPUT_DURATION_SECONDS = 20.0
BURN_IN_SECONDS = 4.0
FRAME_SECONDS = 0.08
WAVEFORM_POINT_COUNT = 1000
MAXIMUM_CANDIDATE_SILENCE_SECONDS = 2.0
FUTURE_ACTIVITY_WINDOWS_MILLISECONDS = ((0, 200), (200, 500), (500, 1000), (1000, 1500))
INTERESTING_RANDOM_LOCATION_COUNT = 24
INTERESTING_ANCHOR_LIMIT = 64
HIGH_CONFIDENCE = 0.8
LOW_CONFIDENCE = 0.2
USER_YIELD_HORIZON_SECONDS = 0.5
MAXIMUM_NON_FLOOR_FEEDBACK_SECONDS = 1.2
MINIMUM_ASSISTANT_CONTINUATION_SECONDS = 0.3
MINIMUM_COMPLETION_INACTIVITY_SECONDS = 1.5


@dataclass(frozen=True)
class DecisionBoundary:
    speech_offset_seconds: float
    candidate_end_seconds: float
    source: CandidateSource


@dataclass(frozen=True)
class ScalarSupervision:
    target: float | None
    valid: bool
    mask_reason: SupervisionMaskReason | None


@dataclass(frozen=True)
class InteractionEventAnchor:
    time_seconds: float
    source: CandidateSource
    distribution: EventTargetDistribution | None
    mask_reason: SupervisionMaskReason | None


@dataclass(frozen=True)
class LogicalUserTurn:
    start_seconds: float
    end_seconds: float
    segments: tuple[SegmentAnnotationTarget, ...]


@dataclass(frozen=True)
class ProbabilitySpan:
    start_seconds: float
    end_seconds: float
    yield_probability: float


def build_training_sample_preview(
    dashboard_sample: DashboardSample,
    user_side: TrackSide,
    requested_start_seconds: float | None,
    selection_mode: TrainingSampleSelectionMode,
    generator: random.Random,
) -> TrainingSamplePreview:
    annotation = _conversation_annotation(dashboard_sample)
    quality = dashboard_sample.latest_quality
    assert quality is not None
    represented_duration_seconds = (
        dashboard_sample.sample.duration_seconds
        if dashboard_sample.sample.duration_seconds is not None
        else annotation.analyzed_duration_seconds
    )
    eligible_duration_seconds = min(
        represented_duration_seconds, annotation.analyzed_duration_seconds
    )
    assistant_side = _other_side(user_side)
    user_annotation = _speaker_annotation(annotation, user_side)
    assistant_annotation = _speaker_annotation(annotation, assistant_side)
    start_seconds = _select_start_seconds(
        duration_seconds=eligible_duration_seconds,
        requested_start_seconds=requested_start_seconds,
        selection_mode=selection_mode,
        user=user_annotation,
        assistant=assistant_annotation,
        generator=generator,
    )
    end_seconds = min(eligible_duration_seconds, start_seconds + INPUT_DURATION_SECONDS)
    user_track = _track(dashboard_sample, user_side)
    assistant_track = _track(dashboard_sample, assistant_side)
    user_track_path = _track_path(user_track)
    assistant_track_path = _track_path(assistant_track)
    sample_rate, waveform = _waveform_window(
        path=user_track_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        point_count=WAVEFORM_POINT_COUNT,
    )
    assistant_sample_rate, assistant_waveform = _waveform_window(
        path=assistant_track_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        point_count=WAVEFORM_POINT_COUNT,
    )
    frames = build_frame_previews(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        annotation_end_seconds=annotation.analyzed_duration_seconds,
        user=user_annotation,
        assistant=assistant_annotation,
    )
    return TrainingSamplePreview(
        sample_id=dashboard_sample.sample.id,
        external_id=dashboard_sample.sample.external_id,
        user_side=user_side,
        assistant_side=assistant_side,
        user_audio_sha256=user_track.audio_sha256,
        assistant_audio_sha256=assistant_track.audio_sha256,
        annotation_version=annotation.annotation_version,
        annotation_generated_at=quality.created_at,
        quality_metric_version=quality.metric_version,
        quality=_training_sample_quality(dashboard_sample),
        represented_duration_seconds=represented_duration_seconds,
        annotated_duration_seconds=annotation.analyzed_duration_seconds,
        eligible_duration_seconds=eligible_duration_seconds,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        burn_in_end_seconds=min(end_seconds, start_seconds + BURN_IN_SECONDS),
        input_duration_seconds=end_seconds - start_seconds,
        supervised_duration_seconds=max(0.0, end_seconds - start_seconds - BURN_IN_SECONDS),
        frame_seconds=FRAME_SECONDS,
        waveform_sample_rate=sample_rate,
        assistant_waveform_sample_rate=assistant_sample_rate,
        user_waveform=waveform,
        assistant_waveform=assistant_waveform,
        user_spans=_preview_spans(user_annotation, start_seconds, end_seconds, is_user=True),
        assistant_spans=_preview_spans(
            assistant_annotation, start_seconds, end_seconds, is_user=False
        ),
        user_points=_preview_points(user_annotation, start_seconds, end_seconds, is_user=True),
        assistant_points=_preview_points(
            assistant_annotation, start_seconds, end_seconds, is_user=False
        ),
        user_segment_targets=_preview_segment_targets(
            user_annotation.segment_targets,
            start_seconds,
            end_seconds,
        ),
        assistant_segment_targets=_preview_segment_targets(
            assistant_annotation.segment_targets,
            start_seconds,
            end_seconds,
        ),
        user_connection_targets=_preview_connection_targets(
            user_annotation.connection_targets,
            start_seconds,
            end_seconds,
        ),
        assistant_connection_targets=_preview_connection_targets(
            assistant_annotation.connection_targets,
            start_seconds,
            end_seconds,
        ),
        frames=frames,
    )


def _conversation_annotation(dashboard_sample: DashboardSample) -> ConversationAnnotation:
    quality = dashboard_sample.latest_quality
    if quality is None:
        raise ValueError("The selected sample has no conversation quality analysis.")
    result = QualityResult.model_validate(quality.payload)
    if result.conversation_annotation is None:
        raise ValueError("The selected sample has no conversation annotation.")
    return result.conversation_annotation


def _training_sample_quality(dashboard_sample: DashboardSample) -> TrainingSampleQuality:
    quality = dashboard_sample.latest_quality
    if quality is None:
        raise ValueError("The selected sample has no conversation quality analysis.")
    return TrainingSampleQuality(
        total_score=quality.total_quality_score,
        interaction_density_score=quality.interaction_density_score,
        timing_reliability_score=quality.timing_reliability_score,
        audio_quality_score=quality.audio_quality_score,
        conversation_quality_score=quality.conversation_quality_score,
        usable_event_count=quality.usable_event_count,
        events_per_hour=quality.conversation_events_per_hour,
        flags=quality.flags,
    )


def _select_start_seconds(
    duration_seconds: float,
    requested_start_seconds: float | None,
    selection_mode: TrainingSampleSelectionMode,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    generator: random.Random,
) -> float:
    if requested_start_seconds is not None:
        return _sample_start_seconds(
            duration_seconds=duration_seconds,
            requested_start_seconds=requested_start_seconds,
            generator=generator,
        )
    match selection_mode:
        case TrainingSampleSelectionMode.RANDOM:
            return _sample_start_seconds(
                duration_seconds=duration_seconds,
                requested_start_seconds=None,
                generator=generator,
            )
        case TrainingSampleSelectionMode.INTERESTING:
            return _interesting_start_seconds(
                duration_seconds=duration_seconds,
                user=user,
                assistant=assistant,
                generator=generator,
            )


def _sample_start_seconds(
    duration_seconds: float,
    requested_start_seconds: float | None,
    generator: random.Random,
) -> float:
    maximum_start = max(0.0, duration_seconds - INPUT_DURATION_SECONDS)
    if requested_start_seconds is None:
        return generator.uniform(0.0, maximum_start) if maximum_start > 0.0 else 0.0
    if requested_start_seconds < 0.0 or requested_start_seconds > maximum_start:
        raise ValueError(
            f"start_seconds must be between 0 and {maximum_start:.3f} for this sample."
        )
    return requested_start_seconds


def _interesting_start_seconds(
    duration_seconds: float,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    generator: random.Random,
) -> float:
    maximum_start = max(0.0, duration_seconds - INPUT_DURATION_SECONDS)
    if maximum_start == 0.0:
        return 0.0
    event_times = _activity_event_times((user, assistant))
    probability_spans = _user_probability_spans(user)
    anchors = [
        *event_times,
        *((span.start_seconds + span.end_seconds) / 2.0 for span in probability_spans),
    ]
    if len(anchors) > INTERESTING_ANCHOR_LIMIT:
        anchors = generator.sample(anchors, INTERESTING_ANCHOR_LIMIT)
    candidate_starts = {
        min(maximum_start, max(0.0, anchor - INPUT_DURATION_SECONDS / 2.0)) for anchor in anchors
    }
    candidate_starts.update(
        generator.uniform(0.0, maximum_start) for _ in range(INTERESTING_RANDOM_LOCATION_COUNT)
    )
    ranked_starts = sorted(
        (
            _interesting_location_score(
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                event_times=event_times,
                probability_spans=probability_spans,
            ),
            start_seconds,
        )
        for start_seconds in candidate_starts
    )
    top_starts = ranked_starts[-min(5, len(ranked_starts)) :]
    return top_starts[generator.randrange(len(top_starts))][1]


def _activity_event_times(
    speakers: Sequence[SpeakerConversationAnnotation],
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                time_seconds
                for speaker in speakers
                for time_seconds in (
                    *(segment.end_seconds for segment in speaker.segment_targets),
                    *(span.start_seconds for span in speaker.pauses),
                    *(span.start_seconds for span in speaker.backchannels),
                    *(point.time_seconds for point in speaker.turns),
                    *(point.time_seconds for point in speaker.interruptions),
                )
            }
        )
    )


def _user_probability_spans(
    user: SpeakerConversationAnnotation,
) -> tuple[ProbabilitySpan, ...]:
    return (
        *(
            ProbabilitySpan(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                yield_probability=segment.keep_playing_confidence,
            )
            for segment in user.segment_targets
        ),
        *(
            ProbabilitySpan(
                start_seconds=connection.earlier_end_seconds,
                end_seconds=connection.later_start_seconds,
                yield_probability=1.0 - connection.merge_confidence,
            )
            for connection in user.connection_targets
        ),
    )


def _interesting_location_score(
    start_seconds: float,
    duration_seconds: float,
    event_times: Sequence[float],
    probability_spans: Sequence[ProbabilitySpan],
) -> float:
    supervision_start_seconds = start_seconds + BURN_IN_SECONDS
    end_seconds = min(duration_seconds, start_seconds + INPUT_DURATION_SECONDS)
    event_score = float(
        sum(supervision_start_seconds <= event_time < end_seconds for event_time in event_times)
    )
    ambiguous_duration_seconds = sum(
        max(
            0.0,
            min(end_seconds, span.end_seconds) - max(supervision_start_seconds, span.start_seconds),
        )
        * (1.0 - 2.0 * abs(span.yield_probability - 0.5))
        for span in probability_spans
    )
    ambiguity_score = 4.0 * ambiguous_duration_seconds
    return max(event_score, ambiguity_score) + 0.25 * min(event_score, ambiguity_score)


def _other_side(side: TrackSide) -> TrackSide:
    match side:
        case TrackSide.SPEAKER1:
            return TrackSide.SPEAKER2
        case TrackSide.SPEAKER2:
            return TrackSide.SPEAKER1


def _speaker_annotation(
    annotation: ConversationAnnotation, side: TrackSide
) -> SpeakerConversationAnnotation:
    match side:
        case TrackSide.SPEAKER1:
            expected_side = SpeakerSide.SPEAKER1
            speaker = annotation.speaker1
        case TrackSide.SPEAKER2:
            expected_side = SpeakerSide.SPEAKER2
            speaker = annotation.speaker2
    assert speaker.side == expected_side
    return speaker


def _track(dashboard_sample: DashboardSample, side: TrackSide) -> SampleTrackRecord:
    track = next(
        (candidate for candidate in dashboard_sample.tracks if candidate.side == side), None
    )
    if track is None:
        raise ValueError(f"The selected sample has no {side.value} track.")
    return track


def _track_path(track: SampleTrackRecord) -> Path:
    path = Path(track.access_uri)
    if not path.is_file():
        raise ValueError(f"Audio file does not exist: {path}")
    return path


def build_frame_previews(
    start_seconds: float,
    end_seconds: float,
    annotation_end_seconds: float,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
) -> tuple[TrainingFramePreview, ...]:
    logical_user_turns = _logical_user_turns(user)
    boundaries = _decision_boundaries(
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    )
    event_anchors = _interaction_event_anchors(
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    )
    frame_count = max(1, math.ceil((end_seconds - start_seconds) / FRAME_SECONDS))
    frames: list[TrainingFramePreview] = []
    for frame_index in range(frame_count):
        time_seconds = min(
            end_seconds,
            start_seconds + (frame_index + 0.5) * FRAME_SECONDS,
        )
        supervised = time_seconds >= start_seconds + BURN_IN_SECONDS
        boundary = _active_boundary(time_seconds=time_seconds, boundaries=boundaries)
        event_anchor = _event_anchor_for_frame(
            frame_index=frame_index,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            frame_count=frame_count,
            anchors=event_anchors,
        )
        yield_supervision = _user_yield_supervision(
            time_seconds=time_seconds,
            supervised=supervised,
            annotation_end_seconds=annotation_end_seconds,
            user=user,
            assistant=assistant,
            logical_user_turns=logical_user_turns,
        )
        has_floor_supervision = _user_has_floor_supervision(
            time_seconds=time_seconds,
            supervised=supervised,
            user=user,
            logical_user_turns=logical_user_turns,
        )
        event_distribution, event_valid, event_mask_reason = _interaction_event_supervision(
            supervised=supervised,
            anchor=event_anchor,
        )
        candidate = boundary is not None or event_anchor is not None
        candidate_source = (
            event_anchor.source
            if event_anchor is not None
            else boundary.source
            if boundary is not None
            else None
        )
        anchor_time_seconds = (
            event_anchor.time_seconds
            if event_anchor is not None
            else boundary.speech_offset_seconds
            if boundary is not None
            else None
        )
        frames.append(
            TrainingFramePreview(
                frame_index=frame_index,
                time_seconds=time_seconds,
                relative_time_seconds=time_seconds - start_seconds,
                supervised=supervised,
                assistant_speaking_input=_assistant_speaking_at(
                    time_seconds=time_seconds,
                    assistant=assistant,
                ),
                candidate=candidate,
                candidate_source=candidate_source,
                seconds_since_speech_offset=(
                    time_seconds - anchor_time_seconds if anchor_time_seconds is not None else None
                ),
                user_yield_target=yield_supervision.target,
                user_yield_valid=yield_supervision.valid,
                user_yield_mask_reason=yield_supervision.mask_reason,
                user_has_floor_target=has_floor_supervision.target,
                user_has_floor_valid=has_floor_supervision.valid,
                user_has_floor_mask_reason=has_floor_supervision.mask_reason,
                interaction_event_distribution=event_distribution,
                interaction_event_valid=event_valid,
                interaction_event_mask_reason=event_mask_reason,
                future_activity=_future_activity_targets(
                    time_seconds=time_seconds,
                    supervised=supervised,
                    annotation_end_seconds=annotation_end_seconds,
                    user=user,
                ),
            )
        )
    return tuple(frames)


def _user_yield_supervision(
    time_seconds: float,
    supervised: bool,
    annotation_end_seconds: float,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    logical_user_turns: Sequence[LogicalUserTurn],
) -> ScalarSupervision:
    if not supervised:
        return _masked_scalar(SupervisionMaskReason.BURN_IN)
    if _ambiguous_user_annotation_at(time_seconds=time_seconds, user=user):
        return _masked_scalar(SupervisionMaskReason.AMBIGUOUS_ANNOTATION)
    logical_turn = _logical_user_turn_at(
        time_seconds=time_seconds,
        logical_user_turns=logical_user_turns,
    )
    if logical_turn is None:
        return _masked_scalar(SupervisionMaskReason.USER_DOES_NOT_HOLD_FLOOR)
    if time_seconds + USER_YIELD_HORIZON_SECONDS < logical_turn.end_seconds:
        return _valid_scalar(0.0)
    final_segment = logical_turn.segments[-1]
    if _logical_turn_end_confirmed(
        logical_turn=logical_turn,
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    ):
        return _valid_scalar(1.0)
    if _completion_is_censored(
        segment=final_segment,
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    ):
        return _masked_scalar(SupervisionMaskReason.CENSORED_ANNOTATION)
    return _masked_scalar(SupervisionMaskReason.AMBIGUOUS_ANNOTATION)


def _user_has_floor_supervision(
    time_seconds: float,
    supervised: bool,
    user: SpeakerConversationAnnotation,
    logical_user_turns: Sequence[LogicalUserTurn],
) -> ScalarSupervision:
    if not supervised:
        return _masked_scalar(SupervisionMaskReason.BURN_IN)
    if _ambiguous_user_annotation_at(time_seconds=time_seconds, user=user):
        return _masked_scalar(SupervisionMaskReason.AMBIGUOUS_ANNOTATION)
    logical_turn = _logical_user_turn_at(
        time_seconds=time_seconds,
        logical_user_turns=logical_user_turns,
    )
    return _valid_scalar(1.0 if logical_turn is not None else 0.0)


def _logical_user_turns(
    user: SpeakerConversationAnnotation,
) -> tuple[LogicalUserTurn, ...]:
    substantive_segments = tuple(
        segment
        for segment in sorted(
            user.segment_targets,
            key=lambda candidate: (candidate.start_seconds, candidate.end_seconds),
        )
        if _is_substantive_user_segment(segment)
    )
    logical_turns: list[LogicalUserTurn] = []
    for segment in substantive_segments:
        if logical_turns and _segments_have_confident_connection(
            earlier_segment=logical_turns[-1].segments[-1],
            later_segment=segment,
            connections=user.connection_targets,
        ):
            previous_turn = logical_turns[-1]
            logical_turns[-1] = LogicalUserTurn(
                start_seconds=previous_turn.start_seconds,
                end_seconds=segment.end_seconds,
                segments=(*previous_turn.segments, segment),
            )
            continue
        logical_turns.append(
            LogicalUserTurn(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                segments=(segment,),
            )
        )
    return tuple(logical_turns)


def _is_substantive_user_segment(segment: SegmentAnnotationTarget) -> bool:
    return (
        segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
        and segment.turn_confidence >= HIGH_CONFIDENCE
        and segment.keep_playing_confidence < HIGH_CONFIDENCE
    )


def _segments_have_confident_connection(
    earlier_segment: SegmentAnnotationTarget,
    later_segment: SegmentAnnotationTarget,
    connections: Sequence[ConnectionAnnotationTarget],
) -> bool:
    return any(
        abs(connection.earlier_end_seconds - earlier_segment.end_seconds) <= FRAME_SECONDS
        and abs(connection.later_start_seconds - later_segment.start_seconds) <= FRAME_SECONDS
        and connection.merge_confidence >= HIGH_CONFIDENCE
        for connection in connections
    )


def _logical_user_turn_at(
    time_seconds: float,
    logical_user_turns: Sequence[LogicalUserTurn],
) -> LogicalUserTurn | None:
    return next(
        (
            logical_turn
            for logical_turn in logical_user_turns
            if logical_turn.start_seconds <= time_seconds < logical_turn.end_seconds
        ),
        None,
    )


def _ambiguous_user_annotation_at(
    time_seconds: float,
    user: SpeakerConversationAnnotation,
) -> bool:
    active_segment = next(
        (
            segment
            for segment in user.segment_targets
            if segment.start_seconds <= time_seconds < segment.end_seconds
        ),
        None,
    )
    if active_segment is not None:
        if _is_substantive_user_segment(active_segment):
            return False
        is_non_floor_feedback = (
            active_segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
            and active_segment.keep_playing_confidence >= HIGH_CONFIDENCE
            and active_segment.turn_confidence < HIGH_CONFIDENCE
        )
        return not is_non_floor_feedback
    active_connection = next(
        (
            connection
            for connection in user.connection_targets
            if connection.earlier_end_seconds <= time_seconds < connection.later_start_seconds
        ),
        None,
    )
    return (
        active_connection is not None
        and LOW_CONFIDENCE < active_connection.merge_confidence < HIGH_CONFIDENCE
    )


def _logical_turn_end_confirmed(
    logical_turn: LogicalUserTurn,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> bool:
    return _detector_turn_point_at(
        time_seconds=logical_turn.end_seconds,
        points=user.turns,
    ) or _completion_confirmed(
        segment=logical_turn.segments[-1],
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    )


def _detector_turn_point_at(
    time_seconds: float,
    points: Sequence[AnnotationPoint],
) -> bool:
    return any(
        abs(point.time_seconds - time_seconds) <= FRAME_SECONDS
        and (point.confidence is None or point.confidence >= HIGH_CONFIDENCE)
        for point in points
    )


def _valid_scalar(target: float) -> ScalarSupervision:
    return ScalarSupervision(target=target, valid=True, mask_reason=None)


def _masked_scalar(reason: SupervisionMaskReason) -> ScalarSupervision:
    return ScalarSupervision(target=None, valid=False, mask_reason=reason)


def _assistant_speaking_at(
    time_seconds: float,
    assistant: SpeakerConversationAnnotation,
) -> bool:
    inside_turn = _inside_span(time_seconds, assistant.speech_segments)
    inside_pause = _inside_span(time_seconds, assistant.pauses)
    inside_backchannel = _inside_span(time_seconds, assistant.backchannels)
    return (inside_turn and not inside_pause) or inside_backchannel


def _assistant_substantively_speaking_at(
    time_seconds: float,
    assistant: SpeakerConversationAnnotation,
) -> bool:
    return _assistant_substantive_segment_at(
        time_seconds, assistant
    ) is not None and _assistant_speaking_at(time_seconds, assistant)


def _assistant_substantive_segment_at(
    time_seconds: float,
    assistant: SpeakerConversationAnnotation,
) -> SegmentAnnotationTarget | None:
    return next(
        (
            segment
            for segment in assistant.segment_targets
            if segment.start_seconds <= time_seconds < segment.end_seconds
            and segment.turn_confidence >= HIGH_CONFIDENCE
            and segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
        ),
        None,
    )


def _inside_span(time_seconds: float, spans: Sequence[AnnotationSpan]) -> bool:
    return any(span.start_seconds <= time_seconds < span.end_seconds for span in spans)


def _decision_boundaries(
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> tuple[DecisionBoundary, ...]:
    return tuple(
        _decision_boundary(
            segment=segment,
            user=user,
            assistant=assistant,
            annotation_end_seconds=annotation_end_seconds,
        )
        for segment in user.segment_targets
    )


def _decision_boundary(
    segment: SegmentAnnotationTarget,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> DecisionBoundary:
    connection = _matching_connection(segment=segment, connections=user.connection_targets)
    next_assistant = _next_segment_after(
        time_seconds=segment.end_seconds,
        segments=assistant.segment_targets,
    )
    next_user = _next_segment_after(
        time_seconds=segment.end_seconds,
        segments=user.segment_targets,
    )
    next_activity_seconds = min(
        (
            candidate.start_seconds
            for candidate in (next_user, next_assistant)
            if candidate is not None
        ),
        default=annotation_end_seconds,
    )
    candidate_end_seconds = min(
        annotation_end_seconds,
        segment.end_seconds + MAXIMUM_CANDIDATE_SILENCE_SECONDS,
        next_activity_seconds,
    )
    source: CandidateSource
    if connection is not None:
        source = CandidateSource.CONNECTION
    elif (
        next_assistant is not None
        or segment.end_seconds + MAXIMUM_CANDIDATE_SILENCE_SECONDS <= annotation_end_seconds
    ):
        source = CandidateSource.SEGMENT_END
    else:
        source = CandidateSource.CENSORED
    return DecisionBoundary(
        speech_offset_seconds=segment.end_seconds,
        candidate_end_seconds=max(segment.end_seconds + FRAME_SECONDS, candidate_end_seconds),
        source=source,
    )


def _matching_connection(
    segment: SegmentAnnotationTarget,
    connections: Sequence[ConnectionAnnotationTarget],
) -> ConnectionAnnotationTarget | None:
    return next(
        (
            connection
            for connection in connections
            if abs(connection.earlier_end_seconds - segment.end_seconds) <= FRAME_SECONDS
        ),
        None,
    )


def _next_segment_after(
    time_seconds: float,
    segments: Sequence[SegmentAnnotationTarget],
) -> SegmentAnnotationTarget | None:
    return min(
        (segment for segment in segments if segment.start_seconds >= time_seconds),
        key=lambda segment: segment.start_seconds,
        default=None,
    )


def _active_boundary(
    time_seconds: float, boundaries: Sequence[DecisionBoundary]
) -> DecisionBoundary | None:
    return max(
        (
            boundary
            for boundary in boundaries
            if boundary.speech_offset_seconds <= time_seconds < boundary.candidate_end_seconds
        ),
        key=lambda boundary: boundary.speech_offset_seconds,
        default=None,
    )


def _interaction_event_anchors(
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> tuple[InteractionEventAnchor, ...]:
    anchors: list[InteractionEventAnchor] = []
    for segment in user.segment_targets:
        anchors.append(
            _boundary_event_anchor(
                segment=segment,
                user=user,
                assistant=assistant,
                annotation_end_seconds=annotation_end_seconds,
            )
        )
        assistant_segment = _assistant_substantive_segment_at(segment.start_seconds, assistant)
        if assistant_segment is not None and _assistant_speaking_at(
            segment.start_seconds, assistant
        ):
            distribution = _overlap_event_distribution(
                user_segment=segment,
                assistant_segment=assistant_segment,
            )
            anchors.append(
                InteractionEventAnchor(
                    time_seconds=segment.start_seconds,
                    source=CandidateSource.OVERLAP_ONSET,
                    distribution=distribution,
                    mask_reason=(
                        None
                        if distribution is not None
                        else SupervisionMaskReason.AMBIGUOUS_ANNOTATION
                    ),
                )
            )
    return tuple(anchors)


def _boundary_event_anchor(
    segment: SegmentAnnotationTarget,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> InteractionEventAnchor:
    connection = _matching_connection(segment, user.connection_targets)
    if segment.evidence_source is not AnnotationEvidenceSource.TRANSCRIPT:
        source = (
            CandidateSource.CONNECTION if connection is not None else CandidateSource.SEGMENT_END
        )
        return _masked_boundary_event(segment, source)
    if connection is not None and connection.merge_confidence >= HIGH_CONFIDENCE:
        return InteractionEventAnchor(
            time_seconds=segment.end_seconds,
            source=CandidateSource.CONNECTION,
            distribution=_one_hot_event_distribution(continuation_pause=1.0),
            mask_reason=None,
        )
    if connection is not None and connection.merge_confidence > LOW_CONFIDENCE:
        return _masked_boundary_event(segment, CandidateSource.CONNECTION)
    source = CandidateSource.CONNECTION if connection is not None else CandidateSource.SEGMENT_END
    if segment.turn_confidence >= HIGH_CONFIDENCE and (
        _detector_turn_point_at(time_seconds=segment.end_seconds, points=user.turns)
        or _completion_confirmed(
            segment=segment,
            user=user,
            assistant=assistant,
            annotation_end_seconds=annotation_end_seconds,
        )
    ):
        return InteractionEventAnchor(
            time_seconds=segment.end_seconds,
            source=source,
            distribution=_one_hot_event_distribution(turn_completion=1.0),
            mask_reason=None,
        )
    if _completion_is_censored(
        segment=segment,
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    ):
        return InteractionEventAnchor(
            time_seconds=segment.end_seconds,
            source=CandidateSource.CENSORED,
            distribution=None,
            mask_reason=SupervisionMaskReason.CENSORED_ANNOTATION,
        )
    return _masked_boundary_event(segment, source)


def _masked_boundary_event(
    segment: SegmentAnnotationTarget,
    source: CandidateSource,
) -> InteractionEventAnchor:
    return InteractionEventAnchor(
        time_seconds=segment.end_seconds,
        source=source,
        distribution=None,
        mask_reason=SupervisionMaskReason.AMBIGUOUS_ANNOTATION,
    )


def _overlap_event_distribution(
    user_segment: SegmentAnnotationTarget,
    assistant_segment: SegmentAnnotationTarget,
) -> EventTargetDistribution | None:
    if (
        user_segment.evidence_source is not AnnotationEvidenceSource.TRANSCRIPT
        or assistant_segment.evidence_source is not AnnotationEvidenceSource.TRANSCRIPT
    ):
        return None
    floor_take = (
        user_segment.turn_confidence >= HIGH_CONFIDENCE
        and user_segment.interruption_confidence >= HIGH_CONFIDENCE
        and user_segment.keep_playing_confidence < HIGH_CONFIDENCE
    )
    non_floor_feedback = (
        user_segment.keep_playing_confidence >= HIGH_CONFIDENCE
        and user_segment.end_seconds - user_segment.start_seconds
        <= MAXIMUM_NON_FLOOR_FEEDBACK_SECONDS
        and assistant_segment.start_seconds <= user_segment.start_seconds
        and assistant_segment.end_seconds
        >= user_segment.end_seconds + MINIMUM_ASSISTANT_CONTINUATION_SECONDS
    )
    if floor_take == non_floor_feedback:
        return None
    if floor_take:
        return _one_hot_event_distribution(floor_take=1.0)
    return _one_hot_event_distribution(non_floor_feedback=1.0)


def _one_hot_event_distribution(
    turn_completion: float = 0.0,
    continuation_pause: float = 0.0,
    non_floor_feedback: float = 0.0,
    floor_take: float = 0.0,
) -> EventTargetDistribution:
    return EventTargetDistribution(
        turn_completion=turn_completion,
        continuation_pause=continuation_pause,
        non_floor_feedback=non_floor_feedback,
        floor_take=floor_take,
    )


def _completion_confirmed(
    segment: SegmentAnnotationTarget,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> bool:
    next_user = _next_segment_after(segment.end_seconds, user.segment_targets)
    next_assistant = _next_segment_after(segment.end_seconds, assistant.segment_targets)
    if next_assistant is not None and (
        next_user is None or next_assistant.start_seconds < next_user.start_seconds
    ):
        return True
    inactivity_end_seconds = segment.end_seconds + MINIMUM_COMPLETION_INACTIVITY_SECONDS
    return annotation_end_seconds >= inactivity_end_seconds and (
        next_user is None or next_user.start_seconds >= inactivity_end_seconds
    )


def _completion_is_censored(
    segment: SegmentAnnotationTarget,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> bool:
    next_user = _next_segment_after(segment.end_seconds, user.segment_targets)
    next_assistant = _next_segment_after(segment.end_seconds, assistant.segment_targets)
    return (
        next_user is None
        and next_assistant is None
        and annotation_end_seconds < segment.end_seconds + MINIMUM_COMPLETION_INACTIVITY_SECONDS
    )


def _event_anchor_for_frame(
    frame_index: int,
    start_seconds: float,
    end_seconds: float,
    frame_count: int,
    anchors: Sequence[InteractionEventAnchor],
) -> InteractionEventAnchor | None:
    matching_anchors = tuple(
        anchor
        for anchor in anchors
        if start_seconds <= anchor.time_seconds <= end_seconds
        and _causal_frame_index(anchor.time_seconds, start_seconds, frame_count) == frame_index
    )
    if not matching_anchors:
        return None
    if len(matching_anchors) == 1:
        return matching_anchors[0]
    source = (
        CandidateSource.OVERLAP_ONSET
        if any(anchor.source is CandidateSource.OVERLAP_ONSET for anchor in matching_anchors)
        else matching_anchors[0].source
    )
    return InteractionEventAnchor(
        time_seconds=min(anchor.time_seconds for anchor in matching_anchors),
        source=source,
        distribution=None,
        mask_reason=SupervisionMaskReason.AMBIGUOUS_ANNOTATION,
    )


def _causal_frame_index(
    time_seconds: float,
    start_seconds: float,
    frame_count: int,
) -> int:
    first_frame_time_seconds = start_seconds + FRAME_SECONDS / 2.0
    unbounded_index = math.ceil((time_seconds - first_frame_time_seconds) / FRAME_SECONDS - 1e-9)
    return min(frame_count - 1, max(0, unbounded_index))


def _interaction_event_supervision(
    supervised: bool,
    anchor: InteractionEventAnchor | None,
) -> tuple[EventTargetDistribution | None, bool, SupervisionMaskReason | None]:
    if not supervised:
        return None, False, SupervisionMaskReason.BURN_IN
    if anchor is None:
        return None, False, SupervisionMaskReason.NO_EVENT_ANCHOR
    if anchor.distribution is None:
        assert anchor.mask_reason is not None
        return None, False, anchor.mask_reason
    return anchor.distribution, True, None


def _future_activity_targets(
    time_seconds: float,
    supervised: bool,
    annotation_end_seconds: float,
    user: SpeakerConversationAnnotation,
) -> tuple[FutureActivityTarget, ...]:
    activity_spans = (*user.speech_segments, *user.backchannels)
    targets: list[FutureActivityTarget] = []
    for start_milliseconds, end_milliseconds in FUTURE_ACTIVITY_WINDOWS_MILLISECONDS:
        start = time_seconds + start_milliseconds / 1000.0
        end = time_seconds + end_milliseconds / 1000.0
        horizon_available = end <= annotation_end_seconds
        valid = supervised and horizon_available
        mask_reason: SupervisionMaskReason | None
        if not supervised:
            mask_reason = SupervisionMaskReason.BURN_IN
        elif not horizon_available:
            mask_reason = SupervisionMaskReason.FUTURE_HORIZON_CENSORED
        else:
            mask_reason = None
        targets.append(
            FutureActivityTarget(
                start_milliseconds=start_milliseconds,
                end_milliseconds=end_milliseconds,
                occupancy=_active_fraction(start, end, activity_spans) if valid else None,
                valid=valid,
                mask_reason=mask_reason,
            )
        )
    return tuple(targets)


def _active_fraction(
    start_seconds: float,
    end_seconds: float,
    spans: Sequence[AnnotationSpan],
) -> float:
    duration_seconds = end_seconds - start_seconds
    assert duration_seconds > 0.0
    clipped_intervals = sorted(
        (
            max(start_seconds, span.start_seconds),
            min(end_seconds, span.end_seconds),
        )
        for span in spans
        if span.end_seconds > start_seconds and span.start_seconds < end_seconds
    )
    active_seconds = 0.0
    merged_end_seconds = start_seconds
    for interval_start_seconds, interval_end_seconds in clipped_intervals:
        if interval_end_seconds <= merged_end_seconds:
            continue
        active_seconds += interval_end_seconds - max(interval_start_seconds, merged_end_seconds)
        merged_end_seconds = interval_end_seconds
    return min(1.0, active_seconds / duration_seconds)


def _preview_spans(
    annotation: SpeakerConversationAnnotation,
    start_seconds: float,
    end_seconds: float,
    is_user: bool,
) -> tuple[PreviewSpan, ...]:
    speech_type = PreviewEventType.USER_SPEECH if is_user else PreviewEventType.ASSISTANT_SPEECH
    pause_type = PreviewEventType.USER_PAUSE if is_user else PreviewEventType.ASSISTANT_PAUSE
    backchannel_type = (
        PreviewEventType.USER_BACKCHANNEL if is_user else PreviewEventType.ASSISTANT_BACKCHANNEL
    )
    spans = [
        *_spans_of_type(annotation.speech_segments, speech_type),
        *_spans_of_type(annotation.pauses, pause_type),
        *_spans_of_type(annotation.backchannels, backchannel_type),
    ]
    return tuple(
        span
        for span in sorted(
            spans, key=lambda candidate: (candidate.start_seconds, candidate.end_seconds)
        )
        if span.end_seconds >= start_seconds and span.start_seconds <= end_seconds
    )


def _spans_of_type(
    spans: Sequence[AnnotationSpan], event_type: PreviewEventType
) -> list[PreviewSpan]:
    return [
        PreviewSpan(
            event_type=event_type,
            start_seconds=span.start_seconds,
            end_seconds=span.end_seconds,
            text=span.text,
        )
        for span in spans
    ]


def _preview_points(
    annotation: SpeakerConversationAnnotation,
    start_seconds: float,
    end_seconds: float,
    is_user: bool,
) -> tuple[PreviewPoint, ...]:
    turn_type = PreviewEventType.USER_END_OF_TURN
    interruption_type = PreviewEventType.USER_INTERRUPTION
    if not is_user:
        turn_type = PreviewEventType.ASSISTANT_END_OF_TURN
        interruption_type = PreviewEventType.ASSISTANT_INTERRUPTION
    points = [
        *(_points_of_type(annotation.turns, turn_type)),
        *(_points_of_type(annotation.interruptions, interruption_type)),
    ]
    return tuple(
        point
        for point in sorted(points, key=lambda candidate: candidate.time_seconds)
        if start_seconds <= point.time_seconds <= end_seconds
    )


def _points_of_type(
    points: Sequence[AnnotationPoint], event_type: PreviewEventType
) -> list[PreviewPoint]:
    return [
        PreviewPoint(
            event_type=event_type,
            time_seconds=point.time_seconds,
            confidence=point.confidence,
            text=point.text,
        )
        for point in points
    ]


def _preview_segment_targets(
    targets: Sequence[SegmentAnnotationTarget],
    start_seconds: float,
    end_seconds: float,
) -> tuple[SegmentAnnotationTarget, ...]:
    return tuple(
        target
        for target in targets
        if target.end_seconds >= start_seconds and target.start_seconds <= end_seconds
    )


def _preview_connection_targets(
    targets: Sequence[ConnectionAnnotationTarget],
    start_seconds: float,
    end_seconds: float,
) -> tuple[ConnectionAnnotationTarget, ...]:
    return tuple(
        target
        for target in targets
        if target.later_start_seconds >= start_seconds and target.earlier_end_seconds <= end_seconds
    )


def _waveform_window(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    point_count: int,
) -> tuple[int, tuple[PreviewWaveformPoint, ...]]:
    try:
        with wave.open(str(path), "rb") as wave_reader:
            sample_rate = wave_reader.getframerate()
            sample_width = wave_reader.getsampwidth()
            channel_count = wave_reader.getnchannels()
            start_frame = round(start_seconds * sample_rate)
            frame_count = max(0, round((end_seconds - start_seconds) * sample_rate))
            wave_reader.setpos(min(start_frame, wave_reader.getnframes()))
            fragment = wave_reader.readframes(frame_count)
    except (OSError, wave.Error) as error:
        raise ValueError(f"Could not read WAV file {path}: {error}") from error
    if not fragment:
        return sample_rate, ()
    samples = mono_samples(
        fragment=fragment,
        sample_width=sample_width,
        channel_count=channel_count,
    )
    frames_per_point = max(1, math.ceil(len(samples) / point_count))
    maximum_amplitude = float(1 << (sample_width * 8 - 1))
    points: list[PreviewWaveformPoint] = []
    for point_start in range(0, len(samples), frames_per_point):
        point_samples = samples[point_start : point_start + frames_per_point]
        points.append(
            PreviewWaveformPoint(
                minimum_amplitude=max(-1.0, float(np.min(point_samples)) / maximum_amplitude),
                maximum_amplitude=min(1.0, float(np.max(point_samples)) / maximum_amplitude),
            )
        )
    return sample_rate, tuple(points)
