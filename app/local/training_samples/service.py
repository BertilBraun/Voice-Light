from __future__ import annotations

import math
import random
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.local.db.models import DashboardSample, TrackSide
from app.local.training_samples.models import (
    CandidateSource,
    EventTargetDistribution,
    FutureActivityTarget,
    PreviewEventType,
    PreviewPoint,
    PreviewSpan,
    PreviewWaveformPoint,
    ReliabilitySource,
    TrainingFramePreview,
    TrainingSamplePreview,
)
from app.shared.audio.wav import mono_samples
from app.shared.quality import (
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


@dataclass(frozen=True)
class DecisionBoundary:
    speech_offset_seconds: float
    candidate_end_seconds: float
    yield_probability: float | None
    source: CandidateSource
    event_distribution: EventTargetDistribution | None


def build_training_sample_preview(
    dashboard_sample: DashboardSample,
    user_side: TrackSide,
    requested_start_seconds: float | None,
    generator: random.Random,
) -> TrainingSamplePreview:
    annotation = _conversation_annotation(dashboard_sample)
    represented_duration_seconds = (
        dashboard_sample.sample.duration_seconds
        if dashboard_sample.sample.duration_seconds is not None
        else annotation.analyzed_duration_seconds
    )
    eligible_duration_seconds = min(
        represented_duration_seconds, annotation.analyzed_duration_seconds
    )
    start_seconds = _sample_start_seconds(
        duration_seconds=eligible_duration_seconds,
        requested_start_seconds=requested_start_seconds,
        generator=generator,
    )
    end_seconds = min(eligible_duration_seconds, start_seconds + INPUT_DURATION_SECONDS)
    assistant_side = _other_side(user_side)
    user_annotation = _speaker_annotation(annotation, user_side)
    assistant_annotation = _speaker_annotation(annotation, assistant_side)
    user_track_path = _track_path(dashboard_sample, user_side)
    sample_rate, waveform = _waveform_window(
        path=user_track_path,
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
        user_waveform=waveform,
        user_spans=_preview_spans(user_annotation, start_seconds, end_seconds, is_user=True),
        assistant_spans=_preview_spans(
            assistant_annotation, start_seconds, end_seconds, is_user=False
        ),
        user_points=_preview_points(user_annotation, start_seconds, end_seconds, is_user=True),
        assistant_points=_preview_points(
            assistant_annotation, start_seconds, end_seconds, is_user=False
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


def _track_path(dashboard_sample: DashboardSample, side: TrackSide) -> Path:
    track = next(
        (candidate for candidate in dashboard_sample.tracks if candidate.side == side), None
    )
    if track is None:
        raise ValueError(f"The selected sample has no {side.value} track.")
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
    boundaries = _decision_boundaries(
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
        boundary = _active_boundary(time_seconds=time_seconds, boundaries=boundaries)
        candidate = boundary is not None
        primary_valid = candidate and boundary.yield_probability is not None
        reliability_source = ReliabilitySource.UNMEASURED if primary_valid else None
        frames.append(
            TrainingFramePreview(
                frame_index=frame_index,
                time_seconds=time_seconds,
                relative_time_seconds=time_seconds - start_seconds,
                supervised=time_seconds >= start_seconds + BURN_IN_SECONDS,
                assistant_speaking_input=_assistant_speaking_at(
                    time_seconds=time_seconds,
                    assistant=assistant,
                ),
                candidate=candidate,
                candidate_source=boundary.source if boundary is not None else None,
                seconds_since_speech_offset=(
                    time_seconds - boundary.speech_offset_seconds if boundary is not None else None
                ),
                yield_probability=(boundary.yield_probability if boundary is not None else None),
                hold_probability=(
                    1.0 - boundary.yield_probability
                    if boundary is not None and boundary.yield_probability is not None
                    else None
                ),
                primary_reliability=None,
                primary_reliability_source=reliability_source,
                primary_valid=primary_valid,
                event_distribution=(boundary.event_distribution if boundary is not None else None),
                event_reliability=None,
                event_reliability_source=reliability_source,
                event_valid=primary_valid and boundary.event_distribution is not None,
                future_activity=(
                    _future_activity_targets(
                        time_seconds=time_seconds,
                        annotation_end_seconds=annotation_end_seconds,
                        user=user,
                    )
                    if candidate
                    else ()
                ),
            )
        )
    return tuple(frames)


def _assistant_speaking_at(
    time_seconds: float,
    assistant: SpeakerConversationAnnotation,
) -> bool:
    inside_turn = _inside_span(time_seconds, assistant.speech_segments)
    inside_pause = _inside_span(time_seconds, assistant.pauses)
    inside_backchannel = _inside_span(time_seconds, assistant.backchannels)
    return (inside_turn and not inside_pause) or inside_backchannel


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
    boundary_yield_probability: float | None
    source: CandidateSource
    pause_probability = 0.0
    if connection is not None:
        boundary_yield_probability = 1.0 - connection.merge_confidence
        pause_probability = connection.pause_confidence
        source = CandidateSource.CONNECTION
    elif (
        next_assistant is not None
        or segment.end_seconds + MAXIMUM_CANDIDATE_SILENCE_SECONDS <= annotation_end_seconds
    ):
        boundary_yield_probability = 1.0
        source = CandidateSource.SEGMENT_END
    else:
        boundary_yield_probability = None
        source = CandidateSource.CENSORED
    yield_probability = (
        segment.turn_confidence * boundary_yield_probability
        if boundary_yield_probability is not None
        else None
    )
    event_distribution = (
        _event_distribution(
            segment=segment,
            pause_probability=pause_probability,
            boundary_yield_probability=boundary_yield_probability,
        )
        if boundary_yield_probability is not None
        else None
    )
    return DecisionBoundary(
        speech_offset_seconds=segment.end_seconds,
        candidate_end_seconds=max(segment.end_seconds + FRAME_SECONDS, candidate_end_seconds),
        yield_probability=yield_probability,
        source=source,
        event_distribution=event_distribution,
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


def _event_distribution(
    segment: SegmentAnnotationTarget,
    pause_probability: float,
    boundary_yield_probability: float,
) -> EventTargetDistribution:
    backchannel = segment.keep_playing_confidence
    remaining = 1.0 - backchannel
    interruption = min(remaining, segment.interruption_confidence)
    remaining -= interruption
    continuation_pause = remaining * pause_probability
    remaining -= continuation_pause
    turn_completion = remaining * boundary_yield_probability
    other = remaining - turn_completion
    return EventTargetDistribution(
        turn_completion=turn_completion,
        continuation_pause=continuation_pause,
        backchannel=backchannel,
        interruption=interruption,
        other=other,
    )


def _future_activity_targets(
    time_seconds: float,
    annotation_end_seconds: float,
    user: SpeakerConversationAnnotation,
) -> tuple[FutureActivityTarget, ...]:
    activity_spans = (*user.speech_segments, *user.backchannels)
    targets: list[FutureActivityTarget] = []
    for start_milliseconds, end_milliseconds in FUTURE_ACTIVITY_WINDOWS_MILLISECONDS:
        start = time_seconds + start_milliseconds / 1000.0
        end = time_seconds + end_milliseconds / 1000.0
        valid = end <= annotation_end_seconds
        active_fraction = _active_fraction(start, end, activity_spans) if valid else 0.0
        targets.append(
            FutureActivityTarget(
                start_milliseconds=start_milliseconds,
                end_milliseconds=end_milliseconds,
                active=active_fraction >= 0.5 if valid else None,
                valid=valid,
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
    active_seconds = sum(
        max(0.0, min(end_seconds, span.end_seconds) - max(start_seconds, span.start_seconds))
        for span in spans
    )
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
