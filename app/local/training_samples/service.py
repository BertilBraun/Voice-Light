from __future__ import annotations

import math
import random
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.local.db.models import DashboardSample, TrackSide
from app.local.training_samples.models import (
    AgentActionLabel,
    PreviewEventType,
    PreviewPoint,
    PreviewSpan,
    PreviewWaveformPoint,
    TrainingFramePreview,
    TrainingSamplePreview,
)
from app.shared.audio.wav import mono_samples
from app.shared.quality import (
    AnnotationPoint,
    AnnotationSpan,
    ConversationAnnotation,
    QualityResult,
    SpeakerConversationAnnotation,
    SpeakerSide,
)

INPUT_DURATION_SECONDS = 20.0
BURN_IN_SECONDS = 4.0
FRAME_SECONDS = 0.08
WAVEFORM_POINT_COUNT = 1000


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
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
) -> tuple[TrainingFramePreview, ...]:
    frame_count = max(1, math.ceil((end_seconds - start_seconds) / FRAME_SECONDS))
    frames: list[TrainingFramePreview] = []
    for frame_index in range(frame_count):
        time_seconds = min(
            end_seconds,
            start_seconds + (frame_index + 0.5) * FRAME_SECONDS,
        )
        user_turn_active = _inside_span(time_seconds, user.speech_segments)
        user_pause = _inside_span(time_seconds, user.pauses)
        user_backchannel = _inside_span(time_seconds, user.backchannels)
        assistant_turn_active = _inside_span(time_seconds, assistant.speech_segments)
        assistant_pause = _inside_span(time_seconds, assistant.pauses)
        assistant_backchannel = _inside_span(time_seconds, assistant.backchannels)
        assistant_speech_active = (
            assistant_turn_active and not assistant_pause
        ) or assistant_backchannel
        should_speak = assistant_turn_active or assistant_backchannel
        frames.append(
            TrainingFramePreview(
                frame_index=frame_index,
                time_seconds=time_seconds,
                relative_time_seconds=time_seconds - start_seconds,
                supervised=time_seconds >= start_seconds + BURN_IN_SECONDS,
                agent_action=(AgentActionLabel.SPEAK if should_speak else AgentActionLabel.LISTEN),
                assistant_playback_active=assistant_speech_active,
                user_turn_active=user_turn_active,
                user_speech_active=(user_turn_active and not user_pause) or user_backchannel,
                assistant_turn_active=assistant_turn_active,
                assistant_speech_active=assistant_speech_active,
                user_pause=user_pause,
                user_end_of_turn=_point_at(time_seconds, user.turns, FRAME_SECONDS / 2.0),
                user_end_within_0_5_seconds=_point_within_future(time_seconds, user.turns, 0.5),
                user_end_within_1_second=_point_within_future(time_seconds, user.turns, 1.0),
                user_end_within_2_seconds=_point_within_future(time_seconds, user.turns, 2.0),
                user_backchannel=user_backchannel,
                user_interruption=_point_at(time_seconds, user.interruptions, FRAME_SECONDS / 2.0),
                assistant_backchannel=assistant_backchannel,
            )
        )
    return tuple(frames)


def _inside_span(time_seconds: float, spans: Sequence[AnnotationSpan]) -> bool:
    return any(span.start_seconds <= time_seconds < span.end_seconds for span in spans)


def _point_at(
    time_seconds: float, points: Sequence[AnnotationPoint], tolerance_seconds: float
) -> bool:
    return any(abs(point.time_seconds - time_seconds) <= tolerance_seconds for point in points)


def _point_within_future(
    time_seconds: float, points: Sequence[AnnotationPoint], horizon_seconds: float
) -> bool:
    return any(0.0 <= point.time_seconds - time_seconds <= horizon_seconds for point in points)


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
