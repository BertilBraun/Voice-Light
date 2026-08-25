from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID, uuid5

from app.local.db.models import TrackSide
from app.local.synthetic_generation.models import (
    BackchannelEvent,
    InterruptionEvent,
    InterruptionOutcome,
    PauseEvent,
    SpeakerRole,
    SpeechEvent,
    SyntheticConversationPlan,
    TurnBoundary,
)
from app.local.training_corpus.export import (
    FRAMES_PER_SAMPLE,
    SCHEMA_VERSION,
    MaterializedTrainingSample,
)
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.local.training_samples.service import (
    FRAME_SECONDS,
    INPUT_DURATION_SECONDS,
    TRAINING_LABEL_VERSION,
)

SYNTHETIC_DATASET_ID = UUID("ca90cb01-b22e-54d8-95cb-167594738e12")
FUTURE_WINDOWS_SECONDS = ((0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5))


def materialize_training_sample(
    plan: SyntheticConversationPlan,
    user_audio_path: Path,
    assistant_audio_path: Path,
) -> MaterializedTrainingSample:
    if plan.duration_seconds != INPUT_DURATION_SECONDS:
        raise ValueError(
            f"Synthetic plans must be exactly {INPUT_DURATION_SECONDS:.1f} seconds for "
            "the current training contract."
        )
    user = plan.speaker_for_role(SpeakerRole.USER)
    assistant = plan.speaker_for_role(SpeakerRole.ASSISTANT)
    frame_times = tuple(index * FRAME_SECONDS for index in range(FRAMES_PER_SAMPLE))
    user_floor_intervals = _floor_intervals(plan, user.speaker_id)
    assistant_floor_intervals = _floor_intervals(plan, assistant.speaker_id)
    completion_by_frame: dict[int, TurnBoundary] = {}
    for event in plan.events:
        if (
            isinstance(event, SpeechEvent)
            and event.speaker_id == user.speaker_id
            and event.boundary_after is not TurnBoundary.NONE
        ):
            frame_index = _boundary_frame(event.end_seconds)
            if frame_index in completion_by_frame:
                raise ValueError(f"Multiple user boundaries map to frame {frame_index}.")
            completion_by_frame[frame_index] = event.boundary_after
    backchannel_frames = {
        _boundary_frame(event.start_seconds)
        for event in plan.events
        if isinstance(event, BackchannelEvent) and event.speaker_id == assistant.speaker_id
    }
    floor_take_frames = {
        _boundary_frame(event.start_seconds)
        for event in plan.events
        if event.speaker_id == assistant.speaker_id and _event_takes_floor(event)
    }
    turn_completion = tuple(
        _boundary_target(index, completion_by_frame, TurnBoundary.COMPLETION)
        for index in range(FRAMES_PER_SAMPLE)
    )
    continuation_pause = tuple(
        _boundary_target(index, completion_by_frame, TurnBoundary.CONTINUATION)
        for index in range(FRAMES_PER_SAMPLE)
    )
    future_activity = tuple(
        tuple(
            _interval_occupancy(
                frame_time + window_start,
                frame_time + window_end,
                user_floor_intervals,
            )
            for frame_time in frame_times
        )
        for window_start, window_end in FUTURE_WINDOWS_SECONDS
    )
    return MaterializedTrainingSample(
        schema_version=SCHEMA_VERSION,
        training_label_version=TRAINING_LABEL_VERSION,
        window_id=hashlib.sha256(f"synthetic:{plan.plan_id}".encode()).hexdigest(),
        dataset_id=SYNTHETIC_DATASET_ID,
        dataset_name="voice_light_synthetic_v1",
        sample_id=uuid5(SYNTHETIC_DATASET_ID, plan.plan_id),
        external_id=plan.plan_id,
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        split=TrainingCorpusSplit.TRAIN,
        user_audio_path=user_audio_path.as_posix(),
        assistant_audio_path=assistant_audio_path.as_posix(),
        start_seconds=0.0,
        end_seconds=INPUT_DURATION_SECONDS,
        quality_score=1.0,
        category="synthetic_planned",
        assistant_has_floor=tuple(
            _active_at(frame_time, assistant_floor_intervals) for frame_time in frame_times
        ),
        p_user_has_floor=tuple(
            _active_at(frame_time, user_floor_intervals) for frame_time in frame_times
        ),
        p_user_yield=tuple(
            _boundary_target(index, completion_by_frame, TurnBoundary.COMPLETION)
            for index in range(FRAMES_PER_SAMPLE)
        ),
        p_assistant_backchannel=tuple(
            1.0 if index in backchannel_frames else 0.0 for index in range(FRAMES_PER_SAMPLE)
        ),
        future_activity_0_200=future_activity[0],
        future_activity_200_500=future_activity[1],
        future_activity_500_1000=future_activity[2],
        future_activity_1000_1500=future_activity[3],
        turn_completion=turn_completion,
        continuation_pause=continuation_pause,
        non_floor_feedback=tuple(
            1.0 if index in backchannel_frames else -1.0 for index in range(FRAMES_PER_SAMPLE)
        ),
        floor_take=tuple(
            1.0 if index in floor_take_frames else -1.0 for index in range(FRAMES_PER_SAMPLE)
        ),
    )


def _floor_intervals(
    plan: SyntheticConversationPlan, speaker_id: str
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (event.start_seconds, event.end_seconds)
        for event in plan.events
        if event.speaker_id == speaker_id and _event_takes_floor(event)
    )


def _event_takes_floor(
    event: SpeechEvent | PauseEvent | BackchannelEvent | InterruptionEvent,
) -> bool:
    match event:
        case SpeechEvent():
            return True
        case InterruptionEvent(outcome=InterruptionOutcome.FLOOR_TAKE):
            return True
        case PauseEvent() | BackchannelEvent() | InterruptionEvent():
            return False


def _boundary_frame(time_seconds: float) -> int:
    frame_index = round(time_seconds / FRAME_SECONDS)
    if not 0 <= frame_index < FRAMES_PER_SAMPLE:
        raise ValueError(f"Boundary at {time_seconds:.3f}s falls outside supervised frames.")
    return frame_index


def _boundary_target(
    frame_index: int,
    boundaries: dict[int, TurnBoundary],
    positive_boundary: TurnBoundary,
) -> float:
    boundary = boundaries.get(frame_index)
    if boundary is None:
        return -1.0
    return 1.0 if boundary is positive_boundary else 0.0


def _active_at(time_seconds: float, intervals: tuple[tuple[float, float], ...]) -> float:
    return float(any(start <= time_seconds < end for start, end in intervals))


def _interval_occupancy(
    start_seconds: float,
    end_seconds: float,
    intervals: tuple[tuple[float, float], ...],
) -> float:
    if end_seconds > INPUT_DURATION_SECONDS:
        return -1.0
    active_seconds = sum(
        max(0.0, min(end_seconds, interval_end) - max(start_seconds, interval_start))
        for interval_start, interval_end in intervals
    )
    return min(1.0, active_seconds / (end_seconds - start_seconds))
