from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.local.training_corpus.export import MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_inventory import (
    DEFAULT_FRAME_SECONDS,
    OBSERVATION_TOLERANCE,
)
from app.training.turn_taking.benchmark_models import (
    CompletionBoundaryKind,
    CompletionTargetPoint,
    TurnCompletionCandidate,
    TurnCompletionInventory,
    TurnCompletionInventoryManifest,
    turn_completion_candidate_rows_sha256,
)

DEFAULT_USER_FLOOR_THRESHOLD = 0.5
DEFAULT_ASSISTANT_ACTIVE_THRESHOLD = 0.5
DEFAULT_CAUSAL_HORIZON_SECONDS = 2.0


@dataclass(frozen=True)
class _StreamKey:
    dataset_id: str
    sample_id: str
    user_side: str
    user_audio_path: str


@dataclass
class _FrameObservation:
    assistant_has_floor: float
    user_has_floor: float | None
    turn_completion: float | None
    continuation_pause: float | None
    categories: set[str] = field(default_factory=set)
    source_window_ids: set[str] = field(default_factory=set)


@dataclass
class _Stream:
    dataset_name: str
    external_id: str
    observations: dict[int, _FrameObservation] = field(default_factory=dict)


def build_turn_completion_inventory(
    samples: Iterable[MaterializedTrainingSample],
    corpus_repository: str,
    corpus_revision: str,
    split: TrainingCorpusSplit,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
    user_floor_threshold: float = DEFAULT_USER_FLOOR_THRESHOLD,
    assistant_active_threshold: float = DEFAULT_ASSISTANT_ACTIVE_THRESHOLD,
    causal_horizon_seconds: float = DEFAULT_CAUSAL_HORIZON_SECONDS,
) -> TurnCompletionInventory:
    if frame_seconds <= 0.0:
        raise ValueError("frame_seconds must be positive.")
    if not 0.0 <= user_floor_threshold <= 1.0:
        raise ValueError("user_floor_threshold must be between zero and one.")
    if not 0.0 <= assistant_active_threshold <= 1.0:
        raise ValueError("assistant_active_threshold must be between zero and one.")
    horizon_frames = round(causal_horizon_seconds / frame_seconds)
    if horizon_frames <= 0 or not math.isclose(
        horizon_frames * frame_seconds,
        causal_horizon_seconds,
        abs_tol=OBSERVATION_TOLERANCE,
    ):
        raise ValueError("causal_horizon_seconds must be a positive whole number of frames.")
    materialized_samples = tuple(samples)
    if not materialized_samples:
        raise ValueError("Turn-completion inventory requires at least one sample.")

    streams: dict[_StreamKey, _Stream] = {}
    for sample in materialized_samples:
        if sample.split is not split:
            raise ValueError(
                f"Sample {sample.window_id} belongs to {sample.split.value}, not {split.value}."
            )
        _add_sample_observations(streams, sample, frame_seconds)

    candidates = tuple(
        candidate
        for stream_key in sorted(streams, key=_stream_sort_key)
        for candidate in _stream_candidates(
            stream_key=stream_key,
            stream=streams[stream_key],
            frame_seconds=frame_seconds,
            user_floor_threshold=user_floor_threshold,
            assistant_active_threshold=assistant_active_threshold,
            horizon_frames=horizon_frames,
        )
    )
    manifest = TurnCompletionInventoryManifest(
        corpus_repository=corpus_repository,
        corpus_revision=corpus_revision,
        split=split,
        frame_seconds=frame_seconds,
        user_floor_threshold=user_floor_threshold,
        assistant_active_threshold=assistant_active_threshold,
        causal_horizon_seconds=causal_horizon_seconds,
        sample_window_count=len(materialized_samples),
        conversation_count=len({(key.dataset_id, key.sample_id) for key in streams}),
        candidate_count=len(candidates),
        candidate_sha256=turn_completion_candidate_rows_sha256(candidates),
    )
    return TurnCompletionInventory(manifest=manifest, candidates=candidates)


def _add_sample_observations(
    streams: dict[_StreamKey, _Stream],
    sample: MaterializedTrainingSample,
    frame_seconds: float,
) -> None:
    start_frame = round(sample.start_seconds / frame_seconds)
    if not math.isclose(start_frame * frame_seconds, sample.start_seconds, abs_tol=1e-6):
        raise ValueError(f"Sample {sample.window_id} is not aligned to the benchmark frame grid.")
    stream_key = _StreamKey(
        dataset_id=str(sample.dataset_id),
        sample_id=str(sample.sample_id),
        user_side=sample.user_side.value,
        user_audio_path=sample.user_audio_path,
    )
    stream = streams.setdefault(
        stream_key,
        _Stream(dataset_name=sample.dataset_name, external_id=sample.external_id),
    )
    if stream.dataset_name != sample.dataset_name or stream.external_id != sample.external_id:
        raise ValueError("Stream provenance differs across overlapping windows.")
    observations = zip(
        sample.assistant_has_floor,
        sample.p_user_has_floor,
        sample.turn_completion,
        sample.continuation_pause,
        strict=True,
    )
    for offset, (assistant_floor, user_floor, completion, continuation) in enumerate(observations):
        frame_index = start_frame + offset
        valid_user_floor = user_floor if user_floor >= 0.0 else None
        valid_completion = completion if completion >= 0.0 else None
        valid_continuation = continuation if continuation >= 0.0 else None
        existing = stream.observations.get(frame_index)
        if existing is None:
            existing = _FrameObservation(
                assistant_has_floor=assistant_floor,
                user_has_floor=valid_user_floor,
                turn_completion=valid_completion,
                continuation_pause=valid_continuation,
            )
            stream.observations[frame_index] = existing
        else:
            _merge_observation(
                existing=existing,
                assistant_floor=assistant_floor,
                user_floor=valid_user_floor,
                completion=valid_completion,
                continuation=valid_continuation,
                audio_path=sample.user_audio_path,
                frame_index=frame_index,
            )
        existing.categories.add(sample.category)
        existing.source_window_ids.add(sample.window_id)


def _merge_observation(
    existing: _FrameObservation,
    assistant_floor: float,
    user_floor: float | None,
    completion: float | None,
    continuation: float | None,
    audio_path: str,
    frame_index: int,
) -> None:
    if not math.isclose(
        existing.assistant_has_floor,
        assistant_floor,
        abs_tol=OBSERVATION_TOLERANCE,
    ):
        raise ValueError(f"Conflicting labels for {audio_path} frame {frame_index}.")
    if (
        _optional_values_conflict(existing.user_has_floor, user_floor)
        or _optional_values_conflict(existing.turn_completion, completion)
        or _optional_values_conflict(
            existing.continuation_pause,
            continuation,
        )
    ):
        raise ValueError(f"Conflicting labels for {audio_path} frame {frame_index}.")
    if existing.user_has_floor is None:
        existing.user_has_floor = user_floor
    if existing.turn_completion is None:
        existing.turn_completion = completion
    if existing.continuation_pause is None:
        existing.continuation_pause = continuation


def _optional_values_conflict(first: float | None, second: float | None) -> bool:
    return (
        first is not None
        and second is not None
        and not math.isclose(
            first,
            second,
            abs_tol=OBSERVATION_TOLERANCE,
        )
    )


def _stream_candidates(
    stream_key: _StreamKey,
    stream: _Stream,
    frame_seconds: float,
    user_floor_threshold: float,
    assistant_active_threshold: float,
    horizon_frames: int,
) -> tuple[TurnCompletionCandidate, ...]:
    candidates: list[TurnCompletionCandidate] = []
    for anchor_frame, anchor in sorted(stream.observations.items()):
        if anchor.turn_completion is None:
            continue
        if anchor.assistant_has_floor >= assistant_active_threshold:
            continue
        if anchor.user_has_floor is None or anchor.user_has_floor >= user_floor_threshold:
            continue
        candidate = _candidate_at_anchor(
            stream_key=stream_key,
            stream=stream,
            anchor_frame=anchor_frame,
            completion_probability=anchor.turn_completion,
            continuation_probability=anchor.continuation_pause,
            frame_seconds=frame_seconds,
            user_floor_threshold=user_floor_threshold,
            horizon_frames=horizon_frames,
        )
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _candidate_at_anchor(
    stream_key: _StreamKey,
    stream: _Stream,
    anchor_frame: int,
    completion_probability: float,
    continuation_probability: float | None,
    frame_seconds: float,
    user_floor_threshold: float,
    horizon_frames: int,
) -> TurnCompletionCandidate | None:
    preceding_speech_start = _preceding_speech_start(
        stream=stream,
        anchor_frame=anchor_frame,
        user_floor_threshold=user_floor_threshold,
    )
    if preceding_speech_start is None:
        return None
    target_frames: list[int] = []
    boundary_kind = CompletionBoundaryKind.TERMINAL
    for frame_index in range(anchor_frame, anchor_frame + horizon_frames):
        observation = stream.observations.get(frame_index)
        if observation is None or observation.user_has_floor is None:
            return None
        if observation.user_has_floor >= user_floor_threshold:
            boundary_kind = CompletionBoundaryKind.CONTINUATION
            break
        target_frames.append(frame_index)
    if not target_frames:
        return None
    end_frame = target_frames[-1] + 1
    observations = tuple(stream.observations[index] for index in target_frames)
    categories = tuple(sorted({value for item in observations for value in item.categories}))
    source_window_ids = tuple(
        sorted({value for item in observations for value in item.source_window_ids})
    )
    anchor_seconds = anchor_frame * frame_seconds
    end_seconds = end_frame * frame_seconds
    return TurnCompletionCandidate(
        candidate_id=_candidate_id(stream_key, anchor_seconds, end_seconds),
        dataset_id=stream_key.dataset_id,
        dataset_name=stream.dataset_name,
        conversation_id=stream_key.sample_id,
        external_id=stream.external_id,
        user_side=stream_key.user_side,
        user_audio_path=stream_key.user_audio_path,
        preceding_speech_start_seconds=preceding_speech_start * frame_seconds,
        anchor_seconds=anchor_seconds,
        end_seconds=end_seconds,
        boundary_kind=boundary_kind,
        continuation_probability=continuation_probability,
        target_points=tuple(
            CompletionTargetPoint(
                absolute_time_seconds=(frame_index + 1) * frame_seconds,
                elapsed_seconds=(offset + 1) * frame_seconds,
                completion_probability=completion_probability,
            )
            for offset, frame_index in enumerate(target_frames)
        ),
        categories=categories,
        source_window_ids=source_window_ids,
    )


def _preceding_speech_start(
    stream: _Stream,
    anchor_frame: int,
    user_floor_threshold: float,
) -> int | None:
    frame_index = anchor_frame - 1
    observation = stream.observations.get(frame_index)
    if observation is None or observation.user_has_floor is None:
        return None
    if observation.user_has_floor < user_floor_threshold:
        return None
    while True:
        preceding = stream.observations.get(frame_index - 1)
        if (
            preceding is None
            or preceding.user_has_floor is None
            or preceding.user_has_floor < user_floor_threshold
        ):
            return frame_index
        frame_index -= 1


def _candidate_id(
    stream_key: _StreamKey,
    anchor_seconds: float,
    end_seconds: float,
) -> str:
    payload = (
        f"turn-completion-candidate-v2:{stream_key.dataset_id}:{stream_key.sample_id}:"
        f"{stream_key.user_side}:{stream_key.user_audio_path}:{anchor_seconds:.6f}:"
        f"{end_seconds:.6f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stream_sort_key(stream_key: _StreamKey) -> tuple[str, str, str, str]:
    return (
        stream_key.dataset_id,
        stream_key.sample_id,
        stream_key.user_side,
        stream_key.user_audio_path,
    )
