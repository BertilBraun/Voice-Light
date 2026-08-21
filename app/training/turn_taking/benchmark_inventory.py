from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.local.training_corpus.export import MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_models import (
    CandidateInventory,
    CandidateInventoryManifest,
    CandidateTargetPoint,
    SilenceCandidate,
    candidate_rows_sha256,
)

DEFAULT_FRAME_SECONDS = 0.08
DEFAULT_MINIMUM_SILENCE_SECONDS = 0.1
DEFAULT_SILENCE_FLOOR_THRESHOLD = 0.5
OBSERVATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class _StreamKey:
    dataset_id: str
    sample_id: str
    user_side: str
    user_audio_path: str


@dataclass
class _FrameObservation:
    user_has_floor: float
    user_yield: float | None
    categories: set[str] = field(default_factory=set)
    source_window_ids: set[str] = field(default_factory=set)


@dataclass
class _Stream:
    dataset_name: str
    external_id: str
    observations: dict[int, _FrameObservation] = field(default_factory=dict)


def build_candidate_inventory(
    samples: Iterable[MaterializedTrainingSample],
    corpus_repository: str,
    corpus_revision: str,
    split: TrainingCorpusSplit,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
    silence_floor_threshold: float = DEFAULT_SILENCE_FLOOR_THRESHOLD,
    minimum_silence_seconds: float = DEFAULT_MINIMUM_SILENCE_SECONDS,
) -> CandidateInventory:
    if frame_seconds <= 0.0:
        raise ValueError("frame_seconds must be positive.")
    if not 0.0 <= silence_floor_threshold <= 1.0:
        raise ValueError("silence_floor_threshold must be between zero and one.")
    if minimum_silence_seconds <= 0.0:
        raise ValueError("minimum_silence_seconds must be positive.")
    materialized_samples = tuple(samples)
    if not materialized_samples:
        raise ValueError("Candidate inventory requires at least one sample.")

    streams: dict[_StreamKey, _Stream] = {}
    for sample in materialized_samples:
        if sample.split is not split:
            raise ValueError(
                f"Sample {sample.window_id} belongs to {sample.split.value}, not {split.value}."
            )
        _add_sample_observations(streams=streams, sample=sample, frame_seconds=frame_seconds)

    candidates = tuple(
        candidate
        for stream_key in sorted(streams, key=_stream_sort_key)
        for candidate in _stream_candidates(
            stream_key=stream_key,
            stream=streams[stream_key],
            frame_seconds=frame_seconds,
            silence_floor_threshold=silence_floor_threshold,
            minimum_silence_seconds=minimum_silence_seconds,
        )
    )
    manifest = CandidateInventoryManifest(
        corpus_repository=corpus_repository,
        corpus_revision=corpus_revision,
        split=split,
        frame_seconds=frame_seconds,
        silence_floor_threshold=silence_floor_threshold,
        minimum_silence_seconds=minimum_silence_seconds,
        sample_window_count=len(materialized_samples),
        conversation_count=len({(key.dataset_id, key.sample_id) for key in streams}),
        candidate_count=len(candidates),
        candidate_sha256=candidate_rows_sha256(candidates),
    )
    return CandidateInventory(manifest=manifest, candidates=candidates)


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
    for offset, (user_has_floor, user_yield) in enumerate(
        zip(sample.p_user_has_floor, sample.p_user_yield, strict=True)
    ):
        if user_has_floor < 0.0:
            continue
        valid_user_yield = user_yield if user_yield >= 0.0 else None
        frame_index = start_frame + offset
        existing = stream.observations.get(frame_index)
        if existing is None:
            existing = _FrameObservation(
                user_has_floor=user_has_floor,
                user_yield=valid_user_yield,
            )
            stream.observations[frame_index] = existing
        else:
            floor_matches = math.isclose(
                existing.user_has_floor, user_has_floor, abs_tol=OBSERVATION_TOLERANCE
            )
            yield_matches = (
                existing.user_yield is None
                or valid_user_yield is None
                or math.isclose(
                    existing.user_yield,
                    valid_user_yield,
                    abs_tol=OBSERVATION_TOLERANCE,
                )
            )
            if not floor_matches or not yield_matches:
                raise ValueError(
                    f"Conflicting labels for {sample.user_audio_path} frame {frame_index}."
                )
            if existing.user_yield is None:
                existing.user_yield = valid_user_yield
        existing.categories.add(sample.category)
        existing.source_window_ids.add(sample.window_id)


def _stream_candidates(
    stream_key: _StreamKey,
    stream: _Stream,
    frame_seconds: float,
    silence_floor_threshold: float,
    minimum_silence_seconds: float,
) -> tuple[SilenceCandidate, ...]:
    candidates: list[SilenceCandidate] = []
    silence_frames: list[int] = []
    has_leading_speech = False
    previous_frame_index: int | None = None
    for frame_index in sorted(stream.observations):
        if previous_frame_index is not None and frame_index != previous_frame_index + 1:
            silence_frames.clear()
            has_leading_speech = False
        observation = stream.observations[frame_index]
        if observation.user_has_floor >= silence_floor_threshold:
            if silence_frames and has_leading_speech:
                candidate = _candidate_from_frames(
                    stream_key=stream_key,
                    stream=stream,
                    silence_frames=tuple(silence_frames),
                    trailing_speech_frame=frame_index,
                    frame_seconds=frame_seconds,
                    minimum_silence_seconds=minimum_silence_seconds,
                )
                if candidate is not None:
                    candidates.append(candidate)
            silence_frames.clear()
            has_leading_speech = True
        elif has_leading_speech:
            silence_frames.append(frame_index)
        previous_frame_index = frame_index
    return tuple(candidates)


def _candidate_from_frames(
    stream_key: _StreamKey,
    stream: _Stream,
    silence_frames: tuple[int, ...],
    trailing_speech_frame: int,
    frame_seconds: float,
    minimum_silence_seconds: float,
) -> SilenceCandidate | None:
    start_seconds = silence_frames[0] * frame_seconds
    end_seconds = trailing_speech_frame * frame_seconds
    if end_seconds - start_seconds + OBSERVATION_TOLERANCE < minimum_silence_seconds:
        return None
    observations = tuple(stream.observations[index] for index in silence_frames)
    categories = tuple(sorted({value for item in observations for value in item.categories}))
    source_window_ids = tuple(
        sorted({value for item in observations for value in item.source_window_ids})
    )
    candidate_id = _candidate_id(
        stream_key=stream_key,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    target_points = tuple(
        CandidateTargetPoint(
            absolute_time_seconds=(frame_index + 1) * frame_seconds,
            silence_duration_seconds=(offset + 1) * frame_seconds,
            yield_probability=user_yield,
        )
        for offset, frame_index in enumerate(silence_frames)
        if (user_yield := stream.observations[frame_index].user_yield) is not None
    )
    if not target_points:
        return None
    return SilenceCandidate(
        candidate_id=candidate_id,
        dataset_id=stream_key.dataset_id,
        dataset_name=stream.dataset_name,
        conversation_id=stream_key.sample_id,
        external_id=stream.external_id,
        user_side=stream_key.user_side,
        user_audio_path=stream_key.user_audio_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        target_points=target_points,
        categories=categories,
        source_window_ids=source_window_ids,
    )


def _candidate_id(
    stream_key: _StreamKey,
    start_seconds: float,
    end_seconds: float,
) -> str:
    payload = (
        f"causal-candidate-v1:{stream_key.dataset_id}:{stream_key.sample_id}:"
        f"{stream_key.user_side}:{stream_key.user_audio_path}:{start_seconds:.6f}:"
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
