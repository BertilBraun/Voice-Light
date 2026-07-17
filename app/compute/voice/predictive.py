from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from app.compute.voice.schemas import (
    CapturedAudioChunk,
    CausalSource,
    TraceStamp,
    TranscriptRevision,
)


class CandidateLifecycle(StrEnum):
    CREATED = "created"
    PREFILLING = "prefilling"
    STREAMING = "streaming"
    READY = "ready"
    COMMITTED = "committed"
    INVALIDATED = "invalidated"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CandidateInvalidationReason(StrEnum):
    USER_ACTIVITY_RESUMED = "user_activity_resumed"
    STABLE_PREFIX_REVISED = "stable_prefix_revised"
    PREDICTION_RETURNED_TO_HOLD = "prediction_returned_to_hold"
    TRANSCRIPT_SUPERSEDED = "transcript_superseded"
    FLOOR_TAKING_OVERLAP = "floor_taking_overlap"
    SESSION_CANCELLED = "session_cancelled"
    GENERATION_CANCELLED = "generation_cancelled"
    LANGUAGE_MODEL_FAILED = "language_model_failed"
    SPEECH_SYNTHESIS_FAILED = "speech_synthesis_failed"
    EMPTY_FINAL_TRANSCRIPT = "empty_final_transcript"
    FINAL_PREFIX_CHANGED = "final_prefix_changed"
    MATERIAL_REQUEST_CHANGE = "material_request_change"


@dataclass(frozen=True)
class MediaLatencyPoint:
    monotonic_time_seconds: float
    input_sample_position: int
    output_sample_position: int | None
    text_offset: int | None

    def __post_init__(self) -> None:
        if self.monotonic_time_seconds < 0:
            raise ValueError("A monotonic latency timestamp cannot be negative.")
        if self.input_sample_position < 0:
            raise ValueError("An input sample position cannot be negative.")
        if self.output_sample_position is not None and self.output_sample_position < 0:
            raise ValueError("An output sample position cannot be negative.")
        if self.text_offset is not None and self.text_offset < 0:
            raise ValueError("A text offset cannot be negative.")


@dataclass(frozen=True)
class ReleasedTextDelta:
    generation_id: int
    text: str


@dataclass(frozen=True)
class ReleasedAudioStart:
    generation_id: int


@dataclass(frozen=True)
class ReleasedWordBoundary:
    generation_id: int
    text_offset: int
    start_sample: int


@dataclass(frozen=True)
class ReleasedAudioChunk:
    generation_id: int
    sequence_number: int
    start_sample: int
    pcm_bytes: bytes


@dataclass(frozen=True)
class ReleasedAudioEnd:
    generation_id: int


CandidateOutput = (
    ReleasedTextDelta
    | ReleasedAudioStart
    | ReleasedWordBoundary
    | ReleasedAudioChunk
    | ReleasedAudioEnd
)


class PlaybackSink(Protocol):
    async def send(self, output: CandidateOutput) -> None: ...


class CandidateReleaseGate:
    def __init__(self, sink: PlaybackSink, released: bool) -> None:
        self.sink = sink
        self._released = released
        self._discarded = False
        self._outputs: list[CandidateOutput] = []
        self._lock = asyncio.Lock()
        self.first_released_pcm_at: float | None = None
        self.first_released_pcm_start_sample: int | None = None

    @property
    def released(self) -> bool:
        return self._released

    @property
    def outputs(self) -> tuple[CandidateOutput, ...]:
        return tuple(self._outputs)

    @property
    def buffered_pcm_sample_count(self) -> int:
        return sum(
            len(output.pcm_bytes) // 2
            for output in self._outputs
            if isinstance(output, ReleasedAudioChunk)
        )

    async def publish(self, output: CandidateOutput) -> None:
        async with self._lock:
            if self._discarded:
                return
            self._outputs.append(output)
            if self._released:
                await self._send(output)

    async def release(self) -> None:
        async with self._lock:
            if self._discarded:
                raise ValueError("A discarded candidate cannot be released.")
            if self._released:
                return
            self._released = True
            for output in self._outputs:
                await self._send(output)

    async def discard(self) -> None:
        async with self._lock:
            self._discarded = True

    async def _send(self, output: CandidateOutput) -> None:
        await self.sink.send(output)
        if isinstance(output, ReleasedAudioChunk) and self.first_released_pcm_at is None:
            self.first_released_pcm_at = time.perf_counter()
            self.first_released_pcm_start_sample = output.start_sample


class TranscriptRevisionTracker:
    def __init__(self) -> None:
        self.next_revision_id = 1
        self.previous_text = ""
        self.stable_prefix = ""
        self.latest: TranscriptRevision | None = None

    def update(
        self,
        text: str,
        chunk: CapturedAudioChunk,
        inference_step: int,
        observed_through_input_sample: int,
        model_name: str | None,
        model_revision: str | None,
    ) -> TranscriptRevision | None:
        normalized_text = text.strip()
        if not normalized_text:
            return None
        stable_prefix = self._updated_stable_prefix(normalized_text)
        volatile_suffix = normalized_text[len(stable_prefix) :]
        if (
            self.latest is not None
            and self.latest.stable_prefix == stable_prefix
            and self.latest.volatile_suffix == volatile_suffix
        ):
            return self.latest
        revision_id = self.next_revision_id
        self.next_revision_id += 1
        revision = TranscriptRevision(
            stamp=TraceStamp(
                event_id=str(uuid4()),
                parent_event_ids=(() if self.latest is None else (self.latest.stamp.event_id,)),
                stream_epoch=chunk.stream_epoch,
                turn_epoch=chunk.turn_epoch,
                inference_step=inference_step,
                observation_id=f"audio:{chunk.stream_epoch}:{chunk.sequence_number}",
                observation_monotonic_time_ns=chunk.monotonic_observation_time_ns,
                emission_monotonic_time_ns=time.perf_counter_ns(),
                encoder_frame_start=None,
                encoder_frame_end=None,
                input_start_sample=chunk.start_input_sample,
                input_end_sample=chunk.end_input_sample,
                observed_through_input_sample=observed_through_input_sample,
                input_sample_position=chunk.end_input_sample,
                output_sample_position=None,
                conditioned_transcript_revision_id=None,
                conditioned_playback_event_id=chunk.playback_condition.event_id,
                source=CausalSource.NEMOTRON_ASR,
                model_name=model_name,
                model_revision=model_revision,
            ),
            revision_id=revision_id,
            supersedes_revision_id=(None if self.latest is None else self.latest.revision_id),
            stable_prefix=stable_prefix,
            volatile_suffix=volatile_suffix,
            audio_sample_position=chunk.end_input_sample,
            stable_prefix_end_sample=chunk.end_input_sample,
            confidence=1.0,
        )
        self.previous_text = normalized_text
        self.stable_prefix = stable_prefix
        self.latest = revision
        return revision

    def reset_turn(self) -> None:
        self.previous_text = ""
        self.stable_prefix = ""
        self.latest = None

    def _updated_stable_prefix(self, text: str) -> str:
        if not self.previous_text:
            return ""
        if not text.startswith(self.stable_prefix):
            return _stable_common_prefix(self.previous_text, text)
        common_prefix = _stable_common_prefix(self.previous_text, text)
        if len(common_prefix) > len(self.stable_prefix):
            return common_prefix
        return self.stable_prefix


@dataclass(frozen=True)
class InvalidationCount:
    reason: CandidateInvalidationReason
    count: int


@dataclass(frozen=True)
class PredictiveMetricsReport:
    candidate_count: int
    candidate_hit_rate: float
    invalidation_rate: float
    invalidations: tuple[InvalidationCount, ...]
    stale_candidate_escape_rate: float
    commit_to_first_played_audio_p50_ms: float | None
    commit_to_first_played_audio_p90_ms: float | None
    commit_to_first_played_audio_p95_ms: float | None
    commit_to_first_released_pcm_p50_ms: float | None
    commit_to_first_released_pcm_p90_ms: float | None
    commit_to_first_released_pcm_p95_ms: float | None
    true_end_to_first_played_audio_p50_ms: float | None
    true_end_to_first_played_audio_p90_ms: float | None
    true_end_to_first_played_audio_p95_ms: float | None
    hidden_work_seconds: float
    hidden_qwen_tokens: int
    hidden_tts_samples: int
    wasted_qwen_tokens: int
    wasted_tts_samples: int
    no_candidate_latency_p50_ms: float | None
    post_invalidation_latency_p50_ms: float | None


class PredictiveMetrics:
    def __init__(self) -> None:
        self.candidate_count = 0
        self.promoted_count = 0
        self.stale_escape_count = 0
        self.invalidations: Counter[CandidateInvalidationReason] = Counter()
        self.commit_to_played_ms: list[float] = []
        self.commit_to_released_pcm_ms: list[float] = []
        self.true_end_to_played_ms: list[float] = []
        self.no_candidate_latency_ms: list[float] = []
        self.post_invalidation_latency_ms: list[float] = []
        self.hidden_work_seconds = 0.0
        self.hidden_qwen_tokens = 0
        self.hidden_tts_samples = 0
        self.wasted_qwen_tokens = 0
        self.wasted_tts_samples = 0

    def record_candidate_created(self) -> None:
        self.candidate_count += 1

    def record_promotion(
        self,
        hidden_work_seconds: float,
        hidden_qwen_tokens: int,
        hidden_tts_samples: int,
    ) -> None:
        self.promoted_count += 1
        self.hidden_work_seconds += max(hidden_work_seconds, 0.0)
        self.hidden_qwen_tokens += hidden_qwen_tokens
        self.hidden_tts_samples += hidden_tts_samples

    def record_invalidation(
        self,
        reason: CandidateInvalidationReason,
        qwen_tokens: int,
        tts_samples: int,
    ) -> None:
        self.invalidations[reason] += 1
        self.wasted_qwen_tokens += qwen_tokens
        self.wasted_tts_samples += tts_samples

    def record_first_playback(
        self,
        commit_at: float,
        playback_at: float,
        true_end_at: float | None,
        had_candidate: bool,
        followed_invalidation: bool,
    ) -> None:
        latency_ms = _milliseconds_between(commit_at, playback_at)
        self.commit_to_played_ms.append(latency_ms)
        if true_end_at is not None:
            self.true_end_to_played_ms.append(_milliseconds_between(true_end_at, playback_at))
        if not had_candidate:
            self.no_candidate_latency_ms.append(latency_ms)
        if followed_invalidation:
            self.post_invalidation_latency_ms.append(latency_ms)

    def record_first_release(self, commit_at: float, release_at: float) -> None:
        self.commit_to_released_pcm_ms.append(_milliseconds_between(commit_at, release_at))

    def report(self) -> PredictiveMetricsReport:
        invalidated_count = sum(self.invalidations.values())
        return PredictiveMetricsReport(
            candidate_count=self.candidate_count,
            candidate_hit_rate=_rate(self.promoted_count, self.candidate_count),
            invalidation_rate=_rate(invalidated_count, self.candidate_count),
            invalidations=tuple(
                InvalidationCount(reason=reason, count=count)
                for reason, count in sorted(
                    self.invalidations.items(),
                    key=lambda item: item[0].value,
                )
            ),
            stale_candidate_escape_rate=_rate(
                self.stale_escape_count,
                self.promoted_count,
            ),
            commit_to_first_played_audio_p50_ms=_percentile(
                self.commit_to_played_ms,
                0.50,
            ),
            commit_to_first_played_audio_p90_ms=_percentile(
                self.commit_to_played_ms,
                0.90,
            ),
            commit_to_first_played_audio_p95_ms=_percentile(
                self.commit_to_played_ms,
                0.95,
            ),
            commit_to_first_released_pcm_p50_ms=_percentile(
                self.commit_to_released_pcm_ms,
                0.50,
            ),
            commit_to_first_released_pcm_p90_ms=_percentile(
                self.commit_to_released_pcm_ms,
                0.90,
            ),
            commit_to_first_released_pcm_p95_ms=_percentile(
                self.commit_to_released_pcm_ms,
                0.95,
            ),
            true_end_to_first_played_audio_p50_ms=_percentile(
                self.true_end_to_played_ms,
                0.50,
            ),
            true_end_to_first_played_audio_p90_ms=_percentile(
                self.true_end_to_played_ms,
                0.90,
            ),
            true_end_to_first_played_audio_p95_ms=_percentile(
                self.true_end_to_played_ms,
                0.95,
            ),
            hidden_work_seconds=self.hidden_work_seconds,
            hidden_qwen_tokens=self.hidden_qwen_tokens,
            hidden_tts_samples=self.hidden_tts_samples,
            wasted_qwen_tokens=self.wasted_qwen_tokens,
            wasted_tts_samples=self.wasted_tts_samples,
            no_candidate_latency_p50_ms=_percentile(
                self.no_candidate_latency_ms,
                0.50,
            ),
            post_invalidation_latency_p50_ms=_percentile(
                self.post_invalidation_latency_ms,
                0.50,
            ),
        )


def candidate_final_invalidation_reason(
    stable_prefix: str,
    prompted_text: str,
    final_text: str,
) -> CandidateInvalidationReason | None:
    if not final_text.startswith(stable_prefix):
        return CandidateInvalidationReason.FINAL_PREFIX_CHANGED
    if _semantic_words(prompted_text) != _semantic_words(final_text):
        return CandidateInvalidationReason.MATERIAL_REQUEST_CHANGE
    return None


def candidate_revision_invalidation_reason(
    source: CausalSource,
    stable_prefix: str,
    prompted_text: str,
    revised_stable_prefix: str,
    revised_volatile_suffix: str,
) -> CandidateInvalidationReason | None:
    match source:
        case CausalSource.SILERO_VAD:
            revised_text = f"{revised_stable_prefix}{revised_volatile_suffix}".strip()
            if revised_text != prompted_text:
                return CandidateInvalidationReason.TRANSCRIPT_SUPERSEDED
        case _:
            if not revised_stable_prefix.startswith(stable_prefix):
                return CandidateInvalidationReason.STABLE_PREFIX_REVISED
    return None


def _stable_common_prefix(previous_text: str, current_text: str) -> str:
    maximum_length = min(len(previous_text), len(current_text))
    common_length = 0
    while (
        common_length < maximum_length
        and previous_text[common_length] == current_text[common_length]
    ):
        common_length += 1
    common_prefix = current_text[:common_length]
    if common_length == len(previous_text) == len(current_text):
        return common_prefix
    boundary_matches = tuple(re.finditer(r"(?:\s+|[.!?,;:]\s*)", common_prefix))
    if not boundary_matches:
        return ""
    return common_prefix[: boundary_matches[-1].end()]


def _semantic_words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold()))


def _milliseconds_between(start_at: float, end_at: float) -> float:
    if end_at < start_at:
        raise AssertionError("Metrics timestamps must increase monotonically.")
    return (end_at - start_at) * 1_000


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(math.ceil(percentile * len(ordered)), 1)
    return ordered[rank - 1]
