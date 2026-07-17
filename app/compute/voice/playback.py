from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from app.compute.voice.schemas import (
    CausalSource,
    PlaybackCommandAcknowledgementEvent,
    PlaybackCommandAction,
    PlaybackCommandEvent,
    PlaybackCompleteEvent,
    PlaybackCondition,
    PlaybackConditionAuthority,
    PlaybackProgressEvent,
    PlaybackStartedEvent,
    PlaybackState,
)


@dataclass(frozen=True)
class PlaybackPolicyConfig:
    duck_decibels: float = -18.0
    duck_ramp_duration_ms: int = 25
    pause_deadline_ms: int = 120
    resume_ramp_duration_ms: int = 25
    maximum_resumable_paused_age_ms: int = 800
    target_paused_buffer_age_ms: int = 500
    maximum_synthesized_ahead_ms: int = 500
    generation_boundary_hold_ms: int = 350
    classification_deadline_ms: int = 500

    def __post_init__(self) -> None:
        if self.duck_decibels >= 0:
            raise ValueError("Duck attenuation must be below zero decibels.")
        durations = (
            self.duck_ramp_duration_ms,
            self.pause_deadline_ms,
            self.resume_ramp_duration_ms,
            self.maximum_resumable_paused_age_ms,
            self.target_paused_buffer_age_ms,
            self.maximum_synthesized_ahead_ms,
            self.generation_boundary_hold_ms,
            self.classification_deadline_ms,
        )
        if any(duration <= 0 for duration in durations):
            raise ValueError("Playback policy durations must be positive.")
        if self.generation_boundary_hold_ms >= self.classification_deadline_ms:
            raise ValueError("Generation must be held before overlap classification expires.")
        if self.target_paused_buffer_age_ms > self.maximum_resumable_paused_age_ms:
            raise ValueError("The target paused-buffer age cannot exceed the resume limit.")

    @property
    def duck_gain(self) -> float:
        return 10 ** (self.duck_decibels / 20)


class PlaybackAcknowledgementDisposition(StrEnum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    STALE = "stale"


@dataclass
class PlaybackCommandRecord:
    command: PlaybackCommandEvent
    estimated_rendered_output_sample_position: int
    acknowledgement: PlaybackCommandAcknowledgementEvent | None = None
    acknowledgement_received_monotonic_time_ns: int | None = None


@dataclass(frozen=True)
class PlaybackMetric:
    command_id: str
    generation_id: int
    action: PlaybackCommandAction
    command_to_acknowledgement_ms: float
    rendered_output_sample_estimate_error: int


@dataclass(frozen=True)
class PlaybackMetricsReport:
    command_count: int
    acknowledgement_count: int
    duplicate_acknowledgement_count: int
    stale_acknowledgement_count: int
    command_to_acknowledgement_p95_ms: float | None
    rendered_output_sample_estimate_error_p95: int | None
    maximum_buffered_source_sample_count: int
    discarded_source_sample_count: int
    replayed_source_sample_count: int
    skipped_source_sample_count: int


class PlaybackMetrics:
    def __init__(self) -> None:
        self.command_count = 0
        self.duplicate_acknowledgement_count = 0
        self.stale_acknowledgement_count = 0
        self.metrics: list[PlaybackMetric] = []
        self.discarded_source_sample_count = 0
        self.maximum_buffered_source_sample_count = 0
        self.replayed_source_sample_count = 0
        self.skipped_source_sample_count = 0

    def record_command(self) -> None:
        self.command_count += 1

    def record_acknowledgement(
        self,
        record: PlaybackCommandRecord,
        acknowledgement: PlaybackCommandAcknowledgementEvent,
        received_monotonic_time_ns: int,
    ) -> None:
        latency_ns = received_monotonic_time_ns - record.command.issued_monotonic_time_ns
        if latency_ns < 0:
            raise AssertionError("Playback acknowledgement preceded its command.")
        self.metrics.append(
            PlaybackMetric(
                command_id=record.command.command_id,
                generation_id=record.command.generation_id,
                action=record.command.action,
                command_to_acknowledgement_ms=latency_ns / 1_000_000,
                rendered_output_sample_estimate_error=(
                    acknowledgement.rendered_output_sample_position
                    - record.estimated_rendered_output_sample_position
                ),
            )
        )
        self.discarded_source_sample_count += acknowledgement.discarded_source_sample_count
        self.maximum_buffered_source_sample_count = max(
            self.maximum_buffered_source_sample_count,
            acknowledgement.queued_source_sample_count,
        )
        self.replayed_source_sample_count += acknowledgement.replayed_source_sample_count
        self.skipped_source_sample_count += acknowledgement.skipped_source_sample_count

    def report(self) -> PlaybackMetricsReport:
        latencies = [metric.command_to_acknowledgement_ms for metric in self.metrics]
        absolute_errors = [
            abs(metric.rendered_output_sample_estimate_error) for metric in self.metrics
        ]
        return PlaybackMetricsReport(
            command_count=self.command_count,
            acknowledgement_count=len(self.metrics),
            duplicate_acknowledgement_count=self.duplicate_acknowledgement_count,
            stale_acknowledgement_count=self.stale_acknowledgement_count,
            command_to_acknowledgement_p95_ms=_percentile(latencies, 0.95),
            rendered_output_sample_estimate_error_p95=_percentile(absolute_errors, 0.95),
            maximum_buffered_source_sample_count=self.maximum_buffered_source_sample_count,
            discarded_source_sample_count=self.discarded_source_sample_count,
            replayed_source_sample_count=self.replayed_source_sample_count,
            skipped_source_sample_count=self.skipped_source_sample_count,
        )


class PlaybackController:
    def __init__(self, source_sample_rate: int, config: PlaybackPolicyConfig) -> None:
        if source_sample_rate <= 0:
            raise ValueError("Playback source sample rate must be positive.")
        self.source_sample_rate = source_sample_rate
        self.config = config
        self.active_generation_id: int | None = None
        self.command_records: dict[str, PlaybackCommandRecord] = {}
        self.metrics = PlaybackMetrics()
        self.paused_at_server_monotonic_time_ns: int | None = None
        self.condition = PlaybackCondition(
            event_id=str(uuid4()),
            generation_id=None,
            state=PlaybackState.IDLE,
            assistant_audible=False,
            latest_output_sample_position=0,
            latest_source_sample_position=0,
            output_sample_rate=None,
            monotonic_time_ns=time.perf_counter_ns(),
            authority=PlaybackConditionAuthority.SERVER_ESTIMATED,
        )

    def observe_condition(self, monotonic_time_ns: int) -> PlaybackCondition:
        condition = self.condition
        return PlaybackCondition(
            event_id=str(uuid4()),
            generation_id=condition.generation_id,
            state=condition.state,
            assistant_audible=condition.assistant_audible,
            latest_output_sample_position=condition.latest_output_sample_position,
            latest_source_sample_position=condition.latest_source_sample_position,
            output_sample_rate=condition.output_sample_rate,
            monotonic_time_ns=monotonic_time_ns,
            authority=condition.authority,
        )

    def replace_generation(self, generation_id: int) -> None:
        if generation_id <= 0:
            raise ValueError("Playback generation IDs must be positive.")
        if self.active_generation_id is not None and generation_id <= self.active_generation_id:
            raise ValueError("Replacement generation IDs must increase.")
        self.active_generation_id = generation_id
        self.paused_at_server_monotonic_time_ns = None
        self._set_condition(
            generation_id=generation_id,
            state=PlaybackState.QUEUED,
            rendered_output_sample_position=0,
            source_sample_position=0,
            output_sample_rate=self.condition.output_sample_rate,
            authority=PlaybackConditionAuthority.SERVER_ESTIMATED,
        )

    def validate_synthesized_source_position(
        self,
        generation_id: int,
        synthesized_source_sample_count: int,
    ) -> None:
        if generation_id != self.active_generation_id:
            return
        if synthesized_source_sample_count < self.condition.latest_source_sample_position:
            raise AssertionError("Synthesized audio cannot trail browser playback.")

    def estimate_state(self, generation_id: int, state: PlaybackState) -> None:
        if generation_id != self.active_generation_id:
            return
        self._set_condition(
            generation_id=generation_id,
            state=state,
            rendered_output_sample_position=self.condition.latest_output_sample_position,
            source_sample_position=self.condition.latest_source_sample_position,
            output_sample_rate=self.condition.output_sample_rate,
            authority=PlaybackConditionAuthority.SERVER_ESTIMATED,
        )

    def record_started(self, event: PlaybackStartedEvent) -> bool:
        if event.generation_id != self.active_generation_id:
            return False
        self._set_condition_from_browser(
            generation_id=event.generation_id,
            state=PlaybackState.SPEAKING,
            rendered_output_sample_position=event.rendered_output_sample_position,
            source_sample_position=event.source_sample_position,
            output_sample_rate=event.output_sample_rate,
        )
        return True

    def record_progress(self, event: PlaybackProgressEvent) -> bool:
        if event.generation_id != self.active_generation_id:
            return False
        if self.condition.state in (PlaybackState.CANCELLED, PlaybackState.COMPLETED):
            return False
        state = (
            PlaybackState.SPEAKING
            if self.condition.state in (PlaybackState.QUEUED, PlaybackState.SPEAKING)
            else self.condition.state
        )
        self._set_condition_from_browser(
            generation_id=event.generation_id,
            state=state,
            rendered_output_sample_position=event.rendered_output_sample_position,
            source_sample_position=event.played_sample_count,
            output_sample_rate=event.output_sample_rate,
        )
        return True

    def record_complete(self, event: PlaybackCompleteEvent) -> bool:
        if event.generation_id != self.active_generation_id:
            return False
        self._set_condition_from_browser(
            generation_id=event.generation_id,
            state=PlaybackState.COMPLETED,
            rendered_output_sample_position=event.rendered_output_sample_position,
            source_sample_position=event.source_sample_position,
            output_sample_rate=event.output_sample_rate,
        )
        return True

    def issue_duck(
        self,
        generation_id: int,
        causal_event_id: str,
        causal_source: CausalSource,
        stream_epoch: int,
        turn_epoch: int,
        confidence: float,
    ) -> PlaybackCommandEvent:
        return self._issue(
            generation_id=generation_id,
            action=PlaybackCommandAction.DUCK,
            causal_event_id=causal_event_id,
            causal_source=causal_source,
            stream_epoch=stream_epoch,
            turn_epoch=turn_epoch,
            confidence=confidence,
            target_gain=self.config.duck_gain,
            gain_ramp_duration_ms=self.config.duck_ramp_duration_ms,
            estimated_state=PlaybackState.DUCKING,
        )

    def issue_pause(
        self,
        generation_id: int,
        causal_event_id: str,
        causal_source: CausalSource,
        stream_epoch: int,
        turn_epoch: int,
        confidence: float,
        requested_boundary_source_sample_position: int | None,
    ) -> PlaybackCommandEvent:
        output_sample_rate = self.condition.output_sample_rate
        if output_sample_rate is None:
            raise ValueError("Cannot issue a sample deadline before playback reports its rate.")
        deadline_sample_count = output_sample_rate * self.config.pause_deadline_ms // 1_000
        return self._issue(
            generation_id=generation_id,
            action=PlaybackCommandAction.PAUSE_AT_BOUNDARY,
            causal_event_id=causal_event_id,
            causal_source=causal_source,
            stream_epoch=stream_epoch,
            turn_epoch=turn_epoch,
            confidence=confidence,
            requested_boundary_source_sample_position=(requested_boundary_source_sample_position),
            rendered_output_sample_deadline=(
                self.condition.latest_output_sample_position + deadline_sample_count
            ),
            estimated_state=PlaybackState.DRAINING_TO_BOUNDARY,
        )

    def issue_resume(
        self,
        generation_id: int,
        causal_event_id: str,
        causal_source: CausalSource,
        stream_epoch: int,
        turn_epoch: int,
        confidence: float,
        now_monotonic_time_ns: int | None = None,
    ) -> PlaybackCommandEvent | None:
        now_ns = now_monotonic_time_ns or time.perf_counter_ns()
        if not self.can_resume(generation_id, now_ns):
            return None
        return self._issue(
            generation_id=generation_id,
            action=PlaybackCommandAction.RESUME,
            causal_event_id=causal_event_id,
            causal_source=causal_source,
            stream_epoch=stream_epoch,
            turn_epoch=turn_epoch,
            confidence=confidence,
            target_gain=1.0,
            gain_ramp_duration_ms=self.config.resume_ramp_duration_ms,
            maximum_paused_age_ms=self.config.maximum_resumable_paused_age_ms,
            estimated_state=PlaybackState.RESUMING,
            issued_monotonic_time_ns=now_ns,
        )

    def issue_cancel(
        self,
        generation_id: int,
        causal_event_id: str,
        causal_source: CausalSource,
        stream_epoch: int,
        turn_epoch: int,
        confidence: float,
    ) -> PlaybackCommandEvent:
        return self._issue(
            generation_id=generation_id,
            action=PlaybackCommandAction.CANCEL,
            causal_event_id=causal_event_id,
            causal_source=causal_source,
            stream_epoch=stream_epoch,
            turn_epoch=turn_epoch,
            confidence=confidence,
            estimated_state=PlaybackState.CANCELLED,
        )

    def acknowledge(
        self,
        event: PlaybackCommandAcknowledgementEvent,
        received_monotonic_time_ns: int | None = None,
    ) -> PlaybackAcknowledgementDisposition:
        record = self.command_records.get(event.command_id)
        if record is None:
            self.metrics.stale_acknowledgement_count += 1
            return PlaybackAcknowledgementDisposition.STALE
        if record.acknowledgement is not None:
            if event != record.acknowledgement:
                raise ValueError("A duplicate playback acknowledgement changed its result.")
            self.metrics.duplicate_acknowledgement_count += 1
            return PlaybackAcknowledgementDisposition.DUPLICATE
        command = record.command
        if (
            event.generation_id != command.generation_id
            or event.action is not command.action
            or event.stream_epoch != command.stream_epoch
            or event.turn_epoch != command.turn_epoch
        ):
            raise ValueError("Playback acknowledgement does not match its command.")
        received_ns = received_monotonic_time_ns or time.perf_counter_ns()
        record.acknowledgement = event
        record.acknowledgement_received_monotonic_time_ns = received_ns
        self.metrics.record_acknowledgement(record, event, received_ns)
        if event.generation_id != self.active_generation_id:
            self.metrics.stale_acknowledgement_count += 1
            return PlaybackAcknowledgementDisposition.STALE
        self._set_condition_from_browser(
            generation_id=event.generation_id,
            state=event.resulting_state,
            rendered_output_sample_position=event.rendered_output_sample_position,
            source_sample_position=event.source_sample_position,
            output_sample_rate=event.output_sample_rate,
        )
        if event.resulting_state is PlaybackState.PAUSED_BUFFERED:
            self.paused_at_server_monotonic_time_ns = received_ns
        elif event.resulting_state in (
            PlaybackState.RESUMING,
            PlaybackState.SPEAKING,
            PlaybackState.CANCELLED,
            PlaybackState.COMPLETED,
        ):
            self.paused_at_server_monotonic_time_ns = None
        return PlaybackAcknowledgementDisposition.APPLIED

    def can_resume(self, generation_id: int, now_monotonic_time_ns: int) -> bool:
        if generation_id != self.active_generation_id:
            return False
        if self.condition.state in (PlaybackState.DUCKING, PlaybackState.DRAINING_TO_BOUNDARY):
            return True
        if self.condition.state is not PlaybackState.PAUSED_BUFFERED:
            return False
        paused_at_ns = self.paused_at_server_monotonic_time_ns
        if paused_at_ns is None:
            return False
        paused_age_ns = now_monotonic_time_ns - paused_at_ns
        if paused_age_ns < 0:
            raise AssertionError("Playback resume time preceded the pause acknowledgement.")
        return paused_age_ns <= self.config.maximum_resumable_paused_age_ms * 1_000_000

    def _issue(
        self,
        generation_id: int,
        action: PlaybackCommandAction,
        causal_event_id: str,
        causal_source: CausalSource,
        stream_epoch: int,
        turn_epoch: int,
        confidence: float,
        estimated_state: PlaybackState,
        requested_boundary_source_sample_position: int | None = None,
        rendered_output_sample_deadline: int | None = None,
        target_gain: float | None = None,
        gain_ramp_duration_ms: int | None = None,
        maximum_paused_age_ms: int | None = None,
        issued_monotonic_time_ns: int | None = None,
    ) -> PlaybackCommandEvent:
        if generation_id != self.active_generation_id:
            raise ValueError("Playback commands must target the active generation.")
        command = PlaybackCommandEvent(
            command_id=str(uuid4()),
            generation_id=generation_id,
            action=action,
            issued_monotonic_time_ns=issued_monotonic_time_ns or time.perf_counter_ns(),
            causal_event_id=causal_event_id,
            causal_source=causal_source,
            stream_epoch=stream_epoch,
            turn_epoch=turn_epoch,
            confidence=confidence,
            requested_boundary_source_sample_position=(requested_boundary_source_sample_position),
            rendered_output_sample_deadline=rendered_output_sample_deadline,
            target_gain=target_gain,
            gain_ramp_duration_ms=gain_ramp_duration_ms,
            maximum_paused_age_ms=maximum_paused_age_ms,
        )
        self.command_records[command.command_id] = PlaybackCommandRecord(
            command=command,
            estimated_rendered_output_sample_position=(
                self.condition.latest_output_sample_position
            ),
        )
        self.metrics.record_command()
        self._set_condition(
            generation_id=generation_id,
            state=estimated_state,
            rendered_output_sample_position=self.condition.latest_output_sample_position,
            source_sample_position=self.condition.latest_source_sample_position,
            output_sample_rate=self.condition.output_sample_rate,
            authority=PlaybackConditionAuthority.SERVER_ESTIMATED,
        )
        return command

    def _set_condition_from_browser(
        self,
        generation_id: int,
        state: PlaybackState,
        rendered_output_sample_position: int,
        source_sample_position: int,
        output_sample_rate: int,
    ) -> None:
        self._set_condition(
            generation_id=generation_id,
            state=state,
            rendered_output_sample_position=rendered_output_sample_position,
            source_sample_position=source_sample_position,
            output_sample_rate=output_sample_rate,
            authority=PlaybackConditionAuthority.BROWSER_AUTHORITATIVE,
        )

    def _set_condition(
        self,
        generation_id: int | None,
        state: PlaybackState,
        rendered_output_sample_position: int,
        source_sample_position: int,
        output_sample_rate: int | None,
        authority: PlaybackConditionAuthority,
    ) -> None:
        self.condition = PlaybackCondition(
            event_id=str(uuid4()),
            generation_id=generation_id,
            state=state,
            assistant_audible=state
            in (
                PlaybackState.SPEAKING,
                PlaybackState.DUCKING,
                PlaybackState.RESUMING,
                PlaybackState.DRAINING_TO_BOUNDARY,
            ),
            latest_output_sample_position=rendered_output_sample_position,
            latest_source_sample_position=source_sample_position,
            output_sample_rate=output_sample_rate,
            monotonic_time_ns=time.perf_counter_ns(),
            authority=authority,
        )


def _percentile(values: list[float] | list[int], percentile: float) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(math.ceil(percentile * len(ordered)), 1)
    return ordered[rank - 1]
