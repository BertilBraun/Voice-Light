from __future__ import annotations

from app.compute.voice.playback import (
    PlaybackAcknowledgementDisposition,
    PlaybackController,
    PlaybackPolicyConfig,
)
from app.compute.voice.schemas import (
    CausalSource,
    PlaybackCommandAcknowledgementEvent,
    PlaybackPauseResult,
    PlaybackProgressEvent,
    PlaybackStartedEvent,
    PlaybackState,
)


def _started_event(generation_id: int) -> PlaybackStartedEvent:
    return PlaybackStartedEvent(
        generation_id=generation_id,
        browser_monotonic_time_ns=1,
        rendered_output_sample_position=128,
        source_sample_position=64,
        output_sample_rate=48_000,
    )


def _acknowledgement(
    command_id: str,
    action: str,
    state: PlaybackState,
    rendered_output_sample_position: int,
    source_sample_position: int,
) -> PlaybackCommandAcknowledgementEvent:
    return PlaybackCommandAcknowledgementEvent(
        command_id=command_id,
        generation_id=1,
        action=action,
        stream_epoch=1,
        turn_epoch=1,
        resulting_state=state,
        browser_monotonic_time_ns=2,
        rendered_output_sample_position=rendered_output_sample_position,
        source_sample_position=source_sample_position,
        output_sample_rate=48_000,
        pause_result=(
            PlaybackPauseResult.WORD_BOUNDARY
            if state is PlaybackState.PAUSED_BUFFERED
            else PlaybackPauseResult.NOT_REQUESTED
        ),
        current_gain=0.1258925,
        gain_ramp_complete=True,
        queued_source_sample_count=100,
        discarded_source_sample_count=0,
        replayed_source_sample_count=0,
        skipped_source_sample_count=0,
        resume_rejected=False,
    )


def test_control_commands_reconcile_server_estimates_with_browser_truth() -> None:
    controller = PlaybackController(24_000, PlaybackPolicyConfig())
    controller.replace_generation(1)
    assert controller.record_started(_started_event(1))
    assert controller.condition.authority.value == "browser_authoritative"

    duck = controller.issue_duck(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
    )
    assert controller.condition.state is PlaybackState.DUCKING
    assert controller.condition.authority.value == "server_estimated"
    duck_acknowledgement = _acknowledgement(
        duck.command_id,
        duck.action,
        PlaybackState.DUCKING,
        rendered_output_sample_position=140,
        source_sample_position=70,
    )
    assert (
        controller.acknowledge(duck_acknowledgement, received_monotonic_time_ns=10**18)
        is PlaybackAcknowledgementDisposition.APPLIED
    )
    assert controller.condition.latest_output_sample_position == 140
    assert controller.condition.latest_source_sample_position == 70
    assert controller.condition.authority.value == "browser_authoritative"


def test_pause_deadline_uses_browser_rendered_output_sample_rate() -> None:
    controller = PlaybackController(24_000, PlaybackPolicyConfig(pause_deadline_ms=120))
    controller.replace_generation(1)
    controller.record_started(_started_event(1))
    pause = controller.issue_pause(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
        requested_boundary_source_sample_position=100,
    )
    assert pause.rendered_output_sample_deadline == 128 + 5_760
    assert pause.requested_boundary_source_sample_position == 100


def test_duplicate_and_stale_acknowledgements_do_not_rewrite_playback_truth() -> None:
    controller = PlaybackController(24_000, PlaybackPolicyConfig())
    controller.replace_generation(1)
    controller.record_started(_started_event(1))
    command = controller.issue_duck(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
    )
    acknowledgement = _acknowledgement(
        command.command_id,
        command.action,
        PlaybackState.DUCKING,
        rendered_output_sample_position=140,
        source_sample_position=70,
    )
    controller.acknowledge(acknowledgement, received_monotonic_time_ns=10**18)
    assert (
        controller.acknowledge(acknowledgement, received_monotonic_time_ns=10**18 + 1)
        is PlaybackAcknowledgementDisposition.DUPLICATE
    )
    controller.replace_generation(2)
    stale_condition = controller.condition
    assert (
        controller.acknowledge(
            acknowledgement.model_copy(update={"command_id": "unknown"}),
            received_monotonic_time_ns=10**18 + 2,
        )
        is PlaybackAcknowledgementDisposition.STALE
    )
    assert controller.condition == stale_condition


def test_server_rejects_resume_after_maximum_paused_age() -> None:
    controller = PlaybackController(24_000, PlaybackPolicyConfig())
    controller.replace_generation(1)
    controller.record_started(_started_event(1))
    pause = controller.issue_pause(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
        requested_boundary_source_sample_position=100,
    )
    pause_acknowledgement = _acknowledgement(
        pause.command_id,
        pause.action,
        PlaybackState.PAUSED_BUFFERED,
        rendered_output_sample_position=200,
        source_sample_position=100,
    )
    paused_at_ns = 10**18
    controller.acknowledge(
        pause_acknowledgement,
        received_monotonic_time_ns=paused_at_ns,
    )

    assert (
        controller.issue_resume(
            generation_id=1,
            causal_event_id="decision-1",
            causal_source=CausalSource.FLOOR_POLICY,
            stream_epoch=1,
            turn_epoch=1,
            confidence=1.0,
            now_monotonic_time_ns=paused_at_ns + 801_000_000,
        )
        is None
    )


def test_boundary_progress_after_pause_does_not_reactivate_playback() -> None:
    controller = PlaybackController(24_000, PlaybackPolicyConfig())
    controller.replace_generation(1)
    controller.record_started(_started_event(1))
    pause = controller.issue_pause(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
        requested_boundary_source_sample_position=100,
    )
    controller.acknowledge(
        _acknowledgement(
            pause.command_id,
            pause.action,
            PlaybackState.PAUSED_BUFFERED,
            rendered_output_sample_position=200,
            source_sample_position=100,
        ),
        received_monotonic_time_ns=10**18,
    )

    assert controller.record_progress(
        PlaybackProgressEvent(
            generation_id=1,
            text_offset=3,
            boundary_start_sample=0,
            played_sample_count=100,
            browser_monotonic_time_ns=3,
            rendered_output_sample_position=200,
            output_sample_rate=48_000,
        )
    )
    assert controller.condition.state is PlaybackState.PAUSED_BUFFERED
