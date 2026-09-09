from __future__ import annotations

import asyncio

import pytest

from app.compute.voice.playback import (
    PlaybackAcknowledgementDisposition,
    PlaybackController,
    PlaybackFlowControl,
    PlaybackPolicyConfig,
)
from app.compute.voice.schemas import (
    CausalSource,
    PlaybackClockEvent,
    PlaybackCommandAcknowledgementEvent,
    PlaybackCommandAction,
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
    action: PlaybackCommandAction,
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


@pytest.mark.parametrize(
    ("late_rendered_output_sample_position", "late_source_sample_position"),
    [(140, 70), (220, 110)],
)
def test_older_acknowledgement_cannot_rewind_authoritative_playback_truth(
    late_rendered_output_sample_position: int,
    late_source_sample_position: int,
) -> None:
    controller = PlaybackController(24_000, PlaybackPolicyConfig())
    controller.replace_generation(1)
    controller.record_started(_started_event(1))
    duck = controller.issue_duck(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
    )
    pause = controller.issue_pause(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
        requested_boundary_source_sample_position=100,
    )
    assert (
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
        is PlaybackAcknowledgementDisposition.APPLIED
    )
    authoritative_condition = controller.condition

    assert (
        controller.acknowledge(
            _acknowledgement(
                duck.command_id,
                duck.action,
                PlaybackState.DUCKING,
                rendered_output_sample_position=late_rendered_output_sample_position,
                source_sample_position=late_source_sample_position,
            ),
            received_monotonic_time_ns=10**18 + 1,
        )
        is PlaybackAcknowledgementDisposition.STALE
    )
    assert controller.condition == authoritative_condition
    report = controller.metrics.report()
    assert report.acknowledgement_count == 2
    assert report.stale_acknowledgement_count == 1


def test_newer_acknowledgement_cannot_regress_authoritative_sample_positions() -> None:
    controller = PlaybackController(24_000, PlaybackPolicyConfig())
    controller.replace_generation(1)
    controller.record_started(_started_event(1))
    duck = controller.issue_duck(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
    )
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
            duck.command_id,
            duck.action,
            PlaybackState.DUCKING,
            rendered_output_sample_position=200,
            source_sample_position=100,
        ),
        received_monotonic_time_ns=10**18,
    )
    authoritative_condition = controller.condition

    assert (
        controller.acknowledge(
            _acknowledgement(
                pause.command_id,
                pause.action,
                PlaybackState.PAUSED_BUFFERED,
                rendered_output_sample_position=140,
                source_sample_position=70,
            ),
            received_monotonic_time_ns=10**18 + 1,
        )
        is PlaybackAcknowledgementDisposition.STALE
    )
    assert controller.condition == authoritative_condition


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


def test_transport_flow_control_waits_for_periodic_browser_credit() -> None:
    async def exercise() -> None:
        flow_control = PlaybackFlowControl(source_sample_rate=1_000, maximum_ahead_ms=500)
        flow_control.replace_generation(1)

        await flow_control.wait_until_send_allowed(1, 500)
        blocked_send = asyncio.create_task(flow_control.wait_until_send_allowed(1, 600))
        await asyncio.sleep(0)
        assert not blocked_send.done()

        assert flow_control.observe(1, 80)
        await asyncio.sleep(0)
        assert not blocked_send.done()
        assert flow_control.observe(1, 100)
        await blocked_send

        assert not flow_control.observe(2, 1_000)
        assert flow_control.acknowledged_source_sample_position == 100

        blocked_after_credit = asyncio.create_task(flow_control.wait_until_send_allowed(1, 700))
        await asyncio.sleep(0)
        assert not blocked_after_credit.done()
        assert flow_control.invalidate_generation(1)
        assert await blocked_after_credit is False

        flow_control.replace_generation(2)
        assert await flow_control.wait_until_send_allowed(1, 1) is False

    asyncio.run(exercise())


def test_transport_ahead_window_is_configurable_from_environment() -> None:
    configuration = PlaybackPolicyConfig.from_environment(
        {"VOICE_LIGHT_MAXIMUM_TRANSPORT_AHEAD_MS": "640"}
    )

    assert configuration.maximum_transport_ahead_ms == 640


def test_default_transport_ahead_covers_intermittent_streaming_synthesis() -> None:
    assert PlaybackPolicyConfig().maximum_transport_ahead_ms == 1_200


@pytest.mark.parametrize("value", ("soon", "0"))
def test_transport_ahead_window_rejects_invalid_environment(value: str) -> None:
    with pytest.raises(ValueError, match="transport|TRANSPORT"):
        PlaybackPolicyConfig.from_environment({"VOICE_LIGHT_MAXIMUM_TRANSPORT_AHEAD_MS": value})


def test_periodic_clock_updates_authoritative_playback_position() -> None:
    controller = PlaybackController(24_000, PlaybackPolicyConfig())
    controller.replace_generation(1)
    controller.record_started(_started_event(1))

    assert controller.record_clock(
        PlaybackClockEvent(
            generation_id=1,
            state=PlaybackState.SPEAKING,
            browser_monotonic_time_ns=2,
            rendered_output_sample_position=4_000,
            source_sample_position=2_000,
            queued_source_sample_count=12_000,
            output_sample_rate=48_000,
        )
    )
    assert controller.condition.latest_source_sample_position == 2_000
    assert controller.metrics.report().maximum_buffered_source_sample_count == 12_000

    controller.issue_pause(
        generation_id=1,
        causal_event_id="silero-1",
        causal_source=CausalSource.SILERO_VAD,
        stream_epoch=1,
        turn_epoch=1,
        confidence=1.0,
        requested_boundary_source_sample_position=3_000,
    )
    assert controller.record_clock(
        PlaybackClockEvent(
            generation_id=1,
            state=PlaybackState.SPEAKING,
            browser_monotonic_time_ns=3,
            rendered_output_sample_position=4_100,
            source_sample_position=2_050,
            queued_source_sample_count=11_950,
            output_sample_rate=48_000,
        )
    )
    assert controller.condition.state is PlaybackState.DRAINING_TO_BOUNDARY

    assert not controller.record_clock(
        PlaybackClockEvent(
            generation_id=1,
            state=PlaybackState.SPEAKING,
            browser_monotonic_time_ns=1,
            rendered_output_sample_position=3_000,
            source_sample_position=1_500,
            queued_source_sample_count=14_000,
            output_sample_rate=48_000,
        )
    )
    assert controller.condition.latest_source_sample_position == 2_050
