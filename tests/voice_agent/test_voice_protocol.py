from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.compute.voice.schemas import (
    PlaybackCommandAcknowledgementEvent,
    PlaybackCommandAction,
    PlaybackCompleteEvent,
    PlaybackPauseResult,
    PlaybackProgressEvent,
    PlaybackStartedEvent,
    PlaybackState,
    PlaybackStoppedEvent,
    SessionStartEvent,
    SessionStopEvent,
    voice_client_event_adapter,
)


@pytest.mark.parametrize(
    ("payload", "expected_event"),
    [
        (
            '{"type":"session.start","input_sample_rate":16000}',
            SessionStartEvent(input_sample_rate=16_000),
        ),
        ('{"type":"session.stop"}', SessionStopEvent()),
        (
            '{"type":"playback.started","generation_id":4,"browser_monotonic_time_ns":10,'
            '"rendered_output_sample_position":1,"source_sample_position":2,'
            '"output_sample_rate":48000}',
            PlaybackStartedEvent(
                generation_id=4,
                browser_monotonic_time_ns=10,
                rendered_output_sample_position=1,
                source_sample_position=2,
                output_sample_rate=48_000,
            ),
        ),
        (
            '{"type":"playback.complete","generation_id":4,"browser_monotonic_time_ns":10,'
            '"rendered_output_sample_position":4800,"source_sample_position":2400,'
            '"output_sample_rate":48000}',
            PlaybackCompleteEvent(
                generation_id=4,
                browser_monotonic_time_ns=10,
                rendered_output_sample_position=4_800,
                source_sample_position=2_400,
                output_sample_rate=48_000,
            ),
        ),
        (
            '{"type":"playback.progress","generation_id":4,"text_offset":12,'
            '"boundary_start_sample":2400,"played_sample_count":2401,'
            '"browser_monotonic_time_ns":10,"rendered_output_sample_position":4802,'
            '"output_sample_rate":48000}',
            PlaybackProgressEvent(
                generation_id=4,
                text_offset=12,
                boundary_start_sample=2_400,
                played_sample_count=2_401,
                browser_monotonic_time_ns=10,
                rendered_output_sample_position=4_802,
                output_sample_rate=48_000,
            ),
        ),
        (
            '{"type":"playback.progress","generation_id":4,"text_offset":2,'
            '"boundary_start_sample":0,"played_sample_count":0,'
            '"browser_monotonic_time_ns":10,"rendered_output_sample_position":0,'
            '"output_sample_rate":48000}',
            PlaybackProgressEvent(
                generation_id=4,
                text_offset=2,
                boundary_start_sample=0,
                played_sample_count=0,
                browser_monotonic_time_ns=10,
                rendered_output_sample_position=0,
                output_sample_rate=48_000,
            ),
        ),
        (
            '{"type":"playback.stopped","generation_id":4,"text_offset":12,'
            '"played_sample_count":2401,"browser_monotonic_time_ns":10,'
            '"rendered_output_sample_position":4802,"output_sample_rate":48000}',
            PlaybackStoppedEvent(
                generation_id=4,
                text_offset=12,
                played_sample_count=2_401,
                browser_monotonic_time_ns=10,
                rendered_output_sample_position=4_802,
                output_sample_rate=48_000,
            ),
        ),
        (
            '{"type":"playback.acknowledgement","command_id":"command-1",'
            '"generation_id":4,"action":"pause_at_boundary","stream_epoch":1,'
            '"turn_epoch":2,"resulting_state":"paused_buffered",'
            '"browser_monotonic_time_ns":10,"rendered_output_sample_position":4802,'
            '"source_sample_position":2401,"output_sample_rate":48000,'
            '"pause_result":"word_boundary","current_gain":0.1258925,'
            '"gain_ramp_complete":true,"queued_source_sample_count":800,'
            '"discarded_source_sample_count":0,"replayed_source_sample_count":0,'
            '"skipped_source_sample_count":0,"resume_rejected":false}',
            PlaybackCommandAcknowledgementEvent(
                command_id="command-1",
                generation_id=4,
                action=PlaybackCommandAction.PAUSE_AT_BOUNDARY,
                stream_epoch=1,
                turn_epoch=2,
                resulting_state=PlaybackState.PAUSED_BUFFERED,
                browser_monotonic_time_ns=10,
                rendered_output_sample_position=4_802,
                source_sample_position=2_401,
                output_sample_rate=48_000,
                pause_result=PlaybackPauseResult.WORD_BOUNDARY,
                current_gain=0.1258925,
                gain_ramp_complete=True,
                queued_source_sample_count=800,
                discarded_source_sample_count=0,
                replayed_source_sample_count=0,
                skipped_source_sample_count=0,
                resume_rejected=False,
            ),
        ),
    ],
)
def test_client_event_protocol_parses_discriminated_events(
    payload: str,
    expected_event: (
        SessionStartEvent
        | SessionStopEvent
        | PlaybackStartedEvent
        | PlaybackCompleteEvent
        | PlaybackProgressEvent
        | PlaybackStoppedEvent
        | PlaybackCommandAcknowledgementEvent
    ),
) -> None:
    assert voice_client_event_adapter.validate_json(payload) == expected_event


@pytest.mark.parametrize(
    "payload",
    [
        '{"type":"asr.start","operation_id":"legacy"}',
        '{"type":"playback.started","generation_id":0}',
        '{"type":"playback.complete","generation_id":0}',
        '{"type":"playback.progress","generation_id":1,"text_offset":0,'
        '"boundary_start_sample":0,"played_sample_count":1}',
        '{"type":"playback.stopped","generation_id":1,"text_offset":-1,"played_sample_count":0}',
    ],
)
def test_client_event_protocol_rejects_legacy_and_invalid_events(payload: str) -> None:
    with pytest.raises(ValidationError):
        voice_client_event_adapter.validate_json(payload)
