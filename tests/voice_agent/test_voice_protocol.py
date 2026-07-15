from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.compute.voice.schemas import (
    PlaybackCompleteEvent,
    PlaybackProgressEvent,
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
            '{"type":"playback.complete","generation_id":4}',
            PlaybackCompleteEvent(generation_id=4),
        ),
        (
            '{"type":"playback.progress","generation_id":4,"text_offset":12,'
            '"boundary_start_sample":2400,"played_sample_count":2401}',
            PlaybackProgressEvent(
                generation_id=4,
                text_offset=12,
                boundary_start_sample=2_400,
                played_sample_count=2_401,
            ),
        ),
        (
            '{"type":"playback.stopped","generation_id":4,"text_offset":12,'
            '"played_sample_count":2401}',
            PlaybackStoppedEvent(
                generation_id=4,
                text_offset=12,
                played_sample_count=2_401,
            ),
        ),
    ],
)
def test_client_event_protocol_parses_discriminated_events(
    payload: str,
    expected_event: (
        SessionStartEvent
        | SessionStopEvent
        | PlaybackCompleteEvent
        | PlaybackProgressEvent
        | PlaybackStoppedEvent
    ),
) -> None:
    assert voice_client_event_adapter.validate_json(payload) == expected_event


@pytest.mark.parametrize(
    "payload",
    [
        '{"type":"asr.start","operation_id":"legacy"}',
        '{"type":"playback.complete","generation_id":0}',
        '{"type":"playback.progress","generation_id":1,"text_offset":0,'
        '"boundary_start_sample":0,"played_sample_count":1}',
        '{"type":"playback.progress","generation_id":1,"text_offset":2,'
        '"boundary_start_sample":0,"played_sample_count":0}',
        '{"type":"playback.stopped","generation_id":1,"text_offset":-1,"played_sample_count":0}',
    ],
)
def test_client_event_protocol_rejects_legacy_and_invalid_events(payload: str) -> None:
    with pytest.raises(ValidationError):
        voice_client_event_adapter.validate_json(payload)
