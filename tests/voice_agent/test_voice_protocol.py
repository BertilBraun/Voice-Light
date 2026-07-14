from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.compute.voice.schemas import (
    PlaybackCompleteEvent,
    PlaybackProgressEvent,
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
            '{"type":"playback.progress","generation_id":4,"sentence_id":0,"text_offset":12}',
            PlaybackProgressEvent(generation_id=4, sentence_id=0, text_offset=12),
        ),
    ],
)
def test_client_event_protocol_parses_discriminated_events(
    payload: str,
    expected_event: (
        SessionStartEvent | SessionStopEvent | PlaybackCompleteEvent | PlaybackProgressEvent
    ),
) -> None:
    assert voice_client_event_adapter.validate_json(payload) == expected_event


@pytest.mark.parametrize(
    "payload",
    [
        '{"type":"asr.start","operation_id":"legacy"}',
        '{"type":"playback.complete","generation_id":0}',
        '{"type":"playback.progress","generation_id":1,"sentence_id":-1,"text_offset":2}',
        '{"type":"playback.progress","generation_id":1,"sentence_id":0,"text_offset":0}',
    ],
)
def test_client_event_protocol_rejects_legacy_and_invalid_events(payload: str) -> None:
    with pytest.raises(ValidationError):
        voice_client_event_adapter.validate_json(payload)
