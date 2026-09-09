from __future__ import annotations

import asyncio
import struct

import pytest

from app.compute.voice.schemas import PlaybackClockEvent, PlaybackStartedEvent
from deployment.compute.benchmark_voice_session import PlaybackDrainState


class RecordingPlaybackSender:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, message: str) -> None:
        self.messages.append(message)


def audio_frame(
    generation_id: int,
    sequence_number: int,
    start_sample: int,
    sample_count: int,
) -> bytes:
    return struct.pack("<III", generation_id, sequence_number, start_sample) + bytes(
        sample_count * 2
    )


def test_playback_drain_acknowledges_every_contiguous_audio_chunk() -> None:
    async def exercise() -> tuple[PlaybackDrainState, RecordingPlaybackSender]:
        state = PlaybackDrainState()
        state.configure(24_000)
        sender = RecordingPlaybackSender()
        assert await state.consume_audio(sender, audio_frame(7, 0, 0, 480)) == 7
        assert await state.consume_audio(sender, audio_frame(7, 1, 480, 240)) == 7
        return state, sender

    state, sender = asyncio.run(exercise())

    started = PlaybackStartedEvent.model_validate_json(sender.messages[0])
    first_clock = PlaybackClockEvent.model_validate_json(sender.messages[1])
    second_clock = PlaybackClockEvent.model_validate_json(sender.messages[2])
    completed = state.complete_event(7)
    assert started.source_sample_position == 0
    assert started.output_sample_rate == 24_000
    assert first_clock.source_sample_position == 480
    assert second_clock.source_sample_position == 720
    assert second_clock.rendered_output_sample_position == 720
    assert completed.source_sample_position == 720
    assert completed.rendered_output_sample_position == 720
    assert completed.output_sample_rate == 24_000


@pytest.mark.parametrize(
    ("first_frame", "second_frame", "message"),
    (
        (audio_frame(7, 0, 0, 10), audio_frame(8, 1, 10, 10), "changed generation"),
        (audio_frame(7, 0, 0, 10), audio_frame(7, 2, 10, 10), "sequence numbers"),
        (audio_frame(7, 0, 0, 10), audio_frame(7, 1, 11, 10), "sample positions"),
    ),
)
def test_playback_drain_rejects_discontinuous_audio(
    first_frame: bytes,
    second_frame: bytes,
    message: str,
) -> None:
    async def exercise() -> None:
        state = PlaybackDrainState()
        state.configure(24_000)
        sender = RecordingPlaybackSender()
        await state.consume_audio(sender, first_frame)
        with pytest.raises(ValueError, match=message):
            await state.consume_audio(sender, second_frame)

    asyncio.run(exercise())


def test_playback_drain_rejects_audio_before_ready_and_mismatched_completion() -> None:
    async def exercise() -> PlaybackDrainState:
        state = PlaybackDrainState()
        sender = RecordingPlaybackSender()
        with pytest.raises(RuntimeError, match="before session readiness"):
            await state.consume_audio(sender, audio_frame(7, 0, 0, 10))
        state.configure(24_000)
        await state.consume_audio(sender, audio_frame(7, 0, 0, 10))
        return state

    state = asyncio.run(exercise())
    with pytest.raises(ValueError, match="must match"):
        state.complete_event(8)
