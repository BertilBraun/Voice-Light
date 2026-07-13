from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator

from app.local.compute_voice import (
    RemoteLanguageModel,
    RemoteSpeechSynthesizer,
    RemoteTranscriptionSession,
    compute_websocket_url,
)
from app.shared.compute_api import (
    AsrFinalEvent,
    AsrPartialEvent,
    LanguageModelDeltaEvent,
    LanguageModelEndEvent,
    SpeechAudioEvent,
    SpeechEndEvent,
    SpeechStartEvent,
    VoiceClientEvent,
    VoiceServerEvent,
)


class RecordingChannel:
    def __init__(self, events: tuple[VoiceServerEvent, ...]) -> None:
        self.events = list(events)
        self.sent: list[VoiceClientEvent] = []
        self.registered_operation_id: str | None = None
        self.output_sample_rate: int | None = 24_000

    def register_operation(self, operation_id: str) -> None:
        self.registered_operation_id = operation_id

    def unregister_operation(self, operation_id: str) -> None:
        assert operation_id == self.registered_operation_id

    async def send(self, event: VoiceClientEvent) -> None:
        self.sent.append(event)

    async def next_event(self, operation_id: str) -> VoiceServerEvent:
        assert operation_id == self.registered_operation_id
        return self.events.pop(0)

    def pending_event(self, operation_id: str) -> VoiceServerEvent | None:
        assert operation_id == self.registered_operation_id
        if self.events and isinstance(self.events[0], AsrPartialEvent):
            return self.events.pop(0)
        return None


def test_compute_websocket_url_preserves_base_path() -> None:
    assert compute_websocket_url("https://gpu.example/api/") == ("wss://gpu.example/api/v1/voice")


def test_remote_transcription_streams_partial_and_final_text() -> None:
    asyncio.run(run_remote_transcription_test())


async def run_remote_transcription_test() -> None:
    operation_id = "operation"
    channel = RecordingChannel(
        (
            AsrPartialEvent(operation_id=operation_id, text="hello"),
            AsrFinalEvent(
                operation_id=operation_id,
                text="hello world",
                inference_time_seconds=0.1,
            ),
        )
    )
    session = RemoteTranscriptionSession(channel)
    session.operation_id = operation_id
    channel.registered_operation_id = operation_id

    partial = await session.add_audio(b"\x01\x00")
    final = await session.finish()

    assert partial == "hello"
    assert final == "hello world"
    assert [event.type for event in channel.sent] == [
        "asr.start",
        "asr.audio",
        "asr.finish",
    ]


def test_remote_language_model_sends_complete_history() -> None:
    asyncio.run(run_remote_language_model_test())


async def run_remote_language_model_test() -> None:
    operation_id = "operation"
    channel = RecordingChannel(
        (
            LanguageModelDeltaEvent(operation_id=operation_id, text="Hi"),
            LanguageModelEndEvent(operation_id=operation_id, inference_time_seconds=0.1),
        )
    )
    language_model = RemoteLanguageModel(channel)

    stream = language_model.stream_response(
        (("user", "Hello"), ("assistant", "Hi"), ("user", "Again"))
    )
    text = await collect_text(stream)

    assert text == "Hi"
    command = channel.sent[0]
    assert command.type == "llm.generate"
    assert [message.content for message in command.conversation] == [
        "Hello",
        "Hi",
        "Again",
    ]


def test_remote_speech_synthesizer_decodes_incremental_audio() -> None:
    asyncio.run(run_remote_speech_synthesizer_test())


async def run_remote_speech_synthesizer_test() -> None:
    operation_id = "operation"
    channel = RecordingChannel(
        (
            SpeechStartEvent(operation_id=operation_id),
            SpeechAudioEvent(
                operation_id=operation_id,
                audio_base64=base64.b64encode(b"\x01\x00").decode("ascii"),
            ),
            SpeechEndEvent(
                operation_id=operation_id,
                inference_time_seconds=0.1,
                audio_duration_seconds=0.01,
            ),
        )
    )
    synthesizer = RemoteSpeechSynthesizer(channel)

    chunks = [chunk async for chunk in synthesizer.stream_audio("Hello")]

    assert chunks == [b"\x01\x00"]
    assert synthesizer.sample_rate == 24_000


async def collect_text(stream: AsyncIterator[str]) -> str:
    return "".join([text async for text in stream])
