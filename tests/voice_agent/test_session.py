from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from app.compute.voice.interfaces import TranscriptionSession
from app.compute.voice.session import SessionPolicy, VoiceSession
from app.shared.compute_api import ConversationMessage, ConversationRole

SPEECH_CHUNK = b"\x01\x00" * 320
SILENCE_CHUNK = b"\x00\x00" * 320
DEFAULT_TEST_POLICY = SessionPolicy(silence_duration_ms=40, pre_roll_duration_ms=20)


class FakeSpeechDetector:
    def process_audio(self, pcm_bytes: bytes) -> bool:
        return any(pcm_bytes)


class RecordingTranscriber:
    def __init__(self) -> None:
        self.sessions: list[FakeTranscriptionSession] = []

    def start_session(self) -> TranscriptionSession:
        session = FakeTranscriptionSession()
        self.sessions.append(session)
        return session


class FakeTranscriptionSession:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.closed = False

    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        self.audio.append(pcm_bytes)
        return "hello" if any(pcm_bytes) else None

    async def finish(self) -> str:
        return "hello agent"

    async def close(self) -> None:
        self.closed = True


class FakeLanguageModel:
    def __init__(self) -> None:
        self.conversations: list[tuple[ConversationMessage, ...]] = []

    async def stream_response(
        self,
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]:
        self.conversations.append(conversation)
        yield "One two three four "
        yield "five six seven eight."


class FakeSpeechSynthesizer:
    @property
    def sample_rate(self) -> int:
        return 24_000

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        assert text == "One two three four five six seven eight."
        yield b"\x01\x00\x02\x00"


class SlowLanguageModel:
    def __init__(self) -> None:
        self.conversations: list[tuple[ConversationMessage, ...]] = []

    async def stream_response(
        self,
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]:
        self.conversations.append(conversation)
        yield "Still generating"
        await asyncio.sleep(10)


class SlowSpeechSynthesizer:
    @property
    def sample_rate(self) -> int:
        return 24_000

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        del text
        yield b"\x01\x00\x02\x00"
        await asyncio.sleep(10)


def test_full_session_streams_binary_audio_and_commits_played_history() -> None:
    language_model = FakeLanguageModel()
    transcriber = RecordingTranscriber()
    web_app = create_test_app(transcriber, language_model, FakeSpeechSynthesizer())

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        ready = websocket.receive_json()
        assert ready["type"] == "session.ready"
        assert ready["output_sample_rate"] == 24_000

        send_turn(websocket)
        first_events, audio_frame = receive_until(websocket, "assistant.audio.end")
        websocket.send_json({"type": "playback.complete", "generation_id": 1})

        send_turn(websocket)
        second_events, _ = receive_until(websocket, "assistant.audio.end")
        websocket.send_json({"type": "playback.complete", "generation_id": 2})
        websocket.send_json({"type": "session.stop"})

    event_types = [event["type"] for event in first_events]
    assert "vad.started" in event_types
    assert "vad.stopped" in event_types
    assert "transcript.partial" in event_types
    assert "transcript.final" in event_types
    assert "turn.committed" in event_types
    assert "assistant.text.delta" in event_types
    assert "assistant.audio.start" in event_types
    assert second_events[-1]["generation_id"] == 2
    assert audio_frame is not None
    assert struct.unpack("<II", audio_frame[:8]) == (1, 0)
    assert audio_frame[8:] == b"\x01\x00\x02\x00"
    assert language_model.conversations == [
        (ConversationMessage(role=ConversationRole.USER, content="hello agent"),),
        (
            ConversationMessage(role=ConversationRole.USER, content="hello agent"),
            ConversationMessage(
                role=ConversationRole.ASSISTANT,
                content="One two three four five six seven eight.",
            ),
            ConversationMessage(role=ConversationRole.USER, content="hello agent"),
        ),
    ]
    assert transcriber.sessions[0].audio == [SPEECH_CHUNK, SILENCE_CHUNK, SILENCE_CHUNK]
    assert all(session.closed for session in transcriber.sessions)


def test_user_speech_cancels_generation_and_does_not_commit_assistant_tail() -> None:
    language_model = SlowLanguageModel()
    transcriber = RecordingTranscriber()
    web_app = create_test_app(transcriber, language_model, SlowSpeechSynthesizer())

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        assert websocket.receive_json()["type"] == "session.ready"
        send_turn(websocket)
        receive_until(websocket, "assistant.text.delta")

        websocket.send_bytes(SPEECH_CHUNK)
        events, _ = receive_until(websocket, "assistant.cancel")
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "turn.committed")
        receive_until(websocket, "assistant.text.delta")
        websocket.send_json({"type": "session.stop"})

    assert events[-1] == {"type": "assistant.cancel", "generation_id": 1}
    assert len(language_model.conversations) == 2
    assert language_model.conversations[1] == (
        ConversationMessage(role=ConversationRole.USER, content="hello agent"),
        ConversationMessage(role=ConversationRole.USER, content="hello agent"),
    )


def test_user_speech_cancels_tts_and_omits_partially_played_response() -> None:
    language_model = FakeLanguageModel()
    transcriber = RecordingTranscriber()
    web_app = create_test_app(transcriber, language_model, SlowSpeechSynthesizer())

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")

        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "assistant.cancel")
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        receive_until(websocket, "assistant.text.delta")
        websocket.send_json({"type": "session.stop"})

    assert language_model.conversations == [
        (ConversationMessage(role=ConversationRole.USER, content="hello agent"),),
        (
            ConversationMessage(role=ConversationRole.USER, content="hello agent"),
            ConversationMessage(role=ConversationRole.USER, content="hello agent"),
        ),
    ]


def test_user_speech_clears_audio_that_is_still_queued_in_browser() -> None:
    language_model = FakeLanguageModel()
    transcriber = RecordingTranscriber()
    web_app = create_test_app(transcriber, language_model, FakeSpeechSynthesizer())

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.end")

        websocket.send_bytes(SPEECH_CHUNK)
        events, _ = receive_until(websocket, "assistant.cancel")
        websocket.send_json({"type": "session.stop"})

    assert events[-1] == {"type": "assistant.cancel", "generation_id": 1}
    assert language_model.conversations == [
        (ConversationMessage(role=ConversationRole.USER, content="hello agent"),)
    ]


def test_pre_roll_is_bounded_before_speech_start() -> None:
    transcriber = RecordingTranscriber()
    web_app = create_test_app(
        transcriber,
        FakeLanguageModel(),
        FakeSpeechSynthesizer(),
        policy=SessionPolicy(silence_duration_ms=20, pre_roll_duration_ms=40),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "vad.started")
        websocket.send_json({"type": "session.stop"})

    # The 40 ms limit retains one 20 ms silence chunk plus the speech-start chunk.
    assert transcriber.sessions[0].audio == [SILENCE_CHUNK, SPEECH_CHUNK]


def test_disconnect_closes_asr_and_cancels_active_generation() -> None:
    transcriber = RecordingTranscriber()
    web_app = create_test_app(transcriber, SlowLanguageModel(), SlowSpeechSynthesizer())

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.text.delta")

    assert all(session.closed for session in transcriber.sessions)


def create_test_app(
    transcriber: RecordingTranscriber,
    language_model: FakeLanguageModel | SlowLanguageModel,
    speech_synthesizer: FakeSpeechSynthesizer | SlowSpeechSynthesizer,
    policy: SessionPolicy = DEFAULT_TEST_POLICY,
) -> FastAPI:
    web_app = FastAPI()

    @web_app.websocket("/session")
    async def endpoint(websocket: WebSocket) -> None:
        session = VoiceSession(
            websocket=websocket,
            speech_detector=FakeSpeechDetector(),
            transcriber=transcriber,
            language_model=language_model,
            speech_synthesizer=speech_synthesizer,
            policy=policy,
        )
        await session.run()

    return web_app


def send_turn(websocket: WebSocketTestSession) -> None:
    websocket.send_bytes(SPEECH_CHUNK)
    websocket.send_bytes(SILENCE_CHUNK)
    websocket.send_bytes(SILENCE_CHUNK)


def receive_until(
    websocket: WebSocketTestSession,
    terminal_event_type: str,
) -> tuple[list[dict[str, str | int]], bytes | None]:
    events: list[dict[str, str | int]] = []
    audio_frame: bytes | None = None
    while not events or events[-1]["type"] != terminal_event_type:
        message = websocket.receive()
        if message.get("bytes") is not None:
            audio_frame = message["bytes"]
        elif message.get("text") is not None:
            events.append(json.loads(message["text"]))
    return events, audio_frame
