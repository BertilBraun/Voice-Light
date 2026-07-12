from __future__ import annotations

import json
import struct
from collections.abc import AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from app.voice_agent.interfaces import TranscriptionSession
from app.voice_agent.session import SessionPolicy, VoiceAgentSession


class FakeSpeechDetector:
    def process_audio(self, pcm_bytes: bytes) -> bool:
        return pcm_bytes != b"\x00\x00"


class FakeTranscriber:
    def start_session(self) -> TranscriptionSession:
        return FakeTranscriptionSession()


class FakeTranscriptionSession:
    async def add_audio(self, pcm_bytes: bytes) -> str | None:
        return "hello" if pcm_bytes != b"\x00\x00" else None

    async def finish(self) -> str:
        return "hello agent"

    async def close(self) -> None:
        pass


class FakeLanguageModel:
    async def stream_response(
        self,
        conversation: tuple[tuple[str, str], ...],
    ) -> AsyncIterator[str]:
        assert conversation == (("user", "hello agent"),)
        yield "One two three four "
        yield "five six seven eight."


class FakeSpeechSynthesizer:
    @property
    def sample_rate(self) -> int:
        return 24_000

    async def stream_audio(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        async for text in text_chunks:
            assert text == "One two three four five six seven eight."
            yield b"\x01\x00\x02\x00"


def test_full_session_streams_text_and_framed_audio() -> None:
    web_app = FastAPI()

    @web_app.websocket("/session")
    async def endpoint(websocket: WebSocket) -> None:
        session = VoiceAgentSession(
            websocket=websocket,
            speech_detector=FakeSpeechDetector(),
            transcriber=FakeTranscriber(),
            language_model=FakeLanguageModel(),
            speech_synthesizer=FakeSpeechSynthesizer(),
            policy=SessionPolicy(silence_duration_ms=40, audio_frame_duration_ms=20),
        )
        await session.run()

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        assert websocket.receive_json()["type"] == "session.ready"

        websocket.send_bytes(b"\x01\x00")
        websocket.send_bytes(b"\x00\x00")
        websocket.send_bytes(b"\x00\x00")

        messages: list[str] = []
        audio_frame: bytes | None = None
        while "assistant.audio.end" not in messages:
            message = websocket.receive()
            if message.get("bytes") is not None:
                audio_frame = message["bytes"]
            elif message.get("text") is not None:
                messages.append(json.loads(message["text"])["type"])

        websocket.send_json({"type": "session.stop"})

    assert "vad.started" in messages
    assert "vad.stopped" in messages
    assert "turn.committed" in messages
    assert "assistant.text.delta" in messages
    assert "assistant.audio.start" in messages
    assert audio_frame is not None
    assert struct.unpack("<II", audio_frame[:8]) == (1, 0)
    assert audio_frame[8:] == b"\x01\x00\x02\x00"
