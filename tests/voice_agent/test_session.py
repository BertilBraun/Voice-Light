from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from app.compute.voice.conversation import ConversationMessage, ConversationRole
from app.compute.voice.interfaces import (
    SpeechSynthesisSession,
    SynthesisEvent,
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
    TranscriptionSession,
)
from app.compute.voice.session import SessionPolicy, VoiceSession

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


class SplitWordLanguageModel:
    async def stream_response(
        self,
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]:
        del conversation
        yield "  Hello"
        yield ", wor"
        yield "ld! Next"


class SlowLanguageModel:
    def __init__(self) -> None:
        self.conversations: list[tuple[ConversationMessage, ...]] = []

    async def stream_response(
        self,
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]:
        self.conversations.append(conversation)
        yield "One two three "
        await asyncio.sleep(10)


class CancellationTrackingLanguageModel:
    def __init__(self) -> None:
        self.active_generation_count = 0
        self.overlapped = False

    async def stream_response(
        self,
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]:
        del conversation
        self.active_generation_count += 1
        if self.active_generation_count > 1:
            self.overlapped = True
        try:
            yield "One two three "
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0.05)
            self.active_generation_count -= 1


class FailingLanguageModel:
    async def stream_response(
        self,
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]:
        del conversation
        raise RuntimeError("synthetic language failure")
        yield


class FakeSpeechSynthesisSession:
    def __init__(self, words: list[SynthesisWord]) -> None:
        self.words = words
        self.events: asyncio.Queue[SynthesisEvent | None] = asyncio.Queue()
        self.next_sample = 0
        self.finished = False

    async def add_word(self, word: SynthesisWord) -> None:
        self.words.append(word)
        await self.events.put(
            SynthesizedWordBoundary(text_offset=word.text_end, start_sample=self.next_sample)
        )
        await self.events.put(
            SynthesizedAudioChunk(
                pcm_bytes=b"\x01\x00\x02\x00",
                start_sample=self.next_sample,
            )
        )
        self.next_sample += 2

    async def finish_input(self) -> None:
        self.finished = True
        await self.events.put(None)

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        while (event := await self.events.get()) is not None:
            yield event

    async def cancel(self) -> None:
        if not self.finished:
            self.finished = True
            await self.events.put(None)


class RecordingSpeechSynthesizer:
    def __init__(self) -> None:
        self.sessions: list[FakeSpeechSynthesisSession] = []
        self.words: list[SynthesisWord] = []

    @property
    def sample_rate(self) -> int:
        return 24_000

    def start_session(self) -> SpeechSynthesisSession:
        session = FakeSpeechSynthesisSession(self.words)
        self.sessions.append(session)
        return session


class FailingSpeechSynthesisSession:
    def __init__(self) -> None:
        self.failure_ready = asyncio.Event()

    async def add_word(self, word: SynthesisWord) -> None:
        del word
        self.failure_ready.set()

    async def finish_input(self) -> None:
        self.failure_ready.set()

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        await self.failure_ready.wait()
        raise RuntimeError("synthetic speech failure")
        yield

    async def cancel(self) -> None:
        self.failure_ready.set()


class FailingSpeechSynthesizer:
    @property
    def sample_rate(self) -> int:
        return 24_000

    def start_session(self) -> SpeechSynthesisSession:
        return FailingSpeechSynthesisSession()


def test_full_session_streams_audio_and_commits_naturally_completed_history() -> None:
    language_model = FakeLanguageModel()
    transcriber = RecordingTranscriber()
    synthesizer = RecordingSpeechSynthesizer()
    web_app = create_test_app(transcriber, language_model, synthesizer)

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

    assert "assistant.audio.text_boundary" in [event["type"] for event in first_events]
    assert second_events[-1]["generation_id"] == 2
    assert audio_frame is not None
    assert struct.unpack("<III", audio_frame[:12]) == (1, 7, 14)
    assert audio_frame[12:] == b"\x01\x00\x02\x00"
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
    assert len(synthesizer.sessions) == 2
    assert synthesizer.sessions[0] is not synthesizer.sessions[1]


def test_words_are_forwarded_on_whitespace_and_trailing_word_is_flushed() -> None:
    synthesizer = RecordingSpeechSynthesizer()
    web_app = create_test_app(
        RecordingTranscriber(),
        SplitWordLanguageModel(),
        synthesizer,
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        events, _ = receive_until(websocket, "assistant.audio.end")
        websocket.send_json({"type": "session.stop"})

    assert synthesizer.words == [
        SynthesisWord(text="Hello,", text_start=2, text_end=8),
        SynthesisWord(text="world!", text_start=9, text_end=15),
        SynthesisWord(text="Next", text_start=16, text_end=20),
    ]
    boundaries = [event for event in events if event["type"] == "assistant.audio.text_boundary"]
    assert boundaries == [
        {
            "type": "assistant.audio.text_boundary",
            "generation_id": 1,
            "text_offset": 8,
            "start_sample": 0,
        },
        {
            "type": "assistant.audio.text_boundary",
            "generation_id": 1,
            "text_offset": 15,
            "start_sample": 2,
        },
        {
            "type": "assistant.audio.text_boundary",
            "generation_id": 1,
            "text_offset": 20,
            "start_sample": 4,
        },
    ]


def test_synthesis_failure_cancels_generation_and_reaches_client() -> None:
    web_app = create_test_app(
        RecordingTranscriber(),
        SlowLanguageModel(),
        FailingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        events, _ = receive_until(websocket, "error")
        websocket.send_json({"type": "session.stop"})

    assert [event["type"] for event in events[-2:]] == ["assistant.cancel", "error"]
    assert events[-1]["message"] == ("Response generation failed: synthetic speech failure")
    assert events[-1]["component"] == "speech_synthesis"
    assert events[-1]["operation"] == "stream_synthesis"
    assert events[-1]["generation_id"] == 1
    assert events[-1]["retryable"] is True


def test_successor_generation_waits_for_cancelled_generation_teardown() -> None:
    language_model = CancellationTrackingLanguageModel()
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "assistant.cancel")
        websocket.send_json(
            {
                "type": "playback.stopped",
                "generation_id": 1,
                "text_offset": 0,
                "played_sample_count": 0,
            }
        )
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "assistant.text.delta")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["generation_id"] == 2
    assert language_model.overlapped is False


def test_language_model_failure_reaches_client_with_component_context() -> None:
    web_app = create_test_app(
        RecordingTranscriber(),
        FailingLanguageModel(),
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        events, _ = receive_until(websocket, "error")
        websocket.send_json({"type": "session.stop"})

    assert events[-1] == {
        "type": "error",
        "component": "language_model",
        "operation": "generate_text",
        "generation_id": 1,
        "retryable": True,
        "message": "Response generation failed: synthetic language failure",
    }


def test_canceled_generation_accepts_final_browser_acknowledgement() -> None:
    language_model = SlowLanguageModel()
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")

        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "assistant.cancel")
        websocket.send_json(
            {
                "type": "playback.stopped",
                "generation_id": 1,
                "text_offset": 7,
                "played_sample_count": 3,
            }
        )
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["messages"] == [
        {"role": "user", "content": "hello agent"},
        {"role": "assistant", "content": "One two..."},
        {"role": "user", "content": "hello agent"},
    ]


def test_invalid_or_stale_playback_progress_is_ignored() -> None:
    language_model = SlowLanguageModel()
    web_app = create_test_app(
        RecordingTranscriber(),
        language_model,
        RecordingSpeechSynthesizer(),
    )

    with TestClient(web_app).websocket_connect("/session") as websocket:
        websocket.send_json({"type": "session.start", "input_sample_rate": 16_000})
        websocket.receive_json()
        send_turn(websocket)
        receive_until(websocket, "assistant.audio.start")
        websocket.send_json(
            {
                "type": "playback.progress",
                "generation_id": 1,
                "text_offset": 7,
                "boundary_start_sample": 99,
                "played_sample_count": 100,
            }
        )
        websocket.send_bytes(SPEECH_CHUNK)
        receive_until(websocket, "assistant.cancel")
        websocket.send_json(
            {
                "type": "playback.stopped",
                "generation_id": 1,
                "text_offset": 0,
                "played_sample_count": 0,
            }
        )
        websocket.send_bytes(SILENCE_CHUNK)
        websocket.send_bytes(SILENCE_CHUNK)
        events, _ = receive_until(websocket, "llm.history")
        websocket.send_json({"type": "session.stop"})

    assert events[-1]["messages"] == [
        {"role": "user", "content": "hello agent"},
        {"role": "user", "content": "hello agent"},
    ]


def test_pre_roll_is_bounded_before_speech_start() -> None:
    transcriber = RecordingTranscriber()
    web_app = create_test_app(
        transcriber,
        FakeLanguageModel(),
        RecordingSpeechSynthesizer(),
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

    assert transcriber.sessions[0].audio == [SILENCE_CHUNK, SPEECH_CHUNK]


def create_test_app(
    transcriber: RecordingTranscriber,
    language_model: (
        FakeLanguageModel
        | SplitWordLanguageModel
        | SlowLanguageModel
        | CancellationTrackingLanguageModel
        | FailingLanguageModel
    ),
    speech_synthesizer: RecordingSpeechSynthesizer | FailingSpeechSynthesizer,
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
) -> tuple[list[dict[str, object]], bytes | None]:
    events: list[dict[str, object]] = []
    audio_frame: bytes | None = None
    while not events or events[-1]["type"] != terminal_event_type:
        message = websocket.receive()
        if message.get("bytes") is not None:
            audio_frame = message["bytes"]
        elif message.get("text") is not None:
            events.append(json.loads(message["text"]))
    return events, audio_frame
