from __future__ import annotations

import asyncio
import contextlib
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from app.frozen_base_config import FrozenBaseModel
from app.voice_agent.interfaces import LanguageModel, SpeechDetector, SpeechSynthesizer, Transcriber
from app.voice_agent.schemas import (
    AssistantAudioBoundaryEvent,
    AssistantTextDeltaEvent,
    ClientEventType,
    ErrorEvent,
    ServerEventType,
    SessionReadyEvent,
    SessionStartEvent,
    SpeechStateEvent,
    TranscriptEvent,
    client_event_adapter,
)
from app.voice_agent.text_chunking import WordTextChunker

INPUT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SessionPolicy:
    silence_duration_ms: int = 400
    audio_frame_duration_ms: int = 20


class VoiceAgentSession:
    def __init__(
        self,
        websocket: WebSocket,
        speech_detector: SpeechDetector,
        transcriber: Transcriber,
        language_model: LanguageModel,
        speech_synthesizer: SpeechSynthesizer,
        policy: SessionPolicy,
    ) -> None:
        self.websocket = websocket
        self.speech_detector = speech_detector
        self.transcriber = transcriber
        self.language_model = language_model
        self.speech_synthesizer = speech_synthesizer
        self.policy = policy
        self.session_id = str(uuid4())
        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)
        self.conversation: list[tuple[str, str]] = []
        self.generation_task: asyncio.Task[None] | None = None
        self.generation_id = 0
        self.audio_time_ms = 0

    async def run(self) -> None:
        await self.websocket.accept()
        receive_task = asyncio.create_task(self._receive_loop())
        recognition_task = asyncio.create_task(self._recognition_loop())
        try:
            await asyncio.gather(receive_task, recognition_task)
        except WebSocketDisconnect:
            pass
        finally:
            receive_task.cancel()
            recognition_task.cancel()
            await self._cancel_generation()

    async def _receive_loop(self) -> None:
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                await self.audio_queue.put(None)
                return
            if message.get("bytes") is not None:
                await self.audio_queue.put(message["bytes"])
                continue
            text = message.get("text")
            if text is None:
                continue
            event = client_event_adapter.validate_json(text)
            match event.type:
                case ClientEventType.SESSION_START:
                    assert isinstance(event, SessionStartEvent)
                    if event.input_sample_rate != INPUT_SAMPLE_RATE:
                        raise ValueError(f"Input sample rate must be {INPUT_SAMPLE_RATE} Hz.")
                    await self._send_event(
                        SessionReadyEvent(
                            session_id=self.session_id,
                            input_sample_rate=INPUT_SAMPLE_RATE,
                            output_sample_rate=self.speech_synthesizer.sample_rate,
                        )
                    )
                case ClientEventType.SESSION_STOP:
                    await self.audio_queue.put(None)
                    return

    async def _recognition_loop(self) -> None:
        transcription = self.transcriber.start_session()
        speech_active = False
        silent_frames = 0
        required_silent_frames = (
            self.policy.silence_duration_ms // self.policy.audio_frame_duration_ms
        )
        try:
            while (pcm_bytes := await self.audio_queue.get()) is not None:
                self.audio_time_ms += self.policy.audio_frame_duration_ms
                is_speech = self.speech_detector.process_audio(pcm_bytes)
                if is_speech:
                    silent_frames = 0
                    if not speech_active:
                        speech_active = True
                        await self._send_speech_state(ServerEventType.VAD_STARTED)
                    partial_text = await transcription.add_audio(pcm_bytes)
                    if partial_text:
                        await self._send_transcript(
                            ServerEventType.TRANSCRIPT_PARTIAL, partial_text
                        )
                    continue
                if not speech_active:
                    continue
                silent_frames += 1
                partial_text = await transcription.add_audio(pcm_bytes)
                if partial_text:
                    await self._send_transcript(ServerEventType.TRANSCRIPT_PARTIAL, partial_text)
                if silent_frames < required_silent_frames:
                    continue
                speech_active = False
                silent_frames = 0
                await self._send_speech_state(ServerEventType.VAD_STOPPED)
                final_text = (await transcription.finish()).strip()
                await transcription.close()
                transcription = self.transcriber.start_session()
                if final_text:
                    await self._commit_turn(final_text)
        finally:
            await transcription.close()

    async def _commit_turn(self, text: str) -> None:
        await self._send_transcript(ServerEventType.TRANSCRIPT_FINAL, text)
        await self._send_transcript(ServerEventType.TURN_COMMITTED, text)
        self.conversation.append(("user", text))
        await self._cancel_generation()
        self.generation_id += 1
        self.generation_task = asyncio.create_task(self._generate_response(self.generation_id))

    async def _generate_response(self, generation_id: int) -> None:
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def text_chunks() -> AsyncIterator[str]:
            while (text := await text_queue.get()) is not None:
                yield text

        audio_task = asyncio.create_task(
            self._stream_speech(generation_id=generation_id, text_chunks=text_chunks())
        )
        chunker = WordTextChunker()
        response_parts: list[str] = []
        try:
            async for text_delta in self.language_model.stream_response(tuple(self.conversation)):
                response_parts.append(text_delta)
                await self._send_event(
                    AssistantTextDeltaEvent(generation_id=generation_id, text=text_delta)
                )
                for text_chunk in chunker.add_text(text_delta):
                    await text_queue.put(text_chunk.text)
            for text_chunk in chunker.finish():
                await text_queue.put(text_chunk.text)
            await text_queue.put(None)
            await audio_task
            self.conversation.append(("assistant", "".join(response_parts).strip()))
        except asyncio.CancelledError:
            await text_queue.put(None)
            audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await audio_task
            raise

    async def _stream_speech(
        self,
        generation_id: int,
        text_chunks: AsyncIterator[str],
    ) -> None:
        sequence_number = 0
        started = False
        async for pcm_bytes in self.speech_synthesizer.stream_audio(text_chunks):
            if not started:
                started = True
                await self._send_audio_boundary(
                    ServerEventType.ASSISTANT_AUDIO_START, generation_id
                )
            header = struct.pack("<II", generation_id, sequence_number)
            await self.websocket.send_bytes(header + pcm_bytes)
            sequence_number += 1
        if started:
            await self._send_audio_boundary(ServerEventType.ASSISTANT_AUDIO_END, generation_id)

    async def _cancel_generation(self) -> None:
        if self.generation_task is None or self.generation_task.done():
            return
        cancelled_generation_id = self.generation_id
        self.generation_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.generation_task
        await self._send_audio_boundary(ServerEventType.ASSISTANT_CANCEL, cancelled_generation_id)

    async def _send_speech_state(self, event_type: ServerEventType) -> None:
        await self._send_event(SpeechStateEvent(type=event_type, audio_time_ms=self.audio_time_ms))

    async def _send_transcript(self, event_type: ServerEventType, text: str) -> None:
        await self._send_event(TranscriptEvent(type=event_type, text=text))

    async def _send_audio_boundary(self, event_type: ServerEventType, generation_id: int) -> None:
        await self._send_event(
            AssistantAudioBoundaryEvent(type=event_type, generation_id=generation_id)
        )

    async def _send_event(self, event: FrozenBaseModel) -> None:
        await self.websocket.send_text(event.model_dump_json())


async def send_session_error(websocket: WebSocket, error: Exception) -> None:
    event = ErrorEvent(message=str(error))
    with contextlib.suppress(RuntimeError):
        await websocket.send_text(event.model_dump_json())
