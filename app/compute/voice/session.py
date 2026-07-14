from __future__ import annotations

import asyncio
import contextlib
import struct
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from app.compute.voice.conversation import ConversationMessage, ConversationRole
from app.compute.voice.interfaces import (
    LanguageModel,
    SpeechDetector,
    SpeechSynthesizer,
    Transcriber,
    TranscriptionSession,
)
from app.compute.voice.schemas import (
    AssistantAudioBoundaryEvent,
    AssistantTextDeltaEvent,
    ErrorEvent,
    PlaybackCompleteEvent,
    SessionReadyEvent,
    SessionStartEvent,
    SessionStopEvent,
    SpeechStateEvent,
    TranscriptEvent,
    VoiceServerEvent,
    VoiceServerEventType,
    voice_client_event_adapter,
)
from app.compute.voice.sentence_chunking import SentenceTextChunker

INPUT_SAMPLE_RATE = 16_000
PCM_BYTES_PER_SAMPLE = 2
AUDIO_QUEUE_MAX_CHUNKS = 100


@dataclass(frozen=True)
class SessionPolicy:
    # This silence begins after Silero's configured 250 ms end-of-speech decision.
    silence_duration_ms: int = 500
    pre_roll_duration_ms: int = 300


@dataclass
class ActiveGeneration:
    generation_id: int
    task: asyncio.Task[None] | None = None
    response_text: str | None = None
    generation_finished: bool = False
    playback_complete: bool = False


class VoiceSession:
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
        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=AUDIO_QUEUE_MAX_CHUNKS
        )
        self.send_lock = asyncio.Lock()
        self.conversation: list[ConversationMessage] = []
        self.active_generation: ActiveGeneration | None = None
        self.next_generation_id = 1
        self.audio_sample_count = 0
        self.started = False

    async def run(self) -> None:
        await self.websocket.accept()
        receive_task = asyncio.create_task(self._receive_loop())
        recognition_task = asyncio.create_task(self._recognition_loop())
        try:
            await asyncio.gather(receive_task, recognition_task)
        except WebSocketDisconnect:
            pass
        except Exception as error:
            await self._send_error(f"Voice session failed: {error}")
        finally:
            receive_task.cancel()
            recognition_task.cancel()
            for task in (receive_task, recognition_task):
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await task
            await self._cancel_generation(send_event=False)
            self.conversation.clear()

    async def _receive_loop(self) -> None:
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect
            pcm_bytes = message.get("bytes")
            if pcm_bytes is not None:
                if not self.started:
                    raise ValueError("session.start must be sent before microphone audio.")
                if len(pcm_bytes) % PCM_BYTES_PER_SAMPLE != 0:
                    raise ValueError("PCM16 audio frames must contain complete samples.")
                await self.audio_queue.put(pcm_bytes)
                continue
            event_json = message.get("text")
            if event_json is None:
                continue
            event = voice_client_event_adapter.validate_json(event_json)
            match event:
                case SessionStartEvent():
                    await self._start(event)
                case PlaybackCompleteEvent():
                    self._complete_playback(event.generation_id)
                case SessionStopEvent():
                    await self.audio_queue.put(None)
                    return

    async def _start(self, event: SessionStartEvent) -> None:
        if self.started:
            raise ValueError("session.start may only be sent once.")
        if event.input_sample_rate != INPUT_SAMPLE_RATE:
            raise ValueError(f"Input sample rate must be {INPUT_SAMPLE_RATE} Hz.")
        self.started = True
        await self._send_event(
            SessionReadyEvent(
                session_id=self.session_id,
                input_sample_rate=INPUT_SAMPLE_RATE,
                output_sample_rate=self.speech_synthesizer.sample_rate,
            )
        )

    async def _recognition_loop(self) -> None:
        transcription = self.transcriber.start_session()
        speech_active = False
        silent_samples = 0
        pre_roll_chunks: deque[bytes] = deque()
        pre_roll_samples = 0
        required_silent_samples = _milliseconds_to_samples(self.policy.silence_duration_ms)
        maximum_pre_roll_samples = _milliseconds_to_samples(self.policy.pre_roll_duration_ms)
        try:
            while (pcm_bytes := await self.audio_queue.get()) is not None:
                sample_count = _pcm_sample_count(pcm_bytes)
                self.audio_sample_count += sample_count
                is_speech = self.speech_detector.process_audio(pcm_bytes)
                if not speech_active:
                    pre_roll_chunks.append(pcm_bytes)
                    pre_roll_samples += sample_count
                    while pre_roll_samples > maximum_pre_roll_samples:
                        pre_roll_samples -= _pcm_sample_count(pre_roll_chunks.popleft())
                    if not is_speech:
                        continue
                    speech_active = True
                    silent_samples = 0
                    await self._cancel_generation(send_event=True)
                    await self._send_speech_state(VoiceServerEventType.VAD_STARTED)
                    for pre_roll_chunk in pre_roll_chunks:
                        await self._add_transcription_audio(transcription, pre_roll_chunk)
                    pre_roll_chunks.clear()
                    pre_roll_samples = 0
                    continue

                await self._add_transcription_audio(transcription, pcm_bytes)
                if is_speech:
                    silent_samples = 0
                    continue
                silent_samples += sample_count
                if silent_samples < required_silent_samples:
                    continue
                speech_active = False
                silent_samples = 0
                await self._send_speech_state(VoiceServerEventType.VAD_STOPPED)
                final_text = (await transcription.finish()).strip()
                await transcription.close()
                transcription = self.transcriber.start_session()
                if final_text:
                    await self._commit_turn(final_text)
        finally:
            await transcription.close()

    async def _add_transcription_audio(
        self,
        transcription: TranscriptionSession,
        pcm_bytes: bytes,
    ) -> None:
        partial_text = await transcription.add_audio(pcm_bytes)
        if partial_text:
            await self._send_transcript(VoiceServerEventType.TRANSCRIPT_PARTIAL, partial_text)

    async def _commit_turn(self, text: str) -> None:
        await self._send_transcript(VoiceServerEventType.TRANSCRIPT_FINAL, text)
        await self._send_transcript(VoiceServerEventType.TURN_COMMITTED, text)
        self.conversation.append(ConversationMessage(role=ConversationRole.USER, content=text))
        await self._cancel_generation(send_event=True)
        generation = ActiveGeneration(generation_id=self.next_generation_id)
        self.next_generation_id += 1
        generation.task = asyncio.create_task(self._run_generation(generation))
        self.active_generation = generation

    async def _run_generation(self, generation: ActiveGeneration) -> None:
        try:
            await self._generate_response(generation)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self.active_generation is generation:
                self.active_generation = None
                await self._send_audio_boundary(
                    VoiceServerEventType.ASSISTANT_CANCEL,
                    generation.generation_id,
                )
            await self._send_error(f"Response generation failed: {error}")

    async def _generate_response(self, generation: ActiveGeneration) -> None:
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def sentences() -> AsyncIterator[str]:
            while (sentence := await sentence_queue.get()) is not None:
                yield sentence

        audio_task = asyncio.create_task(
            self._stream_speech(
                generation_id=generation.generation_id,
                sentences=sentences(),
            )
        )
        sentence_chunker = SentenceTextChunker()
        response_parts: list[str] = []
        try:
            async for text_delta in self.language_model.stream_response(tuple(self.conversation)):
                response_parts.append(text_delta)
                await self._send_event(
                    AssistantTextDeltaEvent(
                        generation_id=generation.generation_id,
                        text=text_delta,
                    )
                )
                for sentence in sentence_chunker.add_text(text_delta):
                    await sentence_queue.put(sentence)
            response_text = "".join(response_parts).strip()
            if not response_text:
                raise RuntimeError("The language model returned an empty response.")
            for sentence in sentence_chunker.finish():
                await sentence_queue.put(sentence)
            await sentence_queue.put(None)
            await audio_task
            generation.response_text = response_text
            generation.generation_finished = True
            self._commit_assistant_if_complete(generation)
        finally:
            if not audio_task.done():
                audio_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await audio_task

    async def _stream_speech(
        self,
        generation_id: int,
        sentences: AsyncIterator[str],
    ) -> None:
        sequence_number = 0
        started = False
        async for sentence in sentences:
            async for pcm_bytes in self.speech_synthesizer.stream_audio(sentence):
                if not started:
                    started = True
                    await self._send_audio_boundary(
                        VoiceServerEventType.ASSISTANT_AUDIO_START,
                        generation_id,
                    )
                header = struct.pack("<II", generation_id, sequence_number)
                await self._send_audio(header + pcm_bytes)
                sequence_number += 1
        if started:
            await self._send_audio_boundary(
                VoiceServerEventType.ASSISTANT_AUDIO_END,
                generation_id,
            )
            return
        raise RuntimeError("The speech synthesizer returned no audio.")

    async def _cancel_generation(self, send_event: bool) -> None:
        generation = self.active_generation
        if generation is None:
            return
        self.active_generation = None
        if generation.task is not None and not generation.task.done():
            generation.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await generation.task
        if send_event:
            await self._send_audio_boundary(
                VoiceServerEventType.ASSISTANT_CANCEL,
                generation.generation_id,
            )

    def _complete_playback(self, generation_id: int) -> None:
        generation = self.active_generation
        if generation is None or generation.generation_id != generation_id:
            return
        generation.playback_complete = True
        self._commit_assistant_if_complete(generation)

    def _commit_assistant_if_complete(self, generation: ActiveGeneration) -> None:
        if not generation.generation_finished or not generation.playback_complete:
            return
        assert generation.response_text is not None
        if self.active_generation is not generation:
            return
        self.conversation.append(
            ConversationMessage(
                role=ConversationRole.ASSISTANT,
                content=generation.response_text,
            )
        )
        self.active_generation = None

    async def _send_speech_state(self, event_type: VoiceServerEventType) -> None:
        audio_time_ms = self.audio_sample_count * 1_000 // INPUT_SAMPLE_RATE
        await self._send_event(SpeechStateEvent(type=event_type, audio_time_ms=audio_time_ms))

    async def _send_transcript(self, event_type: VoiceServerEventType, text: str) -> None:
        await self._send_event(TranscriptEvent(type=event_type, text=text))

    async def _send_audio_boundary(
        self,
        event_type: VoiceServerEventType,
        generation_id: int,
    ) -> None:
        await self._send_event(
            AssistantAudioBoundaryEvent(type=event_type, generation_id=generation_id)
        )

    async def _send_error(self, message: str) -> None:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await self._send_event(ErrorEvent(message=message))

    async def _send_event(self, event: VoiceServerEvent) -> None:
        async with self.send_lock:
            await self.websocket.send_text(event.model_dump_json())

    async def _send_audio(self, frame: bytes) -> None:
        async with self.send_lock:
            await self.websocket.send_bytes(frame)


def _pcm_sample_count(pcm_bytes: bytes) -> int:
    return len(pcm_bytes) // PCM_BYTES_PER_SAMPLE


def _milliseconds_to_samples(duration_ms: int) -> int:
    return duration_ms * INPUT_SAMPLE_RATE // 1_000
