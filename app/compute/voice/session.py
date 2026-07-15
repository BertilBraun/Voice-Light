from __future__ import annotations

import asyncio
import contextlib
import struct
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from app.compute.voice.conversation import ConversationMessage, ConversationRole
from app.compute.voice.interfaces import (
    LanguageModel,
    SpeechDetector,
    SpeechSynthesisSession,
    SpeechSynthesizer,
    SynthesisEvent,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
    Transcriber,
    TranscriptionSession,
)
from app.compute.voice.schemas import (
    AssistantAudioBoundaryEvent,
    AssistantAudioTextBoundaryEvent,
    AssistantTextDeltaEvent,
    ErrorEvent,
    LlmHistoryEvent,
    LlmHistoryMessage,
    PlaybackCompleteEvent,
    PlaybackProgressEvent,
    PlaybackStoppedEvent,
    SessionReadyEvent,
    SessionStartEvent,
    SessionStopEvent,
    SpeechStateEvent,
    TranscriptEvent,
    VoiceServerEvent,
    VoiceServerEventType,
    voice_client_event_adapter,
)
from app.compute.voice.word_stream import CompleteWordStream

INPUT_SAMPLE_RATE = 16_000
PCM_BYTES_PER_SAMPLE = 2
AUDIO_QUEUE_MAX_CHUNKS = 100
PLAYBACK_STOP_TIMEOUT_SECONDS = 0.25


@dataclass(frozen=True)
class SessionPolicy:
    # This silence begins after Silero's configured 250 ms end-of-speech decision.
    silence_duration_ms: int = 500
    pre_roll_duration_ms: int = 300


@dataclass
class ActiveGeneration:
    generation_id: int
    prompt_messages: tuple[ConversationMessage, ...]
    task: asyncio.Task[None] | None = None
    response_text: str = ""
    acknowledged_offset: int = 0
    history_index: int | None = None
    boundary_samples: dict[int, int] = field(default_factory=dict)
    generation_finished: bool = False
    playback_complete: bool = False
    cancelled: bool = False
    accepts_playback: bool = True
    playback_stopped: asyncio.Event = field(default_factory=asyncio.Event)


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
        self.generations: dict[int, ActiveGeneration] = {}
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
            await self._close_generation_tasks()
            self.conversation.clear()
            self.generations.clear()

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
                case PlaybackProgressEvent():
                    self._acknowledge_playback(event)
                case PlaybackStoppedEvent():
                    self._stop_playback(event)
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
        await self._finalize_cancelled_playback()
        self.conversation.append(ConversationMessage(role=ConversationRole.USER, content=text))
        await self._cancel_generation(send_event=True)
        generation = ActiveGeneration(
            generation_id=self.next_generation_id,
            prompt_messages=tuple(self.conversation),
        )
        self.generations[generation.generation_id] = generation
        self.next_generation_id += 1
        await self._send_event(
            LlmHistoryEvent(
                generation_id=generation.generation_id,
                messages=tuple(
                    LlmHistoryMessage(role=message.role, content=message.content)
                    for message in generation.prompt_messages
                ),
            )
        )
        generation.task = asyncio.create_task(self._run_generation(generation))
        self.active_generation = generation

    async def _finalize_cancelled_playback(self) -> None:
        pending = tuple(
            generation
            for generation in self.generations.values()
            if generation.cancelled and generation.accepts_playback
        )
        for generation in pending:
            try:
                await asyncio.wait_for(
                    generation.playback_stopped.wait(),
                    timeout=PLAYBACK_STOP_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                generation.accepts_playback = False
                self._mark_generation_interrupted(generation)

    async def _run_generation(self, generation: ActiveGeneration) -> None:
        try:
            await self._generate_response(generation)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self.active_generation is generation:
                generation.cancelled = True
                self._mark_generation_interrupted(generation)
                self.active_generation = None
                await self._send_audio_boundary(
                    VoiceServerEventType.ASSISTANT_CANCEL,
                    generation.generation_id,
                )
            await self._send_error(f"Response generation failed: {error}")

    async def _generate_response(self, generation: ActiveGeneration) -> None:
        synthesis = self.speech_synthesizer.start_session()
        text_task = asyncio.create_task(self._stream_response_text(generation, synthesis))
        audio_task = asyncio.create_task(self._stream_speech(generation, synthesis.stream_events()))
        try:
            await asyncio.gather(text_task, audio_task)
            generation.generation_finished = True
            self._commit_assistant_if_complete(generation)
        finally:
            pending_tasks = tuple(task for task in (text_task, audio_task) if not task.done())
            for task in pending_tasks:
                task.cancel()
            await synthesis.cancel()
            for task in pending_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _stream_response_text(
        self,
        generation: ActiveGeneration,
        synthesis: SpeechSynthesisSession,
    ) -> None:
        word_stream = CompleteWordStream()
        async for text_delta in self.language_model.stream_response(generation.prompt_messages):
            generation.response_text += text_delta
            await self._send_event(
                AssistantTextDeltaEvent(
                    generation_id=generation.generation_id,
                    text=text_delta,
                )
            )
            for word in word_stream.add_text(text_delta):
                await synthesis.add_word(word)
        if not generation.response_text.strip():
            raise RuntimeError("The language model returned an empty response.")
        for word in word_stream.finish():
            await synthesis.add_word(word)
        await synthesis.finish_input()

    async def _stream_speech(
        self,
        generation: ActiveGeneration,
        events: AsyncIterator[SynthesisEvent],
    ) -> None:
        sequence_number = 0
        expected_start_sample = 0
        started = False
        async for event in events:
            if generation.cancelled:
                return
            match event:
                case SynthesizedWordBoundary():
                    if event.text_offset > len(generation.response_text):
                        raise ValueError("TTS returned a text boundary beyond generated text.")
                    generation.boundary_samples[event.text_offset] = event.start_sample
                    await self._send_event(
                        AssistantAudioTextBoundaryEvent(
                            generation_id=generation.generation_id,
                            text_offset=event.text_offset,
                            start_sample=event.start_sample,
                        )
                    )
                case SynthesizedAudioChunk():
                    if len(event.pcm_bytes) % PCM_BYTES_PER_SAMPLE != 0:
                        raise ValueError("TTS PCM16 frames must contain complete samples.")
                    if event.start_sample != expected_start_sample:
                        raise ValueError("TTS audio chunks must have contiguous sample offsets.")
                    if not started:
                        started = True
                        await self._send_audio_boundary(
                            VoiceServerEventType.ASSISTANT_AUDIO_START,
                            generation.generation_id,
                        )
                    header = struct.pack(
                        "<III",
                        generation.generation_id,
                        sequence_number,
                        event.start_sample,
                    )
                    await self._send_audio(header + event.pcm_bytes)
                    sequence_number += 1
                    expected_start_sample += _pcm_sample_count(event.pcm_bytes)
        if not started:
            raise RuntimeError("The speech synthesizer returned no audio.")
        await self._send_audio_boundary(
            VoiceServerEventType.ASSISTANT_AUDIO_END,
            generation.generation_id,
        )

    async def _cancel_generation(self, send_event: bool) -> None:
        generation = self.active_generation
        if generation is None:
            return
        generation.cancelled = True
        self._mark_generation_interrupted(generation)
        self.active_generation = None
        if send_event:
            await self._send_audio_boundary(
                VoiceServerEventType.ASSISTANT_CANCEL,
                generation.generation_id,
            )
        if generation.task is not None and not generation.task.done():
            generation.task.cancel()
            await asyncio.sleep(0)

    async def _close_generation_tasks(self) -> None:
        tasks = tuple(
            generation.task
            for generation in self.generations.values()
            if generation.task is not None and not generation.task.done()
        )
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _complete_playback(self, generation_id: int) -> None:
        generation = self.active_generation
        if generation is None or generation.generation_id != generation_id:
            return
        generation.playback_complete = True
        generation.accepts_playback = False
        self._commit_assistant_if_complete(generation)

    def _acknowledge_playback(self, event: PlaybackProgressEvent) -> None:
        generation = self.generations.get(event.generation_id)
        if generation is None or not generation.accepts_playback:
            return
        boundary_sample = generation.boundary_samples.get(event.text_offset)
        if boundary_sample != event.boundary_start_sample:
            return
        if event.played_sample_count <= boundary_sample:
            return
        if event.text_offset <= generation.acknowledged_offset:
            return
        self._update_assistant_history(generation, event.text_offset)

    def _stop_playback(self, event: PlaybackStoppedEvent) -> None:
        generation = self.generations.get(event.generation_id)
        if generation is None or not generation.cancelled or not generation.accepts_playback:
            return
        if event.text_offset > 0:
            boundary_sample = generation.boundary_samples.get(event.text_offset)
            if boundary_sample is None or event.played_sample_count <= boundary_sample:
                return
            if event.text_offset > generation.acknowledged_offset:
                self._update_assistant_history(generation, event.text_offset)
        generation.accepts_playback = False
        generation.playback_stopped.set()
        self._mark_generation_interrupted(generation)

    def _update_assistant_history(
        self,
        generation: ActiveGeneration,
        text_offset: int,
    ) -> None:
        assert 0 < text_offset <= len(generation.response_text)
        content = generation.response_text[:text_offset].rstrip()
        assert content
        if generation.cancelled:
            content = f"{content}..."
        message = ConversationMessage(role=ConversationRole.ASSISTANT, content=content)
        if generation.history_index is None:
            generation.history_index = len(self.conversation)
            self.conversation.append(message)
        else:
            self.conversation[generation.history_index] = message
        generation.acknowledged_offset = text_offset

    def _mark_generation_interrupted(self, generation: ActiveGeneration) -> None:
        if generation.history_index is None:
            return
        message = self.conversation[generation.history_index]
        if message.content.endswith("..."):
            return
        self.conversation[generation.history_index] = ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=f"{message.content}...",
        )

    def _commit_assistant_if_complete(self, generation: ActiveGeneration) -> None:
        if not generation.generation_finished or not generation.playback_complete:
            return
        if self.active_generation is not generation:
            return
        complete_message = ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=generation.response_text.strip(),
        )
        if generation.history_index is None:
            self.conversation.append(complete_message)
        else:
            self.conversation[generation.history_index] = complete_message
        generation.accepts_playback = False
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
