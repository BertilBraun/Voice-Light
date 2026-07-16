from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from app.compute.voice.conversation import ConversationMessage, ConversationRole
from app.compute.voice.errors import VoiceComponent, VoiceComponentError, VoiceOperation
from app.compute.voice.interfaces import (
    LanguageModel,
    SpeechDetector,
    SpeechSynthesisSession,
    SpeechSynthesizer,
    SynthesisEvent,
    SynthesisFirstAudioMetrics,
    SynthesisWord,
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
    PlaybackStartedEvent,
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
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionPolicy:
    # This silence begins after Silero's configured 250 ms end-of-speech decision.
    silence_duration_ms: int = 500
    pre_roll_duration_ms: int = 300


class SessionLifecycle(StrEnum):
    CREATED = "created"
    CONNECTED = "connected"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"
    CLOSED = "closed"


class GenerationLifecycle(StrEnum):
    CREATED = "created"
    STREAMING = "streaming"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    STREAM_ENDED = "stream_ended"
    PLAYBACK_COMPLETE = "playback_complete"
    FAILED = "failed"


@dataclass
class GenerationLatency:
    asr_finalization_seconds: float
    turn_ready_at: float
    generation_started_at: float | None = None
    first_language_delta_at: float | None = None
    first_synthesis_word_at: float | None = None
    first_audio_at: float | None = None
    first_audio_sent_at: float | None = None
    playback_started_at: float | None = None
    synthesis_metrics: SynthesisFirstAudioMetrics | None = None


@dataclass
class ActiveGeneration:
    generation_id: int
    prompt_messages: tuple[ConversationMessage, ...]
    latency: GenerationLatency
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
    lifecycle: GenerationLifecycle = GenerationLifecycle.CREATED


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
        self.lifecycle = SessionLifecycle.CREATED
        self.pending_generation_teardown: ActiveGeneration | None = None

    async def run(self) -> None:
        await self.websocket.accept()
        self._transition_session(SessionLifecycle.CONNECTED)
        logger.info("voice session opened: session=%s", self.session_id)
        receive_task = asyncio.create_task(self._receive_loop())
        recognition_task = asyncio.create_task(self._recognition_loop())
        try:
            await asyncio.gather(receive_task, recognition_task)
        except WebSocketDisconnect:
            pass
        except Exception as error:
            self.lifecycle = SessionLifecycle.FAILED
            logger.exception("voice session failed: %s", self.session_id)
            failure = _component_error(
                error,
                component=VoiceComponent.SESSION,
                operation=VoiceOperation.SESSION_RUN,
            )
            await self._send_error(failure, generation_id=None, retryable=False)
        finally:
            self.lifecycle = SessionLifecycle.STOPPING
            receive_task.cancel()
            recognition_task.cancel()
            await self._request_generation_cancellation(send_event=False)
            await self._await_generation_teardown()
            await self._close_generation_tasks()
            for task in (receive_task, recognition_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # The session-level exception path already logged and reported this failure.
                    pass
            self.conversation.clear()
            self.generations.clear()
            self.lifecycle = SessionLifecycle.CLOSED
            logger.info("voice session closed: session=%s", self.session_id)

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
                case PlaybackStartedEvent():
                    self._record_playback_started(event.generation_id)
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
        self._transition_session(SessionLifecycle.READY)
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
                try:
                    is_speech = self.speech_detector.process_audio(pcm_bytes)
                except Exception as error:
                    raise VoiceComponentError(
                        VoiceComponent.SPEECH_DETECTION,
                        VoiceOperation.DETECT_SPEECH,
                        str(error),
                    ) from error
                if not speech_active:
                    pre_roll_chunks.append(pcm_bytes)
                    pre_roll_samples += sample_count
                    while pre_roll_samples > maximum_pre_roll_samples:
                        pre_roll_samples -= _pcm_sample_count(pre_roll_chunks.popleft())
                    if not is_speech:
                        continue
                    speech_active = True
                    silent_samples = 0
                    await self._request_generation_cancellation(send_event=True)
                    logger.info("speech started: session=%s", self.session_id)
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
                finalization_started_at = time.perf_counter()
                logger.info("transcription finalization started: session=%s", self.session_id)
                try:
                    final_text = (await transcription.finish()).strip()
                except Exception as error:
                    raise _component_error(
                        error,
                        component=VoiceComponent.ASR,
                        operation=VoiceOperation.TRANSCRIBE,
                    ) from error
                finalization_seconds = time.perf_counter() - finalization_started_at
                logger.info(
                    "transcription finalization completed: session=%s duration_seconds=%.3f "
                    "character_count=%d",
                    self.session_id,
                    finalization_seconds,
                    len(final_text),
                )
                await transcription.close()
                transcription = self.transcriber.start_session()
                if final_text:
                    await self._commit_turn(final_text, finalization_seconds)
        finally:
            await transcription.close()

    async def _add_transcription_audio(
        self,
        transcription: TranscriptionSession,
        pcm_bytes: bytes,
    ) -> None:
        try:
            partial_text = await transcription.add_audio(pcm_bytes)
        except Exception as error:
            raise _component_error(
                error,
                component=VoiceComponent.ASR,
                operation=VoiceOperation.TRANSCRIBE,
            ) from error
        if partial_text:
            await self._send_transcript(VoiceServerEventType.TRANSCRIPT_PARTIAL, partial_text)

    async def _commit_turn(self, text: str, asr_finalization_seconds: float) -> None:
        turn_ready_at = time.perf_counter()
        await self._send_transcript(VoiceServerEventType.TRANSCRIPT_FINAL, text)
        await self._send_transcript(VoiceServerEventType.TURN_COMMITTED, text)
        await self._finalize_cancelled_playback()
        await self._await_generation_teardown()
        self.conversation.append(ConversationMessage(role=ConversationRole.USER, content=text))
        assert self.active_generation is None
        generation = ActiveGeneration(
            generation_id=self.next_generation_id,
            prompt_messages=tuple(self.conversation),
            latency=GenerationLatency(
                asr_finalization_seconds=asr_finalization_seconds,
                turn_ready_at=turn_ready_at,
            ),
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
        generation.latency.generation_started_at = time.perf_counter()
        logger.info(
            "voice response generation started: session=%s generation=%d",
            self.session_id,
            generation.generation_id,
        )
        self._transition_generation(generation, GenerationLifecycle.STREAMING)
        try:
            await self._generate_response(generation)
            self._transition_generation(generation, GenerationLifecycle.STREAM_ENDED)
            if generation.playback_complete:
                self._transition_generation(generation, GenerationLifecycle.PLAYBACK_COMPLETE)
            logger.info(
                "voice response generation completed: session=%s generation=%d",
                self.session_id,
                generation.generation_id,
            )
        except asyncio.CancelledError:
            self._transition_generation(generation, GenerationLifecycle.CANCELLED)
            raise
        except Exception as error:
            self._transition_generation(generation, GenerationLifecycle.FAILED)
            logger.exception(
                "voice response generation failed: session=%s generation=%d",
                self.session_id,
                generation.generation_id,
            )
            if self.active_generation is generation:
                generation.cancelled = True
                self._mark_generation_interrupted(generation)
                self.active_generation = None
                await self._send_audio_boundary(
                    VoiceServerEventType.ASSISTANT_CANCEL,
                    generation.generation_id,
                )
            failure = _component_error(
                error,
                component=VoiceComponent.SESSION,
                operation=VoiceOperation.SESSION_RUN,
            )
            client_failure = VoiceComponentError(
                failure.component,
                failure.operation,
                f"Response generation failed: {failure}",
            )
            await self._send_error(
                client_failure,
                generation_id=generation.generation_id,
                retryable=True,
            )

    async def _generate_response(self, generation: ActiveGeneration) -> None:
        synthesis = self.speech_synthesizer.start_session()
        text_task = asyncio.create_task(self._stream_response_text(generation, synthesis))
        audio_task = asyncio.create_task(self._stream_speech(generation, synthesis.stream_events()))
        generation_error: BaseException | None = None
        try:
            await asyncio.gather(text_task, audio_task)
            generation.generation_finished = True
            self._commit_assistant_if_complete(generation)
        except BaseException as error:
            generation_error = error
            raise
        finally:
            deferred_cleanup_error: Exception | None = None
            pending_tasks = tuple(task for task in (text_task, audio_task) if not task.done())
            for task in pending_tasks:
                if task.cancelling() == 0:
                    task.cancel()
            try:
                await synthesis.cancel()
            except Exception as error:
                cleanup_error = VoiceComponentError(
                    VoiceComponent.SPEECH_SYNTHESIS,
                    VoiceOperation.STREAM_SYNTHESIS,
                    f"Speech synthesis cleanup failed: {error}",
                )
                if generation_error is None or isinstance(generation_error, asyncio.CancelledError):
                    deferred_cleanup_error = cleanup_error
                else:
                    logger.exception(
                        "speech synthesis cleanup failed after generation failure: "
                        "session=%s generation=%d",
                        self.session_id,
                        generation.generation_id,
                    )
            for task in pending_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as error:
                    if generation_error is None or isinstance(
                        generation_error, asyncio.CancelledError
                    ):
                        if deferred_cleanup_error is None:
                            deferred_cleanup_error = error
                            continue
                    logger.exception(
                        "generation child cleanup failed after another generation failure: "
                        "session=%s generation=%d",
                        self.session_id,
                        generation.generation_id,
                    )
            if deferred_cleanup_error is not None:
                raise deferred_cleanup_error

    async def _stream_response_text(
        self,
        generation: ActiveGeneration,
        synthesis: SpeechSynthesisSession,
    ) -> None:
        language_stream = self.language_model.stream_response(generation.prompt_messages)
        try:
            async with contextlib.aclosing(language_stream):
                await self._consume_response_text(generation, synthesis, language_stream)
        except VoiceComponentError:
            raise
        except Exception as error:
            raise VoiceComponentError(
                VoiceComponent.LANGUAGE_MODEL,
                VoiceOperation.GENERATE_TEXT,
                str(error),
            ) from error

    async def _consume_response_text(
        self,
        generation: ActiveGeneration,
        synthesis: SpeechSynthesisSession,
        language_stream: AsyncIterator[str],
    ) -> None:
        word_stream = CompleteWordStream()
        while True:
            try:
                text_delta = await anext(language_stream)
            except StopAsyncIteration:
                break
            except Exception as error:
                raise VoiceComponentError(
                    VoiceComponent.LANGUAGE_MODEL,
                    VoiceOperation.GENERATE_TEXT,
                    str(error),
                ) from error
            if text_delta and not generation.response_text:
                generation.latency.first_language_delta_at = time.perf_counter()
                logger.info(
                    "language model first delta: session=%s generation=%d",
                    self.session_id,
                    generation.generation_id,
                )
            generation.response_text += text_delta
            await self._send_event(
                AssistantTextDeltaEvent(
                    generation_id=generation.generation_id,
                    text=text_delta,
                )
            )
            for word in word_stream.add_text(text_delta):
                await self._add_synthesis_word(generation, synthesis, word)
        if not generation.response_text.strip():
            raise VoiceComponentError(
                VoiceComponent.LANGUAGE_MODEL,
                VoiceOperation.GENERATE_TEXT,
                "The language model returned an empty response.",
            )
        for word in word_stream.finish():
            await self._add_synthesis_word(generation, synthesis, word)
        try:
            await synthesis.finish_input()
        except Exception as error:
            raise VoiceComponentError(
                VoiceComponent.SPEECH_SYNTHESIS,
                VoiceOperation.STREAM_SYNTHESIS,
                str(error),
            ) from error

    async def _add_synthesis_word(
        self,
        generation: ActiveGeneration,
        synthesis: SpeechSynthesisSession,
        word: SynthesisWord,
    ) -> None:
        if generation.latency.first_synthesis_word_at is None:
            generation.latency.first_synthesis_word_at = time.perf_counter()
            logger.info(
                "speech synthesis first word: session=%s generation=%d text=%r",
                self.session_id,
                generation.generation_id,
                word.text,
            )
        try:
            await synthesis.add_word(word)
        except Exception as error:
            raise VoiceComponentError(
                VoiceComponent.SPEECH_SYNTHESIS,
                VoiceOperation.STREAM_SYNTHESIS,
                str(error),
            ) from error

    async def _stream_speech(
        self,
        generation: ActiveGeneration,
        events: AsyncIterator[SynthesisEvent],
    ) -> None:
        sequence_number = 0
        expected_start_sample = 0
        started = False
        async for event in _synthesis_events_with_component_error(events):
            if generation.cancelled:
                return
            match event:
                case SynthesisFirstAudioMetrics():
                    generation.latency.synthesis_metrics = event
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
                        generation.latency.first_audio_at = time.perf_counter()
                        logger.info(
                            "speech synthesis first audio: session=%s generation=%d",
                            self.session_id,
                            generation.generation_id,
                        )
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
                    if sequence_number == 0:
                        generation.latency.first_audio_sent_at = time.perf_counter()
                    await self._send_audio(header + event.pcm_bytes)
                    if sequence_number == 0:
                        self._log_first_audio_latency(generation)
                    sequence_number += 1
                    expected_start_sample += _pcm_sample_count(event.pcm_bytes)
        if not started:
            raise VoiceComponentError(
                VoiceComponent.SPEECH_SYNTHESIS,
                VoiceOperation.STREAM_SYNTHESIS,
                "The speech synthesizer returned no audio.",
            )
        await self._send_audio_boundary(
            VoiceServerEventType.ASSISTANT_AUDIO_END,
            generation.generation_id,
        )

    async def _request_generation_cancellation(self, send_event: bool) -> None:
        generation = self.active_generation
        if generation is None:
            return
        if generation.lifecycle in (
            GenerationLifecycle.CANCELLATION_REQUESTED,
            GenerationLifecycle.CANCELLED,
        ):
            return
        generation.cancelled = True
        self._transition_generation(generation, GenerationLifecycle.CANCELLATION_REQUESTED)
        self._mark_generation_interrupted(generation)
        self.active_generation = None
        self.pending_generation_teardown = generation
        if send_event:
            await self._send_audio_boundary(
                VoiceServerEventType.ASSISTANT_CANCEL,
                generation.generation_id,
            )
        if generation.task is not None and not generation.task.done():
            generation.task.cancel()

    async def _await_generation_teardown(self) -> None:
        generation = self.pending_generation_teardown
        if generation is None:
            return
        task = generation.task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if generation.lifecycle is GenerationLifecycle.CANCELLATION_REQUESTED:
            self._transition_generation(generation, GenerationLifecycle.CANCELLED)
        if self.pending_generation_teardown is generation:
            self.pending_generation_teardown = None

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
        if generation.lifecycle is GenerationLifecycle.STREAM_ENDED:
            self._transition_generation(generation, GenerationLifecycle.PLAYBACK_COMPLETE)
        self._commit_assistant_if_complete(generation)

    def _record_playback_started(self, generation_id: int) -> None:
        generation = self.generations.get(generation_id)
        if generation is None:
            return
        latency = generation.latency
        if latency.playback_started_at is not None or latency.first_audio_sent_at is None:
            return
        latency.playback_started_at = time.perf_counter()
        logger.info(
            "voice playback started: session=%s generation=%d "
            "audio_send_to_playback_ack_ms=%.1f generation_to_playback_ack_ms=%.1f",
            self.session_id,
            generation.generation_id,
            _milliseconds_between(latency.first_audio_sent_at, latency.playback_started_at),
            _milliseconds_between(
                _require_timestamp(latency.generation_started_at, "generation start"),
                latency.playback_started_at,
            ),
        )

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

    def _log_first_audio_latency(self, generation: ActiveGeneration) -> None:
        latency = generation.latency
        generation_started_at = _require_timestamp(
            latency.generation_started_at,
            "generation start",
        )
        first_language_delta_at = _require_timestamp(
            latency.first_language_delta_at,
            "first language delta",
        )
        first_synthesis_word_at = _require_timestamp(
            latency.first_synthesis_word_at,
            "first synthesis word",
        )
        first_audio_at = _require_timestamp(latency.first_audio_at, "first audio")
        synthesis_metrics = latency.synthesis_metrics
        logger.info(
            "voice first audio latency: session=%s generation=%d "
            "asr_finalization_ms=%.1f turn_commit_ms=%.1f llm_first_delta_ms=%.1f "
            "first_synthesis_word_ms=%.1f first_word_to_audio_ms=%.1f "
            "generation_to_audio_ms=%.1f tts_worker_first_word_to_audio_ms=%s "
            "tts_tokenization_ms=%s tts_lm_step_ms=%s tts_mimi_decode_ms=%s "
            "tts_model_steps=%s tts_first_audio_model_step=%s",
            self.session_id,
            generation.generation_id,
            latency.asr_finalization_seconds * 1_000,
            _milliseconds_between(latency.turn_ready_at, generation_started_at),
            _milliseconds_between(generation_started_at, first_language_delta_at),
            _milliseconds_between(generation_started_at, first_synthesis_word_at),
            _milliseconds_between(first_synthesis_word_at, first_audio_at),
            _milliseconds_between(generation_started_at, first_audio_at),
            _optional_milliseconds(
                None if synthesis_metrics is None else synthesis_metrics.first_word_to_audio_seconds
            ),
            _optional_milliseconds(
                None if synthesis_metrics is None else synthesis_metrics.tokenization_seconds
            ),
            _optional_milliseconds(
                None if synthesis_metrics is None else synthesis_metrics.language_model_step_seconds
            ),
            _optional_milliseconds(
                None if synthesis_metrics is None else synthesis_metrics.mimi_decode_seconds
            ),
            "unknown" if synthesis_metrics is None else synthesis_metrics.model_step_count,
            "unknown" if synthesis_metrics is None else synthesis_metrics.first_audio_model_step,
        )

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

    async def _send_error(
        self,
        error: VoiceComponentError,
        generation_id: int | None,
        retryable: bool,
    ) -> None:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await self._send_event(
                ErrorEvent(
                    component=error.component,
                    operation=error.operation,
                    generation_id=generation_id,
                    retryable=retryable,
                    message=str(error),
                )
            )

    async def _send_event(self, event: VoiceServerEvent) -> None:
        async with self.send_lock:
            await self.websocket.send_text(event.model_dump_json())

    async def _send_audio(self, frame: bytes) -> None:
        async with self.send_lock:
            await self.websocket.send_bytes(frame)

    def _transition_session(self, target: SessionLifecycle) -> None:
        allowed_transitions = {
            SessionLifecycle.CREATED: (SessionLifecycle.CONNECTED,),
            SessionLifecycle.CONNECTED: (SessionLifecycle.READY,),
        }
        if target not in allowed_transitions.get(self.lifecycle, ()):
            raise AssertionError(
                f"Invalid voice session lifecycle transition: {self.lifecycle} -> {target}."
            )
        self.lifecycle = target

    @staticmethod
    def _transition_generation(
        generation: ActiveGeneration,
        target: GenerationLifecycle,
    ) -> None:
        allowed_transitions = {
            GenerationLifecycle.CREATED: (
                GenerationLifecycle.STREAMING,
                GenerationLifecycle.CANCELLATION_REQUESTED,
            ),
            GenerationLifecycle.STREAMING: (
                GenerationLifecycle.CANCELLATION_REQUESTED,
                GenerationLifecycle.STREAM_ENDED,
                GenerationLifecycle.FAILED,
            ),
            GenerationLifecycle.CANCELLATION_REQUESTED: (
                GenerationLifecycle.CANCELLED,
                GenerationLifecycle.FAILED,
            ),
            GenerationLifecycle.STREAM_ENDED: (
                GenerationLifecycle.CANCELLATION_REQUESTED,
                GenerationLifecycle.PLAYBACK_COMPLETE,
            ),
        }
        if target not in allowed_transitions.get(generation.lifecycle, ()):
            raise AssertionError(
                "Invalid voice generation lifecycle transition: "
                f"{generation.lifecycle} -> {target}."
            )
        generation.lifecycle = target


def _pcm_sample_count(pcm_bytes: bytes) -> int:
    return len(pcm_bytes) // PCM_BYTES_PER_SAMPLE


def _milliseconds_to_samples(duration_ms: int) -> int:
    return duration_ms * INPUT_SAMPLE_RATE // 1_000


def _milliseconds_between(started_at: float, finished_at: float) -> float:
    if finished_at < started_at:
        raise AssertionError("Latency timestamps must increase monotonically.")
    return (finished_at - started_at) * 1_000


def _optional_milliseconds(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "unknown"
    if duration_seconds < 0:
        raise AssertionError("Latency duration must not be negative.")
    return f"{duration_seconds * 1_000:.1f}"


def _require_timestamp(timestamp: float | None, label: str) -> float:
    if timestamp is None:
        raise AssertionError(f"Missing {label} latency timestamp.")
    return timestamp


def _component_error(
    error: Exception,
    component: VoiceComponent,
    operation: VoiceOperation,
) -> VoiceComponentError:
    if isinstance(error, VoiceComponentError):
        return error
    return VoiceComponentError(component, operation, str(error))


async def _synthesis_events_with_component_error(
    events: AsyncIterator[SynthesisEvent],
) -> AsyncIterator[SynthesisEvent]:
    try:
        async for event in events:
            yield event
    except VoiceComponentError:
        raise
    except Exception as error:
        raise VoiceComponentError(
            VoiceComponent.SPEECH_SYNTHESIS,
            VoiceOperation.STREAM_SYNTHESIS,
            str(error),
        ) from error
