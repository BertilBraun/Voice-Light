from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import time
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from app.compute.voice.conversation import ConversationMessage, ConversationRole
from app.compute.voice.errors import VoiceComponent, VoiceComponentError, VoiceOperation
from app.compute.voice.interfaces import (
    KyutaiSynthesisFirstAudioMetrics,
    LanguageModel,
    LanguageModelTextDelta,
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
    TurnPredictionObservation,
    TurnPredictionSource,
    VoxtreamSynthesisFirstAudioMetrics,
)
from app.compute.voice.predictive import (
    CandidateInvalidationReason,
    CandidateLifecycle,
    CandidateOutput,
    CandidateReleaseGate,
    MediaLatencyPoint,
    PlaybackSink,
    PredictiveMetrics,
    ReleasedAudioChunk,
    ReleasedAudioEnd,
    ReleasedAudioStart,
    ReleasedTextDelta,
    ReleasedWordBoundary,
    TranscriptRevisionTracker,
    candidate_final_invalidation_reason,
)
from app.compute.voice.schemas import (
    AssistantAudioBoundaryEvent,
    AssistantAudioTextBoundaryEvent,
    AssistantTextDeltaEvent,
    CausalSource,
    ErrorEvent,
    InteractionPrediction,
    LlmHistoryEvent,
    LlmHistoryMessage,
    PlaybackCompleteEvent,
    PlaybackProgressEvent,
    PlaybackStartedEvent,
    PlaybackState,
    PlaybackStoppedEvent,
    SessionReadyEvent,
    SessionStartEvent,
    SessionStopEvent,
    SpeechStateEvent,
    TraceStamp,
    TranscriptEvent,
    TranscriptRevision,
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
    speculative_yield_threshold: float = 0.65
    commitment_yield_threshold: float = 0.9
    minimum_prediction_confidence: float = 0.7
    decisive_hold_threshold: float = 0.65
    floor_taking_overlap_threshold: float = 0.7
    vad_speculation_enabled: bool = True
    vad_endpoint_yield_probability: float = 0.7
    vad_endpoint_confidence: float = 0.7

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Self:
        value = environment.get("VOICE_LIGHT_VAD_SPECULATION_ENABLED")
        if value is None:
            return cls()
        normalized_value = value.strip().casefold()
        if normalized_value == "true":
            return cls(vad_speculation_enabled=True)
        if normalized_value == "false":
            return cls(vad_speculation_enabled=False)
        raise ValueError("VOICE_LIGHT_VAD_SPECULATION_ENABLED must be either 'true' or 'false'.")

    def __post_init__(self) -> None:
        thresholds = (
            self.speculative_yield_threshold,
            self.commitment_yield_threshold,
            self.minimum_prediction_confidence,
            self.decisive_hold_threshold,
            self.floor_taking_overlap_threshold,
            self.vad_endpoint_yield_probability,
            self.vad_endpoint_confidence,
        )
        if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
            raise ValueError("Prediction policy thresholds must be between zero and one.")
        if self.speculative_yield_threshold > self.commitment_yield_threshold:
            raise ValueError("The speculative threshold cannot exceed the commitment threshold.")


class SessionLifecycle(StrEnum):
    CREATED = "created"
    CONNECTED = "connected"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"
    CLOSED = "closed"


@dataclass
class GenerationLatency:
    asr_finalization_seconds: float
    turn_ready_at: float
    first_endpoint_at: float | None = None
    speculation_started_at: float | None = None
    generation_started_at: float | None = None
    first_language_delta_at: float | None = None
    first_synthesis_word_at: float | None = None
    first_audio_at: float | None = None
    first_audio_sent_at: float | None = None
    turn_committed_at: float | None = None
    asr_finalized_at: float | None = None
    candidate_promoted_at: float | None = None
    candidate_invalidated_at: float | None = None
    playback_started_at: float | None = None
    synthesis_metrics: SynthesisFirstAudioMetrics | None = None
    first_endpoint: MediaLatencyPoint | None = None
    speculation_start: MediaLatencyPoint | None = None
    qwen_start: MediaLatencyPoint | None = None
    qwen_first_complete_word: MediaLatencyPoint | None = None
    tts_first_word: MediaLatencyPoint | None = None
    tts_first_pcm: MediaLatencyPoint | None = None
    turn_commitment: MediaLatencyPoint | None = None
    asr_finalization: MediaLatencyPoint | None = None
    candidate_resolution: MediaLatencyPoint | None = None
    first_released_pcm: MediaLatencyPoint | None = None
    first_browser_playback_ack: MediaLatencyPoint | None = None


@dataclass
class ActiveGeneration:
    candidate_id: int
    generation_id: int
    transcript_revision_id: int | None
    stable_transcript_prefix: str
    anchored_text: str
    input_audio_sample_position: int
    monotonic_creation_time: float
    causal_prediction: InteractionPrediction | None
    prediction_confidence: float | None
    prompt_messages: tuple[ConversationMessage, ...]
    release_gate: CandidateReleaseGate
    speculative: bool
    latency: GenerationLatency
    task: asyncio.Task[None] | None = None
    response_text: str = ""
    qwen_token_count: int = 0
    tts_sample_count: int = 0
    acknowledged_offset: int = 0
    history_index: int | None = None
    boundary_samples: dict[int, int] = field(default_factory=dict)
    generation_finished: bool = False
    playback_complete: bool = False
    cancelled: bool = False
    first_release_recorded: bool = False
    accepts_playback: bool = True
    playback_stopped: asyncio.Event = field(default_factory=asyncio.Event)
    lifecycle: CandidateLifecycle = CandidateLifecycle.CREATED
    invalidation_reason: CandidateInvalidationReason | None = None
    followed_invalidation: bool = False


class WebSocketPlaybackSink:
    def __init__(self, websocket: WebSocket, send_lock: asyncio.Lock) -> None:
        self.websocket = websocket
        self.send_lock = send_lock

    async def send(self, output: CandidateOutput) -> None:
        async with self.send_lock:
            match output:
                case ReleasedTextDelta():
                    await self.websocket.send_text(
                        AssistantTextDeltaEvent(
                            generation_id=output.generation_id,
                            text=output.text,
                        ).model_dump_json()
                    )
                case ReleasedAudioStart():
                    await self.websocket.send_text(
                        AssistantAudioBoundaryEvent(
                            type=VoiceServerEventType.ASSISTANT_AUDIO_START,
                            generation_id=output.generation_id,
                        ).model_dump_json()
                    )
                case ReleasedWordBoundary():
                    await self.websocket.send_text(
                        AssistantAudioTextBoundaryEvent(
                            generation_id=output.generation_id,
                            text_offset=output.text_offset,
                            start_sample=output.start_sample,
                        ).model_dump_json()
                    )
                case ReleasedAudioChunk():
                    header = struct.pack(
                        "<III",
                        output.generation_id,
                        output.sequence_number,
                        output.start_sample,
                    )
                    await self.websocket.send_bytes(header + output.pcm_bytes)
                case ReleasedAudioEnd():
                    await self.websocket.send_text(
                        AssistantAudioBoundaryEvent(
                            type=VoiceServerEventType.ASSISTANT_AUDIO_END,
                            generation_id=output.generation_id,
                        ).model_dump_json()
                    )


class VoiceSession:
    def __init__(
        self,
        websocket: WebSocket,
        speech_detector: SpeechDetector,
        transcriber: Transcriber,
        language_model: LanguageModel,
        speech_synthesizer: SpeechSynthesizer,
        policy: SessionPolicy,
        turn_prediction_source: TurnPredictionSource | None = None,
        playback_sink: PlaybackSink | None = None,
    ) -> None:
        self.websocket = websocket
        self.speech_detector = speech_detector
        self.transcriber = transcriber
        self.language_model = language_model
        self.speech_synthesizer = speech_synthesizer
        self.policy = policy
        self.turn_prediction_source = turn_prediction_source
        self.session_id = str(uuid4())
        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=AUDIO_QUEUE_MAX_CHUNKS
        )
        self.send_lock = asyncio.Lock()
        self.playback_sink = playback_sink or WebSocketPlaybackSink(websocket, self.send_lock)
        self.conversation: list[ConversationMessage] = []
        self.generations: dict[int, ActiveGeneration] = {}
        self.active_generation: ActiveGeneration | None = None
        self.next_generation_id = 1
        self.audio_sample_count = 0
        self.started = False
        self.lifecycle = SessionLifecycle.CREATED
        self.pending_generation_teardown: ActiveGeneration | None = None
        self.transcript_revisions = TranscriptRevisionTracker()
        self.predictive_metrics = PredictiveMetrics()
        self.latest_prediction: InteractionPrediction | None = None
        self.first_endpoint_at: float | None = None
        self.first_endpoint_input_sample_position: int | None = None
        self.turn_had_invalidated_candidate = False

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
            if self.turn_prediction_source is not None:
                await self.turn_prediction_source.close()
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
            logger.info(
                "voice predictive metrics: session=%s report=%r",
                self.session_id,
                self.predictive_metrics.report(),
            )
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
        previous_chunk_was_speech = False
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
                    previous_chunk_was_speech = True
                    silent_samples = 0
                    await self._request_generation_cancellation(send_event=True)
                    logger.info("speech started: session=%s", self.session_id)
                    await self._send_speech_state(VoiceServerEventType.VAD_STARTED)
                    for pre_roll_chunk in pre_roll_chunks:
                        await self._add_transcription_audio(transcription, pre_roll_chunk)
                    pre_roll_chunks.clear()
                    pre_roll_samples = 0
                    continue

                vad_endpoint_detected = not is_speech and previous_chunk_was_speech
                if is_speech and not previous_chunk_was_speech:
                    await self._invalidate_speculative_candidate(
                        CandidateInvalidationReason.USER_ACTIVITY_RESUMED
                    )
                revision = await self._add_transcription_audio(transcription, pcm_bytes)
                if revision is not None:
                    await self._validate_candidate_revision(revision)
                if not is_speech and self.first_endpoint_at is None:
                    self.first_endpoint_at = time.perf_counter()
                    self.first_endpoint_input_sample_position = self.audio_sample_count
                    generation = self.active_generation
                    if generation is not None and generation.latency.first_endpoint is None:
                        generation.latency.first_endpoint_at = self.first_endpoint_at
                        generation.latency.first_endpoint = MediaLatencyPoint(
                            monotonic_time_seconds=self.first_endpoint_at,
                            input_sample_position=self.audio_sample_count,
                            output_sample_position=None,
                            text_offset=None,
                        )
                prediction = await self._predict_turn(pcm_bytes, is_speech)
                if (
                    prediction is None
                    and vad_endpoint_detected
                    and self.policy.vad_speculation_enabled
                ):
                    prediction = self._vad_endpoint_prediction()
                if prediction is not None:
                    self.latest_prediction = prediction
                    await self._handle_prediction(prediction)
                if is_speech:
                    silent_samples = 0
                    previous_chunk_was_speech = True
                    continue
                silent_samples += sample_count
                previous_chunk_was_speech = False
                prediction_commits = (
                    prediction is not None
                    and prediction.confidence >= self.policy.minimum_prediction_confidence
                    and prediction.p_user_yield >= self.policy.commitment_yield_threshold
                )
                prediction_blocks_silence_commit = (
                    prediction is not None
                    and prediction.confidence >= self.policy.minimum_prediction_confidence
                    and prediction.p_user_speech >= self.policy.decisive_hold_threshold
                )
                silence_commits = (
                    silent_samples >= required_silent_samples
                    and not prediction_blocks_silence_commit
                )
                if not prediction_commits and not silence_commits:
                    continue
                speech_active = False
                silent_samples = 0
                await self._send_speech_state(VoiceServerEventType.VAD_STOPPED)
                await self._finalize_turn(transcription)
                await transcription.close()
                transcription = self.transcriber.start_session()
                self.transcript_revisions.reset_turn()
                self.latest_prediction = None
                self.first_endpoint_at = None
                self.first_endpoint_input_sample_position = None
                self.turn_had_invalidated_candidate = False
        finally:
            await transcription.close()

    async def _add_transcription_audio(
        self,
        transcription: TranscriptionSession,
        pcm_bytes: bytes,
    ) -> TranscriptRevision | None:
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
            return self.transcript_revisions.update(
                partial_text,
                self.audio_sample_count,
            )
        return None

    def _vad_endpoint_prediction(self) -> InteractionPrediction:
        revision = self.transcript_revisions.latest
        return InteractionPrediction(
            stamp=TraceStamp(
                event_id=str(uuid4()),
                parent_event_ids=(() if revision is None else (revision.stamp.event_id,)),
                monotonic_time_ns=time.perf_counter_ns(),
                input_sample_position=self.audio_sample_count,
                output_sample_position=None,
                transcript_revision_id=(None if revision is None else revision.revision_id),
                source=CausalSource.SILERO_VAD,
                model_name="silero-vad",
                model_revision=None,
            ),
            p_user_speech=0.0,
            p_user_yield=self.policy.vad_endpoint_yield_probability,
            p_user_backchannel=0.0,
            p_user_interruption=0.0,
            future_user_activity_horizons=(),
            assistant_playback_state=PlaybackState.IDLE,
            confidence=self.policy.vad_endpoint_confidence,
        )

    async def _predict_turn(
        self,
        pcm_bytes: bytes,
        is_speech: bool,
    ) -> InteractionPrediction | None:
        source = self.turn_prediction_source
        if source is None:
            return None
        observation = TurnPredictionObservation(
            pcm_bytes=pcm_bytes,
            is_speech=is_speech,
            input_sample_position=self.audio_sample_count,
            monotonic_time_ns=time.perf_counter_ns(),
            transcript_revision=self.transcript_revisions.latest,
        )
        try:
            prediction = await source.predict(observation)
        except Exception as error:
            raise VoiceComponentError(
                VoiceComponent.TURN_PREDICTION,
                VoiceOperation.PREDICT_TURN,
                str(error),
            ) from error
        if prediction is None:
            return None
        if prediction.stamp.input_sample_position != observation.input_sample_position:
            raise ValueError("Turn predictions must identify the observed input sample position.")
        revision_id = (
            None
            if observation.transcript_revision is None
            else observation.transcript_revision.revision_id
        )
        if prediction.stamp.transcript_revision_id != revision_id:
            raise ValueError("Turn predictions must identify the observed transcript revision.")
        if (
            observation.transcript_revision is not None
            and observation.transcript_revision.stamp.event_id
            not in prediction.stamp.parent_event_ids
        ):
            raise ValueError("Turn predictions must cite the transcript revision as a parent.")
        self.latest_prediction = prediction
        return prediction

    async def _handle_prediction(self, prediction: InteractionPrediction) -> None:
        generation = self.active_generation
        if generation is not None and generation.speculative:
            if prediction.p_user_interruption >= self.policy.floor_taking_overlap_threshold:
                await self._invalidate_speculative_candidate(
                    CandidateInvalidationReason.FLOOR_TAKING_OVERLAP
                )
            elif prediction.p_user_speech >= self.policy.decisive_hold_threshold:
                await self._invalidate_speculative_candidate(
                    CandidateInvalidationReason.PREDICTION_RETURNED_TO_HOLD
                )
        revision = self.transcript_revisions.latest
        prompted_text = (
            "" if revision is None else self._speculative_prompt_text(revision, prediction)
        )
        if (
            self.active_generation is None
            and revision is not None
            and prompted_text
            and prediction.confidence >= self.policy.minimum_prediction_confidence
            and prediction.p_user_yield >= self.policy.speculative_yield_threshold
        ):
            await self._start_speculative_candidate(revision, prediction)

    async def _validate_candidate_revision(self, revision: TranscriptRevision) -> None:
        generation = self.active_generation
        if generation is None or not generation.speculative:
            return
        stable_prefix = generation.stable_transcript_prefix
        if not revision.stable_prefix.startswith(stable_prefix):
            await self._invalidate_speculative_candidate(
                CandidateInvalidationReason.STABLE_PREFIX_REVISED
            )
            return
        prediction = generation.causal_prediction
        if (
            prediction is not None
            and prediction.stamp.source is CausalSource.SILERO_VAD
            and candidate_final_invalidation_reason(
                stable_prefix=stable_prefix,
                prompted_text=generation.anchored_text,
                final_text=f"{revision.stable_prefix}{revision.volatile_suffix}",
            )
            is not None
        ):
            await self._invalidate_speculative_candidate(
                CandidateInvalidationReason.TRANSCRIPT_SUPERSEDED
            )

    async def _start_speculative_candidate(
        self,
        revision: TranscriptRevision,
        prediction: InteractionPrediction,
    ) -> None:
        await self._finalize_cancelled_playback()
        await self._await_generation_teardown()
        assert self.active_generation is None
        prompted_text = self._speculative_prompt_text(revision, prediction)
        prompt_messages = (
            *self.conversation,
            ConversationMessage(role=ConversationRole.USER, content=prompted_text),
        )
        created_at = time.perf_counter()
        generation = self._create_generation(
            transcript_revision_id=revision.revision_id,
            stable_transcript_prefix=revision.stable_prefix,
            anchored_text=prompted_text,
            input_audio_sample_position=revision.audio_sample_position,
            causal_prediction=prediction,
            prediction_confidence=prediction.confidence,
            prompt_messages=prompt_messages,
            speculative=True,
            asr_finalization_seconds=0.0,
            turn_ready_at=created_at,
            created_at=created_at,
        )
        generation.latency.first_endpoint_at = self.first_endpoint_at
        generation.latency.speculation_started_at = created_at
        if (
            self.first_endpoint_at is not None
            and self.first_endpoint_input_sample_position is not None
        ):
            generation.latency.first_endpoint = MediaLatencyPoint(
                monotonic_time_seconds=self.first_endpoint_at,
                input_sample_position=self.first_endpoint_input_sample_position,
                output_sample_position=None,
                text_offset=None,
            )
        generation.latency.speculation_start = MediaLatencyPoint(
            monotonic_time_seconds=created_at,
            input_sample_position=revision.audio_sample_position,
            output_sample_position=None,
            text_offset=len(prompted_text),
        )
        self.predictive_metrics.record_candidate_created()
        self.active_generation = generation
        generation.task = asyncio.create_task(self._run_generation(generation))
        logger.info(
            "speculative candidate created: session=%s generation=%d revision=%d "
            "input_sample=%d confidence=%.3f",
            self.session_id,
            generation.generation_id,
            revision.revision_id,
            revision.audio_sample_position,
            prediction.confidence,
        )

    @staticmethod
    def _speculative_prompt_text(
        revision: TranscriptRevision,
        prediction: InteractionPrediction,
    ) -> str:
        if prediction.stamp.source is CausalSource.SILERO_VAD:
            return f"{revision.stable_prefix}{revision.volatile_suffix}".strip()
        return revision.stable_prefix.strip()

    async def _finalize_turn(self, transcription: TranscriptionSession) -> None:
        committed_at = time.perf_counter()
        logger.info("transcription finalization started: session=%s", self.session_id)
        try:
            final_text = (await transcription.finish()).strip()
        except Exception as error:
            raise _component_error(
                error,
                component=VoiceComponent.ASR,
                operation=VoiceOperation.TRANSCRIBE,
            ) from error
        finalized_at = time.perf_counter()
        finalization_seconds = finalized_at - committed_at
        logger.info(
            "transcription finalization completed: session=%s duration_seconds=%.3f "
            "character_count=%d",
            self.session_id,
            finalization_seconds,
            len(final_text),
        )
        await self._send_transcript(VoiceServerEventType.TRANSCRIPT_FINAL, final_text)
        generation = self.active_generation
        if not final_text:
            await self._invalidate_speculative_candidate(
                CandidateInvalidationReason.EMPTY_FINAL_TRANSCRIPT
            )
            return
        await self._send_transcript(VoiceServerEventType.TURN_COMMITTED, final_text)
        await self._finalize_cancelled_playback()
        if generation is not None and generation.speculative:
            invalidation_reason = candidate_final_invalidation_reason(
                stable_prefix=generation.stable_transcript_prefix,
                prompted_text=generation.anchored_text,
                final_text=final_text,
            )
            if invalidation_reason is not None:
                await self._invalidate_speculative_candidate(invalidation_reason)
                generation = None
        self.conversation.append(
            ConversationMessage(role=ConversationRole.USER, content=final_text)
        )
        if generation is not None and generation.speculative:
            generation.latency.turn_committed_at = committed_at
            generation.latency.asr_finalized_at = finalized_at
            generation.latency.asr_finalization_seconds = finalization_seconds
            generation.latency.turn_commitment = MediaLatencyPoint(
                monotonic_time_seconds=committed_at,
                input_sample_position=self.audio_sample_count,
                output_sample_position=None,
                text_offset=len(final_text),
            )
            generation.latency.asr_finalization = MediaLatencyPoint(
                monotonic_time_seconds=finalized_at,
                input_sample_position=self.audio_sample_count,
                output_sample_position=None,
                text_offset=len(final_text),
            )
            await self._promote_candidate(generation)
            return
        await self._await_generation_teardown()
        await self._start_authoritative_generation(
            final_text=final_text,
            finalization_seconds=finalization_seconds,
            committed_at=committed_at,
            finalized_at=finalized_at,
        )

    async def _promote_candidate(self, generation: ActiveGeneration) -> None:
        assert self.active_generation is generation
        promoted_at = time.perf_counter()
        generation.speculative = False
        generation.latency.candidate_promoted_at = promoted_at
        generation.latency.candidate_resolution = MediaLatencyPoint(
            monotonic_time_seconds=promoted_at,
            input_sample_position=self.audio_sample_count,
            output_sample_position=None,
            text_offset=len(generation.response_text),
        )
        self._transition_generation(generation, CandidateLifecycle.COMMITTED)
        await self._send_event(
            LlmHistoryEvent(
                generation_id=generation.generation_id,
                messages=tuple(
                    LlmHistoryMessage(role=message.role, content=message.content)
                    for message in generation.prompt_messages
                ),
            )
        )
        await generation.release_gate.release()
        generation_started_at = generation.latency.generation_started_at
        hidden_work_seconds = (
            0.0 if generation_started_at is None else max(promoted_at - generation_started_at, 0.0)
        )
        self.predictive_metrics.record_promotion(
            hidden_work_seconds=hidden_work_seconds,
            hidden_qwen_tokens=generation.qwen_token_count,
            hidden_tts_samples=generation.release_gate.buffered_pcm_sample_count,
        )
        if generation.release_gate.first_released_pcm_at is not None:
            generation.latency.first_audio_sent_at = generation.release_gate.first_released_pcm_at
            generation.latency.first_released_pcm = MediaLatencyPoint(
                monotonic_time_seconds=generation.release_gate.first_released_pcm_at,
                input_sample_position=self.audio_sample_count,
                output_sample_position=(generation.release_gate.first_released_pcm_start_sample),
                text_offset=None,
            )
            self._record_first_released_pcm(generation)
            self._log_first_audio_latency(generation)
        logger.info(
            "speculative candidate promoted: session=%s generation=%d hidden_work_ms=%.1f "
            "buffered_tts_samples=%d",
            self.session_id,
            generation.generation_id,
            hidden_work_seconds * 1_000,
            generation.release_gate.buffered_pcm_sample_count,
        )

    async def _start_authoritative_generation(
        self,
        final_text: str,
        finalization_seconds: float,
        committed_at: float,
        finalized_at: float,
    ) -> None:
        assert self.active_generation is None
        created_at = time.perf_counter()
        generation = self._create_generation(
            transcript_revision_id=(
                None
                if self.transcript_revisions.latest is None
                else self.transcript_revisions.latest.revision_id
            ),
            stable_transcript_prefix=final_text,
            anchored_text=final_text,
            input_audio_sample_position=self.audio_sample_count,
            causal_prediction=self.latest_prediction,
            prediction_confidence=(
                None if self.latest_prediction is None else self.latest_prediction.confidence
            ),
            prompt_messages=tuple(self.conversation),
            speculative=False,
            asr_finalization_seconds=finalization_seconds,
            turn_ready_at=finalized_at,
            created_at=created_at,
        )
        generation.followed_invalidation = self.turn_had_invalidated_candidate
        generation.latency.first_endpoint_at = self.first_endpoint_at
        generation.latency.turn_committed_at = committed_at
        generation.latency.asr_finalized_at = finalized_at
        if (
            self.first_endpoint_at is not None
            and self.first_endpoint_input_sample_position is not None
        ):
            generation.latency.first_endpoint = MediaLatencyPoint(
                monotonic_time_seconds=self.first_endpoint_at,
                input_sample_position=self.first_endpoint_input_sample_position,
                output_sample_position=None,
                text_offset=None,
            )
        generation.latency.turn_commitment = MediaLatencyPoint(
            monotonic_time_seconds=committed_at,
            input_sample_position=self.audio_sample_count,
            output_sample_position=None,
            text_offset=len(final_text),
        )
        generation.latency.asr_finalization = MediaLatencyPoint(
            monotonic_time_seconds=finalized_at,
            input_sample_position=self.audio_sample_count,
            output_sample_position=None,
            text_offset=len(final_text),
        )
        self._transition_generation(generation, CandidateLifecycle.COMMITTED)
        self.active_generation = generation
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

    def _create_generation(
        self,
        transcript_revision_id: int | None,
        stable_transcript_prefix: str,
        anchored_text: str,
        input_audio_sample_position: int,
        causal_prediction: InteractionPrediction | None,
        prediction_confidence: float | None,
        prompt_messages: tuple[ConversationMessage, ...],
        speculative: bool,
        asr_finalization_seconds: float,
        turn_ready_at: float,
        created_at: float,
    ) -> ActiveGeneration:
        generation_id = self.next_generation_id
        self.next_generation_id += 1
        generation = ActiveGeneration(
            candidate_id=generation_id,
            generation_id=generation_id,
            transcript_revision_id=transcript_revision_id,
            stable_transcript_prefix=stable_transcript_prefix,
            anchored_text=anchored_text,
            input_audio_sample_position=input_audio_sample_position,
            monotonic_creation_time=created_at,
            causal_prediction=causal_prediction,
            prediction_confidence=prediction_confidence,
            prompt_messages=prompt_messages,
            release_gate=CandidateReleaseGate(
                sink=self.playback_sink,
                released=not speculative,
            ),
            speculative=speculative,
            latency=GenerationLatency(
                asr_finalization_seconds=asr_finalization_seconds,
                turn_ready_at=turn_ready_at,
            ),
        )
        self.generations[generation_id] = generation
        return generation

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
        generation.latency.qwen_start = MediaLatencyPoint(
            monotonic_time_seconds=generation.latency.generation_started_at,
            input_sample_position=generation.input_audio_sample_position,
            output_sample_position=None,
            text_offset=0,
        )
        logger.info(
            "voice response generation started: session=%s generation=%d",
            self.session_id,
            generation.generation_id,
        )
        if generation.lifecycle is CandidateLifecycle.CREATED:
            self._transition_generation(generation, CandidateLifecycle.PREFILLING)
        try:
            await self._generate_response(generation)
            if generation.speculative:
                self._transition_generation(generation, CandidateLifecycle.READY)
            logger.info(
                "voice response generation completed: session=%s generation=%d",
                self.session_id,
                generation.generation_id,
            )
        except asyncio.CancelledError:
            if generation.lifecycle is CandidateLifecycle.CANCELLATION_REQUESTED:
                self._transition_generation(generation, CandidateLifecycle.CANCELLED)
            raise
        except Exception as error:
            if generation.lifecycle is not CandidateLifecycle.FAILED:
                self._transition_generation(generation, CandidateLifecycle.FAILED)
            logger.exception(
                "voice response generation failed: session=%s generation=%d",
                self.session_id,
                generation.generation_id,
            )
            if generation.speculative:
                await generation.release_gate.discard()
                failure = _component_error(
                    error,
                    component=VoiceComponent.SESSION,
                    operation=VoiceOperation.SESSION_RUN,
                )
                reason = (
                    CandidateInvalidationReason.LANGUAGE_MODEL_FAILED
                    if failure.component is VoiceComponent.LANGUAGE_MODEL
                    else CandidateInvalidationReason.SPEECH_SYNTHESIS_FAILED
                )
                generation.invalidation_reason = reason
                generation.latency.candidate_invalidated_at = time.perf_counter()
                generation.latency.candidate_resolution = MediaLatencyPoint(
                    monotonic_time_seconds=(generation.latency.candidate_invalidated_at),
                    input_sample_position=self.audio_sample_count,
                    output_sample_position=None,
                    text_offset=len(generation.response_text),
                )
                self.predictive_metrics.record_invalidation(
                    reason,
                    generation.qwen_token_count,
                    generation.tts_sample_count,
                )
                self.turn_had_invalidated_candidate = True
            if self.active_generation is generation:
                generation.cancelled = True
                self.active_generation = None
                if generation.release_gate.released:
                    self._mark_generation_interrupted(generation)
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
        language_stream: AsyncIterator[LanguageModelTextDelta],
    ) -> None:
        word_stream = CompleteWordStream()
        while True:
            try:
                delta = await anext(language_stream)
            except StopAsyncIteration:
                break
            except Exception as error:
                raise VoiceComponentError(
                    VoiceComponent.LANGUAGE_MODEL,
                    VoiceOperation.GENERATE_TEXT,
                    str(error),
                ) from error
            if delta.cumulative_token_count < generation.qwen_token_count:
                raise ValueError("Qwen cumulative token counts must be monotonic.")
            generation.qwen_token_count = delta.cumulative_token_count
            text_delta = delta.text
            if text_delta and not generation.response_text:
                generation.latency.first_language_delta_at = time.perf_counter()
                if generation.lifecycle is CandidateLifecycle.PREFILLING:
                    self._transition_generation(generation, CandidateLifecycle.STREAMING)
                logger.info(
                    "language model first delta: session=%s generation=%d",
                    self.session_id,
                    generation.generation_id,
                )
            generation.response_text += text_delta
            await generation.release_gate.publish(
                ReleasedTextDelta(
                    generation_id=generation.generation_id,
                    text=text_delta,
                )
            )
            for word in word_stream.add_text(text_delta):
                self._record_first_qwen_complete_word(generation, word)
                await self._add_synthesis_word(generation, synthesis, word)
        if not generation.response_text.strip():
            raise VoiceComponentError(
                VoiceComponent.LANGUAGE_MODEL,
                VoiceOperation.GENERATE_TEXT,
                "The language model returned an empty response.",
            )
        for word in word_stream.finish():
            self._record_first_qwen_complete_word(generation, word)
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
            generation.latency.tts_first_word = MediaLatencyPoint(
                monotonic_time_seconds=generation.latency.first_synthesis_word_at,
                input_sample_position=generation.input_audio_sample_position,
                output_sample_position=None,
                text_offset=word.text_end,
            )
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

    @staticmethod
    def _record_first_qwen_complete_word(
        generation: ActiveGeneration,
        word: SynthesisWord,
    ) -> None:
        if generation.latency.qwen_first_complete_word is not None:
            return
        generation.latency.qwen_first_complete_word = MediaLatencyPoint(
            monotonic_time_seconds=time.perf_counter(),
            input_sample_position=generation.input_audio_sample_position,
            output_sample_position=None,
            text_offset=word.text_end,
        )

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
                case KyutaiSynthesisFirstAudioMetrics() | VoxtreamSynthesisFirstAudioMetrics():
                    generation.latency.synthesis_metrics = event
                case SynthesizedWordBoundary():
                    if event.text_offset > len(generation.response_text):
                        raise ValueError("TTS returned a text boundary beyond generated text.")
                    generation.boundary_samples[event.text_offset] = event.start_sample
                    await generation.release_gate.publish(
                        ReleasedWordBoundary(
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
                        generation.latency.tts_first_pcm = MediaLatencyPoint(
                            monotonic_time_seconds=generation.latency.first_audio_at,
                            input_sample_position=generation.input_audio_sample_position,
                            output_sample_position=event.start_sample,
                            text_offset=None,
                        )
                        logger.info(
                            "speech synthesis first audio: session=%s generation=%d",
                            self.session_id,
                            generation.generation_id,
                        )
                        await generation.release_gate.publish(
                            ReleasedAudioStart(generation_id=generation.generation_id)
                        )
                    await generation.release_gate.publish(
                        ReleasedAudioChunk(
                            generation_id=generation.generation_id,
                            sequence_number=sequence_number,
                            start_sample=event.start_sample,
                            pcm_bytes=event.pcm_bytes,
                        )
                    )
                    generation.tts_sample_count += _pcm_sample_count(event.pcm_bytes)
                    if sequence_number == 0 and generation.release_gate.released:
                        generation.latency.first_audio_sent_at = (
                            generation.release_gate.first_released_pcm_at
                        )
                        generation.latency.first_released_pcm = MediaLatencyPoint(
                            monotonic_time_seconds=_require_timestamp(
                                generation.release_gate.first_released_pcm_at,
                                "first released PCM",
                            ),
                            input_sample_position=self.audio_sample_count,
                            output_sample_position=(
                                generation.release_gate.first_released_pcm_start_sample
                            ),
                            text_offset=None,
                        )
                        self._record_first_released_pcm(generation)
                        self._log_first_audio_latency(generation)
                    sequence_number += 1
                    expected_start_sample += _pcm_sample_count(event.pcm_bytes)
        if not started:
            raise VoiceComponentError(
                VoiceComponent.SPEECH_SYNTHESIS,
                VoiceOperation.STREAM_SYNTHESIS,
                "The speech synthesizer returned no audio.",
            )
        await generation.release_gate.publish(
            ReleasedAudioEnd(generation_id=generation.generation_id)
        )

    def _record_first_released_pcm(self, generation: ActiveGeneration) -> None:
        if generation.first_release_recorded:
            return
        commit_at = generation.latency.turn_committed_at
        release_at = generation.release_gate.first_released_pcm_at
        assert commit_at is not None
        assert release_at is not None
        self.predictive_metrics.record_first_release(
            commit_at=commit_at,
            release_at=release_at,
        )
        generation.first_release_recorded = True

    async def _request_generation_cancellation(self, send_event: bool) -> None:
        generation = self.active_generation
        if generation is None:
            return
        if generation.speculative:
            await self._invalidate_speculative_candidate(
                CandidateInvalidationReason.SESSION_CANCELLED
            )
            return
        if generation.lifecycle in (
            CandidateLifecycle.CANCELLATION_REQUESTED,
            CandidateLifecycle.CANCELLED,
        ):
            return
        generation.cancelled = True
        self._transition_generation(generation, CandidateLifecycle.CANCELLATION_REQUESTED)
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

    async def _invalidate_speculative_candidate(
        self,
        reason: CandidateInvalidationReason,
    ) -> None:
        generation = self.active_generation
        if generation is None or not generation.speculative:
            return
        if generation.lifecycle in (
            CandidateLifecycle.CANCELLATION_REQUESTED,
            CandidateLifecycle.CANCELLED,
            CandidateLifecycle.FAILED,
        ):
            return
        generation.cancelled = True
        generation.invalidation_reason = reason
        generation.latency.candidate_invalidated_at = time.perf_counter()
        generation.latency.candidate_resolution = MediaLatencyPoint(
            monotonic_time_seconds=generation.latency.candidate_invalidated_at,
            input_sample_position=self.audio_sample_count,
            output_sample_position=None,
            text_offset=len(generation.response_text),
        )
        self._transition_generation(generation, CandidateLifecycle.INVALIDATED)
        await generation.release_gate.discard()
        self.predictive_metrics.record_invalidation(
            reason,
            generation.qwen_token_count,
            generation.tts_sample_count,
        )
        self.turn_had_invalidated_candidate = True
        self._transition_generation(
            generation,
            CandidateLifecycle.CANCELLATION_REQUESTED,
        )
        self.active_generation = None
        self.pending_generation_teardown = generation
        if generation.task is not None and not generation.task.done():
            generation.task.cancel()
        await self._await_generation_teardown()
        logger.info(
            "speculative candidate invalidated: session=%s generation=%d reason=%s "
            "wasted_qwen_tokens=%d wasted_tts_samples=%d",
            self.session_id,
            generation.generation_id,
            reason,
            generation.qwen_token_count,
            generation.tts_sample_count,
        )

    async def _await_generation_teardown(self) -> None:
        generation = self.pending_generation_teardown
        if generation is None:
            return
        task = generation.task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if generation.lifecycle is CandidateLifecycle.CANCELLATION_REQUESTED:
            self._transition_generation(generation, CandidateLifecycle.CANCELLED)
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
        self._commit_assistant_if_complete(generation)

    def _record_playback_started(self, generation_id: int) -> None:
        generation = self.generations.get(generation_id)
        if generation is None:
            return
        latency = generation.latency
        if latency.playback_started_at is not None or latency.first_audio_sent_at is None:
            return
        latency.playback_started_at = time.perf_counter()
        latency.first_browser_playback_ack = MediaLatencyPoint(
            monotonic_time_seconds=latency.playback_started_at,
            input_sample_position=self.audio_sample_count,
            output_sample_position=0,
            text_offset=0,
        )
        turn_committed_at = _require_timestamp(
            latency.turn_committed_at,
            "turn commitment",
        )
        self.predictive_metrics.record_first_playback(
            commit_at=turn_committed_at,
            playback_at=latency.playback_started_at,
            true_end_at=None,
            had_candidate=latency.candidate_promoted_at is not None,
            followed_invalidation=generation.followed_invalidation,
        )
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
        match synthesis_metrics:
            case KyutaiSynthesisFirstAudioMetrics():
                tts_backend = "kyutai"
                tokenization_ms = _optional_milliseconds(synthesis_metrics.tokenization_seconds)
                language_model_step_ms = _optional_milliseconds(
                    synthesis_metrics.language_model_step_seconds
                )
                mimi_decode_ms = _optional_milliseconds(synthesis_metrics.mimi_decode_seconds)
                model_steps: int | str = synthesis_metrics.model_step_count
                first_audio_model_step: int | str = synthesis_metrics.first_audio_model_step
                prompt_preparation_ms = "unknown"
                first_frame_generation_ms = "unknown"
            case VoxtreamSynthesisFirstAudioMetrics():
                tts_backend = "voxtream"
                tokenization_ms = "unknown"
                language_model_step_ms = "unknown"
                mimi_decode_ms = "unknown"
                model_steps = "unknown"
                first_audio_model_step = "unknown"
                prompt_preparation_ms = _optional_milliseconds(
                    synthesis_metrics.prompt_preparation_seconds
                )
                first_frame_generation_ms = _optional_milliseconds(
                    synthesis_metrics.first_frame_generation_seconds
                )
            case None:
                tts_backend = "unknown"
                tokenization_ms = "unknown"
                language_model_step_ms = "unknown"
                mimi_decode_ms = "unknown"
                model_steps = "unknown"
                first_audio_model_step = "unknown"
                prompt_preparation_ms = "unknown"
                first_frame_generation_ms = "unknown"
        logger.info(
            "voice first audio latency: session=%s generation=%d "
            "asr_finalization_ms=%.1f turn_commit_ms=%.1f llm_first_delta_ms=%.1f "
            "first_synthesis_word_ms=%.1f first_word_to_audio_ms=%.1f "
            "generation_to_audio_ms=%.1f tts_backend=%s "
            "tts_worker_first_word_to_audio_ms=%s "
            "tts_tokenization_ms=%s tts_lm_step_ms=%s tts_mimi_decode_ms=%s "
            "tts_model_steps=%s tts_first_audio_model_step=%s "
            "tts_prompt_preparation_ms=%s tts_first_frame_generation_ms=%s",
            self.session_id,
            generation.generation_id,
            latency.asr_finalization_seconds * 1_000,
            _milliseconds_between(latency.turn_ready_at, generation_started_at),
            _milliseconds_between(generation_started_at, first_language_delta_at),
            _milliseconds_between(generation_started_at, first_synthesis_word_at),
            _milliseconds_between(first_synthesis_word_at, first_audio_at),
            _milliseconds_between(generation_started_at, first_audio_at),
            tts_backend,
            _optional_milliseconds(
                None if synthesis_metrics is None else synthesis_metrics.first_word_to_audio_seconds
            ),
            tokenization_ms,
            language_model_step_ms,
            mimi_decode_ms,
            model_steps,
            first_audio_model_step,
            prompt_preparation_ms,
            first_frame_generation_ms,
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
        target: CandidateLifecycle,
    ) -> None:
        allowed_transitions = {
            CandidateLifecycle.CREATED: (
                CandidateLifecycle.PREFILLING,
                CandidateLifecycle.COMMITTED,
                CandidateLifecycle.INVALIDATED,
                CandidateLifecycle.CANCELLATION_REQUESTED,
            ),
            CandidateLifecycle.PREFILLING: (
                CandidateLifecycle.STREAMING,
                CandidateLifecycle.READY,
                CandidateLifecycle.COMMITTED,
                CandidateLifecycle.INVALIDATED,
                CandidateLifecycle.CANCELLATION_REQUESTED,
                CandidateLifecycle.FAILED,
            ),
            CandidateLifecycle.STREAMING: (
                CandidateLifecycle.READY,
                CandidateLifecycle.COMMITTED,
                CandidateLifecycle.INVALIDATED,
                CandidateLifecycle.CANCELLATION_REQUESTED,
                CandidateLifecycle.FAILED,
            ),
            CandidateLifecycle.READY: (
                CandidateLifecycle.COMMITTED,
                CandidateLifecycle.INVALIDATED,
                CandidateLifecycle.CANCELLATION_REQUESTED,
            ),
            CandidateLifecycle.COMMITTED: (
                CandidateLifecycle.CANCELLATION_REQUESTED,
                CandidateLifecycle.FAILED,
            ),
            CandidateLifecycle.INVALIDATED: (CandidateLifecycle.CANCELLATION_REQUESTED,),
            CandidateLifecycle.CANCELLATION_REQUESTED: (
                CandidateLifecycle.CANCELLED,
                CandidateLifecycle.FAILED,
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
