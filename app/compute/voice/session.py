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

from app.compute.voice.conversation import (
    ConversationMessage,
    ConversationRole,
    ModelAssistantMessage,
    ModelConversationTurn,
    ModelMessage,
    ModelToolMessage,
    ModelTurnIdentity,
    ModelUserMessage,
    PrivateModelContext,
)
from app.compute.voice.errors import VoiceComponent, VoiceComponentError, VoiceOperation
from app.compute.voice.interfaces import (
    KyutaiSynthesisFirstAudioMetrics,
    LanguageModel,
    LanguageModelCompleted,
    LanguageModelFailed,
    LanguageModelRequest,
    LanguageModelTextDelta,
    LanguageModelToolCall,
    LanguageModelToolCallFailure,
    LanguageModelToolCallStarted,
    SpeechDetector,
    SpeechSynthesizer,
    SpeechUnderstandingProvider,
    SpeechUnderstandingSession,
    SynthesisEvent,
    SynthesisFirstAudioMetrics,
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
    VoxtreamSynthesisFirstAudioMetrics,
)
from app.compute.voice.model_constants import LANGUAGE_MODEL_SYSTEM_PROMPT
from app.compute.voice.overlap import (
    OverlapEvidence,
    OverlapMetrics,
    OverlapResolutionKind,
    ProvisionalOverlapDecision,
    ProvisionalOverlapPolicyConfig,
    ProvisionalVadTranscriptOverlapPolicy,
)
from app.compute.voice.playback import (
    PlaybackAcknowledgementDisposition,
    PlaybackController,
    PlaybackFlowControl,
    PlaybackPolicyConfig,
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
    candidate_final_invalidation_reason,
    candidate_revision_invalidation_reason,
)
from app.compute.voice.schemas import (
    AssistantAudioBoundaryEvent,
    AssistantAudioTextBoundaryEvent,
    AssistantLatencyEvent,
    AssistantTextDeltaEvent,
    CapturedAudioChunk,
    CausalSource,
    ErrorEvent,
    InteractionActionDebugEvent,
    InteractionPolicyDebugEvent,
    InteractionPrediction,
    LlmHistoryEvent,
    LlmHistoryMessage,
    LlmModelAssistantMessage,
    LlmModelMessage,
    LlmModelRequestEvent,
    LlmModelSystemMessage,
    LlmModelToolMessage,
    LlmModelUserMessage,
    PlaybackClockEvent,
    PlaybackCommandAcknowledgementEvent,
    PlaybackCommandEvent,
    PlaybackCompleteEvent,
    PlaybackCondition,
    PlaybackProgressEvent,
    PlaybackStartedEvent,
    PlaybackState,
    PlaybackStoppedEvent,
    SearchDebugEvent,
    SearchDebugResultEvent,
    SessionReadyEvent,
    SessionStartEvent,
    SessionStopEvent,
    SileroEvidence,
    SpeechStateEvent,
    SpeechUnderstandingDebugEvent,
    SpeechUnderstandingDegradedEvent,
    TraceStamp,
    TranscriptEvent,
    TranscriptRevision,
    TurnAdapterStatus,
    TurnPredictionDisposition,
    VoiceServerEvent,
    VoiceServerEventType,
    voice_client_event_adapter,
)
from app.compute.voice.speech_understanding import InteractionPredictionReducer
from app.compute.voice.synthesis_sequence import SpeechSynthesisSequence
from app.compute.voice.tool_routing import (
    TemperatureConversion,
    required_search_reason,
    route_required_calculation_call,
    route_required_search_call,
    temperature_conversion_spoken_result,
)
from app.compute.voice.tools import (
    CalculateToolCallFunction,
    SearchToolAvailability,
    SerializedToolCall,
    ToolCall,
    ToolCallAdmission,
    ToolCallFailure,
    ToolCallFailureReason,
    ToolCircuitBreaker,
    ToolExecutionFailure,
    ToolExecutionFailureReason,
    ToolExecutor,
    ToolInvalidationReason,
    ToolLifecycle,
    ToolName,
    ToolResultCommitStatus,
    ToolResultDelivery,
    ToolSpecification,
    ToolSuccess,
    tool_failure_spoken_response,
)
from app.compute.voice.tts_text import normalize_synthesis_word
from app.compute.voice.word_stream import CompleteWordStream

INPUT_SAMPLE_RATE = 16_000
PCM_BYTES_PER_SAMPLE = 2
AUDIO_QUEUE_MAX_CHUNKS = 100
PLAYBACK_STOP_TIMEOUT_SECONDS = 0.25
SPEECH_DEBUG_INTERVAL_SAMPLES = INPUT_SAMPLE_RATE * 80 // 1_000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionPolicy:
    # This silence begins after Silero's configured 250 ms end-of-speech decision.
    silence_duration_ms: int = 500
    pre_roll_duration_ms: int = 300
    speculative_yield_threshold: float = 0.55
    speculative_turn_completion_threshold: float = 0.55
    speculative_minimum_confidence: float = 0.6
    commitment_yield_threshold: float = 0.9
    minimum_prediction_confidence: float = 0.7
    decisive_hold_threshold: float = 0.65
    floor_taking_overlap_threshold: float = 0.82
    non_floor_feedback_overlap_threshold: float = 0.82
    overlap_classification_deadline_ms: int = 500
    transcript_free_floor_take_deadline_ms: int = 900
    minimum_normal_turn_commit_silence_ms: int = 240
    overlap_finalization_grace_ms: int = 120
    overlap_prediction_settle_ms: int = 80
    overlap_rearm_silence_ms: int = 160
    vad_speculation_enabled: bool = True
    pending_silence_speculation_ms: int = 80
    vad_speculation_debounce_ms: int = 0
    vad_endpoint_yield_probability: float = 0.7
    vad_endpoint_confidence: float = 0.7
    maximum_prediction_lag_ms: int = 240
    tool_timeout_seconds: float = 30.0
    tool_cancellation_timeout_seconds: float = 0.25
    maximum_tool_rounds: int = 8

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Self:
        enabled_value = environment.get("VOICE_LIGHT_VAD_SPECULATION_ENABLED")
        vad_speculation_enabled = True
        if enabled_value is not None:
            normalized_value = enabled_value.strip().casefold()
            if normalized_value == "true":
                vad_speculation_enabled = True
            elif normalized_value == "false":
                vad_speculation_enabled = False
            else:
                raise ValueError(
                    "VOICE_LIGHT_VAD_SPECULATION_ENABLED must be either 'true' or 'false'."
                )
        debounce_value = environment.get("VOICE_LIGHT_VAD_SPECULATION_DEBOUNCE_MS")
        vad_speculation_debounce_ms = 0
        if debounce_value is not None:
            try:
                vad_speculation_debounce_ms = int(debounce_value)
            except ValueError as error:
                raise ValueError(
                    "VOICE_LIGHT_VAD_SPECULATION_DEBOUNCE_MS must be an integer."
                ) from error
        pending_silence_speculation_ms = _environment_integer(
            environment,
            "VOICE_LIGHT_PENDING_SILENCE_SPECULATION_MS",
            80,
        )
        tool_timeout_value = environment.get("VOICE_LIGHT_TOOL_TIMEOUT_SECONDS")
        tool_timeout_seconds = 30.0
        if tool_timeout_value is not None:
            try:
                tool_timeout_seconds = float(tool_timeout_value)
            except ValueError as error:
                raise ValueError("VOICE_LIGHT_TOOL_TIMEOUT_SECONDS must be numeric.") from error
        tool_cancellation_value = environment.get("VOICE_LIGHT_TOOL_CANCELLATION_TIMEOUT_SECONDS")
        tool_cancellation_timeout_seconds = 0.25
        if tool_cancellation_value is not None:
            try:
                tool_cancellation_timeout_seconds = float(tool_cancellation_value)
            except ValueError as error:
                raise ValueError(
                    "VOICE_LIGHT_TOOL_CANCELLATION_TIMEOUT_SECONDS must be numeric."
                ) from error
        maximum_tool_rounds_value = environment.get("VOICE_LIGHT_MAXIMUM_TOOL_ROUNDS")
        maximum_tool_rounds = 8
        if maximum_tool_rounds_value is not None:
            try:
                maximum_tool_rounds = int(maximum_tool_rounds_value)
            except ValueError as error:
                raise ValueError("VOICE_LIGHT_MAXIMUM_TOOL_ROUNDS must be an integer.") from error
        floor_taking_overlap_threshold = _environment_float(
            environment,
            "VOICE_LIGHT_FLOOR_TAKE_THRESHOLD",
            0.82,
        )
        non_floor_feedback_overlap_threshold = _environment_float(
            environment,
            "VOICE_LIGHT_NON_FLOOR_FEEDBACK_THRESHOLD",
            0.82,
        )
        overlap_classification_deadline_ms = _environment_integer(
            environment,
            "VOICE_LIGHT_OVERLAP_CLASSIFICATION_DEADLINE_MS",
            500,
        )
        transcript_free_floor_take_deadline_ms = _environment_integer(
            environment,
            "VOICE_LIGHT_TRANSCRIPT_FREE_FLOOR_TAKE_DEADLINE_MS",
            900,
        )
        minimum_normal_turn_commit_silence_ms = _environment_integer(
            environment,
            "VOICE_LIGHT_MINIMUM_NORMAL_TURN_COMMIT_SILENCE_MS",
            240,
        )
        overlap_finalization_grace_ms = _environment_integer(
            environment, "VOICE_LIGHT_OVERLAP_FINALIZATION_GRACE_MS", 120
        )
        overlap_prediction_settle_ms = _environment_integer(
            environment, "VOICE_LIGHT_OVERLAP_PREDICTION_SETTLE_MS", 80
        )
        overlap_rearm_silence_ms = _environment_integer(
            environment,
            "VOICE_LIGHT_OVERLAP_REARM_SILENCE_MS",
            160,
        )
        maximum_prediction_lag_ms = _environment_integer(
            environment,
            "VOICE_LIGHT_MAXIMUM_PREDICTION_LAG_MS",
            240,
        )
        speculative_yield_threshold = _environment_float(
            environment,
            "VOICE_LIGHT_SPECULATIVE_YIELD_THRESHOLD",
            0.55,
        )
        speculative_turn_completion_threshold = _environment_float(
            environment,
            "VOICE_LIGHT_SPECULATIVE_TURN_COMPLETION_THRESHOLD",
            0.55,
        )
        speculative_minimum_confidence = _environment_float(
            environment,
            "VOICE_LIGHT_SPECULATIVE_MINIMUM_CONFIDENCE",
            0.6,
        )
        vad_endpoint_yield_probability = _environment_float(
            environment,
            "VOICE_LIGHT_VAD_ENDPOINT_YIELD_PROBABILITY",
            0.7,
        )
        vad_endpoint_confidence = _environment_float(
            environment,
            "VOICE_LIGHT_VAD_ENDPOINT_CONFIDENCE",
            0.7,
        )
        return cls(
            speculative_yield_threshold=speculative_yield_threshold,
            speculative_turn_completion_threshold=(speculative_turn_completion_threshold),
            speculative_minimum_confidence=speculative_minimum_confidence,
            vad_speculation_enabled=vad_speculation_enabled,
            pending_silence_speculation_ms=pending_silence_speculation_ms,
            vad_speculation_debounce_ms=vad_speculation_debounce_ms,
            tool_timeout_seconds=tool_timeout_seconds,
            tool_cancellation_timeout_seconds=tool_cancellation_timeout_seconds,
            maximum_tool_rounds=maximum_tool_rounds,
            floor_taking_overlap_threshold=floor_taking_overlap_threshold,
            non_floor_feedback_overlap_threshold=non_floor_feedback_overlap_threshold,
            overlap_classification_deadline_ms=overlap_classification_deadline_ms,
            transcript_free_floor_take_deadline_ms=(transcript_free_floor_take_deadline_ms),
            minimum_normal_turn_commit_silence_ms=(minimum_normal_turn_commit_silence_ms),
            overlap_finalization_grace_ms=overlap_finalization_grace_ms,
            overlap_prediction_settle_ms=overlap_prediction_settle_ms,
            overlap_rearm_silence_ms=overlap_rearm_silence_ms,
            maximum_prediction_lag_ms=maximum_prediction_lag_ms,
            vad_endpoint_yield_probability=vad_endpoint_yield_probability,
            vad_endpoint_confidence=vad_endpoint_confidence,
        )

    def __post_init__(self) -> None:
        thresholds = (
            self.speculative_yield_threshold,
            self.speculative_turn_completion_threshold,
            self.speculative_minimum_confidence,
            self.commitment_yield_threshold,
            self.minimum_prediction_confidence,
            self.decisive_hold_threshold,
            self.floor_taking_overlap_threshold,
            self.non_floor_feedback_overlap_threshold,
            self.vad_endpoint_yield_probability,
            self.vad_endpoint_confidence,
        )
        if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
            raise ValueError("Prediction policy thresholds must be between zero and one.")
        if self.speculative_yield_threshold > self.commitment_yield_threshold:
            raise ValueError("The speculative threshold cannot exceed the commitment threshold.")
        if self.vad_speculation_debounce_ms < 0:
            raise ValueError("The VAD speculation debounce cannot be negative.")
        if self.pending_silence_speculation_ms <= 0:
            raise ValueError("The pending-silence speculation window must be positive.")
        if self.maximum_prediction_lag_ms < 0:
            raise ValueError("The maximum prediction lag cannot be negative.")
        if self.overlap_classification_deadline_ms <= 0:
            raise ValueError("The overlap classification deadline must be positive.")
        if self.transcript_free_floor_take_deadline_ms <= self.overlap_classification_deadline_ms:
            raise ValueError(
                "The transcript-free floor-take deadline must exceed the overlap "
                "classification deadline."
            )
        if self.minimum_normal_turn_commit_silence_ms <= 0:
            raise ValueError("The minimum normal-turn commit silence must be positive.")
        if self.overlap_finalization_grace_ms <= 0:
            raise ValueError("The overlap finalization grace must be positive.")
        if self.overlap_prediction_settle_ms <= 0:
            raise ValueError("The overlap prediction settle time must be positive.")
        if self.overlap_rearm_silence_ms <= 0:
            raise ValueError("The overlap rearm silence must be positive.")
        if self.tool_timeout_seconds <= 0.0:
            raise ValueError("The tool timeout must be positive.")
        if self.tool_cancellation_timeout_seconds <= 0.0:
            raise ValueError("The tool cancellation timeout must be positive.")
        if self.maximum_tool_rounds <= 0:
            raise ValueError("The maximum tool round count must be positive.")


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
    final_vad_endpoint_at: float | None = None
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
    first_bridge_text: MediaLatencyPoint | None = None
    first_bridge_pcm: MediaLatencyPoint | None = None
    tool_call_started_at: float | None = None
    tool_call_completed_at: float | None = None
    tool_execution_started_at: float | None = None
    tool_execution_completed_at: float | None = None
    tool_result_committed_at: float | None = None
    second_invocation_started_at: float | None = None
    first_final_answer_text: MediaLatencyPoint | None = None
    first_final_answer_pcm: MediaLatencyPoint | None = None
    cancelled_or_invalidated_at: float | None = None
    commit_readiness: GenerationCommitReadiness | None = None


@dataclass(frozen=True)
class GenerationCommitReadiness:
    speculative_candidate: bool
    hidden_work_ms: float
    qwen_token_count: int
    synthesis_word_count: int
    first_tts_pcm_ready: bool
    buffered_pcm_sample_count: int

    def __post_init__(self) -> None:
        if self.hidden_work_ms < 0.0:
            raise ValueError("Hidden generation work cannot be negative.")
        if self.qwen_token_count < 0:
            raise ValueError("The prepared Qwen token count cannot be negative.")
        if self.synthesis_word_count < 0:
            raise ValueError("The prepared synthesis word count cannot be negative.")
        if self.buffered_pcm_sample_count < 0:
            raise ValueError("The buffered PCM sample count cannot be negative.")


@dataclass
class ToolExecutionJournalEntry:
    identity: ModelTurnIdentity
    round_index: int
    invocation_id: int
    call_id: str
    created_at: float
    lifecycle: ToolLifecycle = ToolLifecycle.NOT_REQUESTED
    result_commit_status: ToolResultCommitStatus = ToolResultCommitStatus.NOT_STAGED
    continuation_invocation_id: int | None = None
    request: SerializedToolCall | None = None
    call: ToolCall | None = None
    outcome: ToolSuccess | ToolExecutionFailure | None = None
    call_started_at: float | None = None
    call_completed_at: float | None = None
    execution_started_at: float | None = None
    execution_completed_at: float | None = None
    result_committed_at: float | None = None
    invalidated_at: float | None = None
    invalidation_reason: ToolInvalidationReason | None = None
    candidate_invalidation_reason: CandidateInvalidationReason | None = None
    task: asyncio.Task[ToolSuccess | ToolExecutionFailure] | None = None
    cancellation_observer: asyncio.Task[None] | None = None
    wasted: bool = False

    def __post_init__(self) -> None:
        if self.round_index <= 0:
            raise ValueError("The tool round index must be positive.")
        if self.invocation_id <= 0:
            raise ValueError("The tool invocation ID must be positive.")
        if not self.call_id:
            raise ValueError("The tool call ID cannot be empty.")


@dataclass(frozen=True)
class ModelInvocationResult:
    invocation_id: int
    tool_request: SerializedToolCall | None
    tool_failure: ToolCallFailure | None
    temperature_conversion: TemperatureConversion | None


@dataclass
class ActiveGeneration:
    candidate_id: int
    generation_id: int
    transcript_revision_id: int | None
    prediction_evidence_event_id: str | None
    stable_transcript_prefix: str
    anchored_text: str
    input_audio_sample_position: int
    monotonic_creation_time: float
    causal_prediction: InteractionPrediction | None
    prediction_confidence: float | None
    prompt_messages: tuple[ConversationMessage, ...]
    model_context_prefix: tuple[ModelMessage, ...]
    model_context_turn: ModelConversationTurn
    model_messages: tuple[ModelMessage, ...]
    user_turn_id: int
    release_gate: CandidateReleaseGate
    speculative: bool
    latency: GenerationLatency
    task: asyncio.Task[None] | None = None
    release_task: asyncio.Task[None] | None = None
    response_text: str = ""
    qwen_token_count: int = 0
    invocation_token_counts: dict[int, int] = field(default_factory=dict)
    invocation_ids: list[int] = field(default_factory=list)
    tool_circuit_breaker: ToolCircuitBreaker = field(default_factory=ToolCircuitBreaker)
    tts_sample_count: int = 0
    synthesis_word_count: int = 0
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
    continuation_allowed: asyncio.Event = field(default_factory=asyncio.Event)
    synthesis_budget_available: asyncio.Event = field(default_factory=asyncio.Event)
    tool_executions: list[ToolExecutionJournalEntry] = field(default_factory=list)
    final_answer_text_start: int | None = None
    final_answer_first_sample: int | None = None


@dataclass
class ActiveUserOverlap:
    overlap_id: str
    generation_id: int
    onset_event_id: str
    onset_input_sample: int
    onset_monotonic_time_ns: int
    stream_epoch: int
    turn_epoch: int
    duck_command_id: str | None
    pause_command_id: str | None
    synthesized_source_sample_count_at_onset: int
    resume_command_id: str | None = None
    cancel_command_id: str | None = None
    pause_acknowledged_monotonic_time_ns: int | None = None
    decision_event_id: str | None = None
    decision: ProvisionalOverlapDecision | None = None
    decision_monotonic_time_ns: int | None = None
    decision_input_sample_position: int | None = None
    decision_rendered_output_sample_position: int | None = None
    decision_source_sample_position: int | None = None
    generation_hold_applied: bool = False
    resolution_metrics_recorded: bool = False
    first_applicable_prediction_monotonic_time_ns: int | None = None
    final_endpoint_input_sample: int | None = None


class WebSocketPlaybackSink:
    def __init__(
        self,
        websocket: WebSocket,
        send_lock: asyncio.Lock,
        flow_control: PlaybackFlowControl,
    ) -> None:
        self.websocket = websocket
        self.send_lock = send_lock
        self.flow_control = flow_control

    async def send(self, output: CandidateOutput) -> None:
        match output:
            case ReleasedAudioChunk():
                send_allowed = await self.flow_control.wait_until_send_allowed(
                    output.generation_id,
                    output.start_sample + len(output.pcm_bytes) // PCM_BYTES_PER_SAMPLE,
                )
                if not send_allowed:
                    return
            case _:
                pass
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
        speech_understanding_provider: SpeechUnderstandingProvider,
        language_model: LanguageModel,
        speech_synthesizer: SpeechSynthesizer,
        policy: SessionPolicy,
        tool_executor: ToolExecutor,
        playback_sink: PlaybackSink | None = None,
        playback_policy: PlaybackPolicyConfig | None = None,
        overlap_policy: ProvisionalVadTranscriptOverlapPolicy | None = None,
    ) -> None:
        self.websocket = websocket
        self.speech_detector = speech_detector
        self.speech_understanding = speech_understanding_provider.create_session(stream_epoch=1)
        self.language_model = language_model
        self.speech_synthesizer = speech_synthesizer
        self.policy = policy
        self.tool_executor = tool_executor
        self.playback_controller = PlaybackController(
            source_sample_rate=speech_synthesizer.sample_rate,
            config=playback_policy or PlaybackPolicyConfig(),
        )
        self.overlap_policy = overlap_policy or ProvisionalVadTranscriptOverlapPolicy(
            ProvisionalOverlapPolicyConfig(
                classification_deadline_ms=(policy.overlap_classification_deadline_ms),
                transcript_free_floor_take_deadline_ms=(
                    policy.transcript_free_floor_take_deadline_ms
                ),
                interruption_probability_threshold=policy.floor_taking_overlap_threshold,
                non_floor_feedback_probability_threshold=(
                    policy.non_floor_feedback_overlap_threshold
                ),
            )
        )
        self.session_id = str(uuid4())
        self.audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=AUDIO_QUEUE_MAX_CHUNKS
        )
        self.send_lock = asyncio.Lock()
        self.playback_flow_control = PlaybackFlowControl(
            source_sample_rate=speech_synthesizer.sample_rate,
            maximum_ahead_ms=self.playback_controller.config.maximum_transport_ahead_ms,
        )
        self.playback_sink = playback_sink or WebSocketPlaybackSink(
            websocket,
            self.send_lock,
            self.playback_flow_control,
        )
        self.conversation: list[ConversationMessage] = []
        self.model_context = PrivateModelContext()
        self.tool_execution_journal: list[ToolExecutionJournalEntry] = []
        self.generations: dict[int, ActiveGeneration] = {}
        self.active_generation: ActiveGeneration | None = None
        self.next_generation_id = 1
        self.next_user_turn_id = 1
        self.audio_sample_count = 0
        self.started = False
        self.lifecycle = SessionLifecycle.CREATED
        self.pending_generation_teardown: ActiveGeneration | None = None
        self.latest_transcript_revision: TranscriptRevision | None = None
        self.interaction_prediction_reducer = InteractionPredictionReducer()
        self.predictive_metrics = PredictiveMetrics()
        self.latest_prediction: InteractionPrediction | None = None
        self.first_endpoint_at: float | None = None
        self.first_endpoint_input_sample_position: int | None = None
        self.final_vad_endpoint_at: float | None = None
        self.turn_had_invalidated_candidate = False
        self.latest_user_speech_input_sample: int | None = None
        self.next_audio_sequence_number = 0
        self.last_speech_debug_input_sample: int | None = None
        self.last_speech_debug_silero_speech: bool | None = None
        self.last_speech_debug_assistant_audible: bool | None = None
        self.active_user_overlap: ActiveUserOverlap | None = None
        self.user_overlap_traces: list[ActiveUserOverlap] = []
        self.overlap_metrics = OverlapMetrics()

    @property
    def playback_condition(self) -> PlaybackCondition:
        return self.playback_controller.condition

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
            await self._request_generation_cancellation(
                send_event=False,
                tool_invalidation_reason=ToolInvalidationReason.SESSION_STOPPED,
            )
            await self._await_generation_teardown()
            await self._close_generation_tasks()
            await self.speech_understanding.close()
            for task in (receive_task, recognition_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # The session-level exception path already logged and reported this failure.
                    pass
            self.conversation.clear()
            self.model_context.clear()
            self.generations.clear()
            self.lifecycle = SessionLifecycle.CLOSED
            logger.info(
                "voice predictive metrics: session=%s report=%r",
                self.session_id,
                self.predictive_metrics.report(),
            )
            logger.info(
                "voice playback metrics: session=%s report=%r",
                self.session_id,
                self.playback_controller.metrics.report(),
            )
            logger.info(
                "voice overlap metrics: session=%s report=%r",
                self.session_id,
                self.overlap_metrics.report(),
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
                    await self._record_playback_started(event)
                case PlaybackCompleteEvent():
                    self._complete_playback(event)
                case PlaybackClockEvent():
                    self._record_playback_clock(event)
                case PlaybackProgressEvent():
                    self._acknowledge_playback(event)
                case PlaybackStoppedEvent():
                    self._stop_playback(event)
                case PlaybackCommandAcknowledgementEvent():
                    await self._acknowledge_playback_command(event)
                case SessionStopEvent():
                    await self.audio_queue.put(None)
                    return

    async def _start(self, event: SessionStartEvent) -> None:
        if self.started:
            raise ValueError("session.start may only be sent once.")
        if event.input_sample_rate != INPUT_SAMPLE_RATE:
            raise ValueError(f"Input sample rate must be {INPUT_SAMPLE_RATE} Hz.")
        self.tool_executor.configure_session(event.local_time_zone)
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
        speech_understanding = self.speech_understanding
        speech_active = False
        previous_chunk_was_speech = False
        silent_samples = 0
        vad_speculation_pending = False
        pre_roll_chunks: deque[CapturedAudioChunk] = deque()
        pre_roll_samples = 0
        required_silent_samples = _milliseconds_to_samples(self.policy.silence_duration_ms)
        required_normal_turn_commit_silent_samples = _milliseconds_to_samples(
            self.policy.minimum_normal_turn_commit_silence_ms
        )
        required_vad_speculation_debounce_samples = _milliseconds_to_samples(
            self.policy.vad_speculation_debounce_ms
        )
        required_pending_silence_speculation_samples = _milliseconds_to_samples(
            self.policy.pending_silence_speculation_ms
        )
        maximum_pre_roll_samples = _milliseconds_to_samples(self.policy.pre_roll_duration_ms)
        required_overlap_rearm_silent_samples = _milliseconds_to_samples(
            self.policy.overlap_rearm_silence_ms
        )
        overlap_rearm_silent_samples: int | None = None
        try:
            while (pcm_bytes := await self.audio_queue.get()) is not None:
                sample_count = _pcm_sample_count(pcm_bytes)
                start_input_sample = self.audio_sample_count
                self.audio_sample_count += sample_count
                observation_time_ns = time.perf_counter_ns()
                try:
                    speech_detection = self.speech_detector.process_audio(pcm_bytes)
                except Exception as error:
                    raise VoiceComponentError(
                        VoiceComponent.SPEECH_DETECTION,
                        VoiceOperation.DETECT_SPEECH,
                        str(error),
                    ) from error
                observed_playback_condition = self.playback_controller.observe_condition(
                    observation_time_ns
                )
                chunk = CapturedAudioChunk(
                    pcm16=pcm_bytes,
                    sequence_number=self.next_audio_sequence_number,
                    start_input_sample=start_input_sample,
                    end_input_sample=self.audio_sample_count,
                    monotonic_observation_time_ns=observation_time_ns,
                    stream_epoch=speech_understanding.stream_epoch,
                    turn_epoch=speech_understanding.turn_epoch,
                    silero_evidence=SileroEvidence(
                        is_speech=speech_detection.is_speech,
                        monotonic_time_ns=observation_time_ns,
                    ),
                    playback_condition=observed_playback_condition,
                )
                is_speech = speech_detection.is_speech
                self.next_audio_sequence_number += 1
                await self._send_periodic_speech_debug(chunk)
                if not speech_active:
                    pre_roll_chunks.append(chunk)
                    pre_roll_samples += sample_count
                    while pre_roll_samples > maximum_pre_roll_samples:
                        pre_roll_samples -= _pcm_sample_count(pre_roll_chunks.popleft().pcm16)
                    if overlap_rearm_silent_samples is not None:
                        if is_speech:
                            overlap_rearm_silent_samples = 0
                        else:
                            overlap_rearm_silent_samples += sample_count
                            if (
                                overlap_rearm_silent_samples
                                >= required_overlap_rearm_silent_samples
                            ):
                                overlap_rearm_silent_samples = None
                        continue
                    if not is_speech:
                        continue
                    speech_active = True
                    previous_chunk_was_speech = True
                    silent_samples = 0
                    vad_speculation_pending = False
                    overlap_started = await self._start_user_overlap(chunk)
                    if not overlap_started:
                        await self._request_generation_cancellation(
                            send_event=True,
                            tool_invalidation_reason=ToolInvalidationReason.USER_ACTIVITY,
                        )
                    logger.info("speech started: session=%s", self.session_id)
                    await self._send_speech_state(VoiceServerEventType.VAD_STARTED)
                    for pre_roll_chunk in pre_roll_chunks:
                        prediction = await self._add_speech_understanding_audio(
                            speech_understanding,
                            pre_roll_chunk,
                        )
                        if prediction is not None:
                            self.latest_prediction = prediction
                            await self._handle_prediction(prediction)
                    pre_roll_chunks.clear()
                    pre_roll_samples = 0
                    await self._evaluate_active_overlap(
                        speech_understanding,
                        speech_active_now=True,
                        current_input_sample=chunk.end_input_sample,
                    )
                    continue

                vad_endpoint_detected = not is_speech and previous_chunk_was_speech
                if is_speech and not previous_chunk_was_speech:
                    self.final_vad_endpoint_at = None
                    vad_speculation_pending = False
                    await self._invalidate_speculative_candidate(
                        CandidateInvalidationReason.USER_ACTIVITY_RESUMED
                    )
                if vad_endpoint_detected and self.active_user_overlap is not None:
                    await speech_understanding.settle_predictions(
                        self.policy.overlap_prediction_settle_ms / 1_000
                    )
                prediction = await self._add_speech_understanding_audio(
                    speech_understanding,
                    chunk,
                )
                prediction_handled = False
                generation = self.active_generation
                if (
                    is_speech
                    and speech_detection.pending_silence_samples == 0
                    and generation is not None
                    and generation.speculative
                    and generation.causal_prediction is not None
                    and generation.causal_prediction.stamp.source
                    is CausalSource.SILERO_PENDING_SILENCE
                ):
                    await self._invalidate_speculative_candidate(
                        CandidateInvalidationReason.USER_ACTIVITY_RESUMED
                    )
                if (
                    self.policy.vad_speculation_enabled
                    and is_speech
                    and speech_detection.pending_silence_samples
                    >= required_pending_silence_speculation_samples
                    and self.active_generation is None
                ):
                    if prediction is not None:
                        self.latest_prediction = prediction
                        await self._handle_prediction(prediction)
                        prediction_handled = True
                    if self.active_generation is None:
                        pending_prediction = self._silero_prediction(
                            chunk,
                            source=CausalSource.SILERO_PENDING_SILENCE,
                            model_name="silero-vad-pending-silence",
                        )
                        self.latest_prediction = pending_prediction
                        await self._handle_prediction(pending_prediction)
                if vad_endpoint_detected:
                    endpoint_at = time.perf_counter()
                    self.final_vad_endpoint_at = endpoint_at
                    generation = self.active_generation
                    if generation is not None and generation.speculative:
                        generation.latency.final_vad_endpoint_at = endpoint_at
                    if self.first_endpoint_at is None:
                        self.first_endpoint_at = endpoint_at
                        self.first_endpoint_input_sample_position = self.audio_sample_count
                        if generation is not None and generation.latency.first_endpoint is None:
                            generation.latency.first_endpoint_at = self.first_endpoint_at
                            generation.latency.first_endpoint = MediaLatencyPoint(
                                monotonic_time_seconds=self.first_endpoint_at,
                                input_sample_position=self.audio_sample_count,
                                output_sample_position=None,
                                text_offset=None,
                            )
                if vad_endpoint_detected and self.policy.vad_speculation_enabled:
                    vad_speculation_pending = True
                prospective_silent_samples = silent_samples + sample_count
                if (
                    vad_speculation_pending
                    and not is_speech
                    and prospective_silent_samples >= required_vad_speculation_debounce_samples
                ):
                    if prediction is not None:
                        self.latest_prediction = prediction
                        await self._handle_prediction(prediction)
                        prediction_handled = True
                    if self.active_generation is None:
                        prediction = self._silero_prediction(
                            chunk,
                            source=CausalSource.SILERO_VAD,
                            model_name="silero-vad",
                        )
                        prediction_handled = False
                    vad_speculation_pending = False
                if prediction is not None:
                    self.latest_prediction = prediction
                    if not prediction_handled:
                        await self._handle_prediction(prediction)
                    if self.active_generation is not None:
                        vad_speculation_pending = False
                overlap_resolution = await self._evaluate_active_overlap(
                    speech_understanding,
                    speech_active_now=is_speech,
                    current_input_sample=chunk.end_input_sample,
                )
                if overlap_resolution is OverlapResolutionKind.NON_FLOOR_TAKING:
                    speech_active = False
                    previous_chunk_was_speech = False
                    silent_samples = 0
                    vad_speculation_pending = False
                    self.latest_transcript_revision = None
                    self.interaction_prediction_reducer.reset_turn()
                    self.latest_prediction = None
                    self.first_endpoint_at = None
                    self.first_endpoint_input_sample_position = None
                    self.final_vad_endpoint_at = None
                    self.turn_had_invalidated_candidate = False
                    self.latest_user_speech_input_sample = None
                    overlap_rearm_silent_samples = 0
                    await self._send_speech_state(VoiceServerEventType.VAD_STOPPED)
                    continue
                if is_speech:
                    silent_samples = 0
                    vad_speculation_pending = False
                    previous_chunk_was_speech = True
                    continue
                silent_samples += sample_count
                previous_chunk_was_speech = False
                prediction_commits = (
                    prediction is not None
                    and silent_samples >= required_normal_turn_commit_silent_samples
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
                vad_speculation_pending = False
                await self._send_speech_state(VoiceServerEventType.VAD_STOPPED)
                await self._finalize_turn(speech_understanding)
                self.latest_transcript_revision = None
                self.interaction_prediction_reducer.reset_turn()
                self.latest_prediction = None
                self.first_endpoint_at = None
                self.first_endpoint_input_sample_position = None
                self.final_vad_endpoint_at = None
                self.turn_had_invalidated_candidate = False
                self.latest_user_speech_input_sample = None
        finally:
            await speech_understanding.close()

    async def _start_user_overlap(self, chunk: CapturedAudioChunk) -> bool:
        generation = self.active_generation
        condition = self.playback_condition
        if (
            generation is None
            or generation.speculative
            or generation.cancelled
            or generation.playback_complete
            or not generation.accepts_playback
            or condition.generation_id != generation.generation_id
        ):
            return False
        onset_event_id = str(uuid4())
        overlap_id = str(uuid4())
        duck_command: PlaybackCommandEvent | None = None
        pause_command: PlaybackCommandEvent | None = None
        next_boundary: int | None = None
        if condition.output_sample_rate is not None:
            next_boundary = min(
                (
                    boundary_sample
                    for boundary_sample in generation.boundary_samples.values()
                    if boundary_sample > condition.latest_source_sample_position
                ),
                default=None,
            )
            duck_command = self.playback_controller.issue_duck(
                generation_id=generation.generation_id,
                causal_event_id=onset_event_id,
                causal_source=CausalSource.SILERO_VAD,
                stream_epoch=chunk.stream_epoch,
                turn_epoch=chunk.turn_epoch,
                confidence=1.0,
            )
            pause_command = self.playback_controller.issue_pause(
                generation_id=generation.generation_id,
                causal_event_id=onset_event_id,
                causal_source=CausalSource.SILERO_VAD,
                stream_epoch=chunk.stream_epoch,
                turn_epoch=chunk.turn_epoch,
                confidence=1.0,
                requested_boundary_source_sample_position=next_boundary,
            )
        self.active_user_overlap = ActiveUserOverlap(
            overlap_id=overlap_id,
            generation_id=generation.generation_id,
            onset_event_id=onset_event_id,
            onset_input_sample=chunk.start_input_sample,
            onset_monotonic_time_ns=chunk.monotonic_observation_time_ns,
            stream_epoch=chunk.stream_epoch,
            turn_epoch=chunk.turn_epoch,
            duck_command_id=None if duck_command is None else duck_command.command_id,
            pause_command_id=None if pause_command is None else pause_command.command_id,
            synthesized_source_sample_count_at_onset=generation.tts_sample_count,
        )
        if not condition.assistant_audible:
            generation.continuation_allowed.clear()
            generation.synthesis_budget_available.clear()
            self.active_user_overlap.generation_hold_applied = True
        assert self.active_user_overlap is not None
        self.user_overlap_traces.append(self.active_user_overlap)
        self.overlap_metrics.record_started()
        if duck_command is not None and pause_command is not None:
            await self._send_event(duck_command)
            await self._send_event(pause_command)
        logger.info(
            "user overlap started: session=%s overlap=%s generation=%d input_sample=%d "
            "assistant_audible=%s duck_command=%s pause_command=%s boundary_sample=%s",
            self.session_id,
            overlap_id,
            generation.generation_id,
            chunk.start_input_sample,
            condition.assistant_audible,
            None if duck_command is None else duck_command.command_id,
            None if pause_command is None else pause_command.command_id,
            next_boundary,
        )
        return True

    async def _evaluate_active_overlap(
        self,
        speech_understanding: SpeechUnderstandingSession,
        speech_active_now: bool,
        current_input_sample: int,
    ) -> OverlapResolutionKind | None:
        overlap = self.active_user_overlap
        if overlap is None:
            return None
        elapsed_ms = (
            max(current_input_sample - overlap.onset_input_sample, 0) * 1_000 // INPUT_SAMPLE_RATE
        )
        if speech_active_now:
            overlap.final_endpoint_input_sample = None
        elif overlap.final_endpoint_input_sample is None:
            overlap.final_endpoint_input_sample = current_input_sample
        generation = self.generations[overlap.generation_id]
        if (
            elapsed_ms >= self.playback_controller.config.generation_boundary_hold_ms
            and not overlap.generation_hold_applied
        ):
            overlap.generation_hold_applied = True
            generation.continuation_allowed.clear()
            generation.synthesis_budget_available.clear()
        revision = self.latest_transcript_revision
        transcript = (
            ""
            if revision is None
            else f"{revision.stable_prefix}{revision.volatile_suffix}".strip()
        )
        prediction = self.latest_prediction
        prediction_is_causal = (
            prediction is not None
            and prediction.stamp.input_end_sample >= overlap.onset_input_sample
        )
        decision = self.overlap_policy.classify(
            OverlapEvidence(
                elapsed_ms=elapsed_ms,
                speech_active=speech_active_now,
                transcript=transcript,
                transcript_event_id=None if revision is None else revision.stamp.event_id,
                interruption_probability=(
                    prediction.p_user_interruption if prediction_is_causal else None
                ),
                interruption_evidence_event_id=(
                    prediction.stamp.event_id if prediction_is_causal else None
                ),
                non_floor_feedback_probability=(
                    prediction.p_user_backchannel if prediction_is_causal else None
                ),
                non_floor_feedback_evidence_event_id=(
                    prediction.stamp.event_id if prediction_is_causal else None
                ),
            )
        )
        if decision.kind is OverlapResolutionKind.UNRESOLVED:
            return decision.kind
        overlap.decision = decision
        overlap.decision_event_id = str(uuid4())
        overlap.decision_monotonic_time_ns = time.perf_counter_ns()
        overlap.decision_input_sample_position = current_input_sample
        overlap.decision_rendered_output_sample_position = (
            self.playback_condition.latest_output_sample_position
        )
        overlap.decision_source_sample_position = (
            self.playback_condition.latest_source_sample_position
        )
        if decision.kind is OverlapResolutionKind.NON_FLOOR_TAKING:
            active_generation = self.active_generation
            overlap_outlived_generation = (
                active_generation is None
                or active_generation.generation_id != overlap.generation_id
            )
            if overlap_outlived_generation and not self._normal_turn_commit_silence_elapsed(
                overlap,
                current_input_sample,
            ):
                return OverlapResolutionKind.UNRESOLVED
            return await self._finalize_non_floor_taking_overlap(
                speech_understanding,
                overlap,
                decision,
                elapsed_ms,
            )
        await self._record_overlap_resolution(overlap, decision, generation)
        await self._promote_overlap_to_user_turn(overlap, decision, transcript)
        return decision.kind

    def _normal_turn_commit_silence_elapsed(
        self,
        overlap: ActiveUserOverlap,
        current_input_sample: int,
    ) -> bool:
        endpoint_input_sample = overlap.final_endpoint_input_sample
        if endpoint_input_sample is None:
            return False
        required_silence_samples = _milliseconds_to_samples(
            self.policy.minimum_normal_turn_commit_silence_ms
        )
        return current_input_sample - endpoint_input_sample >= required_silence_samples

    async def _finalize_non_floor_taking_overlap(
        self,
        speech_understanding: SpeechUnderstandingSession,
        overlap: ActiveUserOverlap,
        provisional_decision: ProvisionalOverlapDecision,
        elapsed_ms: int,
    ) -> OverlapResolutionKind:
        generation = self.active_generation
        assert overlap.decision_event_id is not None
        if generation is None or generation.generation_id != overlap.generation_id:
            logger.info(
                "user overlap outlived assistant generation; committing as a normal turn: "
                "session=%s overlap=%s generation=%d",
                self.session_id,
                overlap.overlap_id,
                overlap.generation_id,
            )
            self.active_user_overlap = None
            await self._finalize_turn(speech_understanding)
            return OverlapResolutionKind.NON_FLOOR_TAKING
        finalization_started_at = time.perf_counter()
        finalization_task = asyncio.create_task(speech_understanding.finalize_turn())
        resumed_early = False
        try:
            try:
                finalized_turn = await asyncio.wait_for(
                    asyncio.shield(finalization_task),
                    timeout=self.policy.overlap_finalization_grace_ms / 1_000,
                )
            except TimeoutError:
                await self._resume_provisional_overlap(generation, overlap, provisional_decision)
                resumed_early = True
                finalized_turn = await finalization_task
            final_text = finalized_turn.text.strip()
            for event in speech_understanding.drain_events():
                if isinstance(event, TranscriptRevision):
                    self.latest_transcript_revision = event
                elif isinstance(event, SpeechUnderstandingDegradedEvent):
                    logger.warning(
                        "speech understanding degraded during ephemeral overlap finalization: "
                        "session=%s component=%s reason=%s dropped_observations=%d",
                        self.session_id,
                        event.component,
                        event.reason,
                        event.dropped_observation_count,
                    )
        except asyncio.CancelledError:
            finalization_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await finalization_task
            raise
        except Exception as error:
            raise _component_error(
                error,
                component=VoiceComponent.ASR,
                operation=VoiceOperation.TRANSCRIBE,
            ) from error
        transcript_decision = self.overlap_policy.classify(
            OverlapEvidence(
                elapsed_ms=elapsed_ms,
                speech_active=False,
                transcript=final_text,
                transcript_event_id=(
                    None
                    if finalized_turn.transcript_revision is None
                    else finalized_turn.transcript_revision.stamp.event_id
                ),
                interruption_probability=None,
                interruption_evidence_event_id=None,
            )
        )
        final_decision = (
            provisional_decision
            if transcript_decision.kind is OverlapResolutionKind.NON_FLOOR_TAKING
            else transcript_decision
        )
        overlap.decision = final_decision
        decision_event_id = str(uuid4())
        overlap.decision_event_id = decision_event_id
        overlap.decision_monotonic_time_ns = time.perf_counter_ns()
        overlap.decision_input_sample_position = self.audio_sample_count
        overlap.decision_rendered_output_sample_position = (
            self.playback_condition.latest_output_sample_position
        )
        overlap.decision_source_sample_position = (
            self.playback_condition.latest_source_sample_position
        )
        if final_decision.kind is not OverlapResolutionKind.NON_FLOOR_TAKING:
            generation = self.generations[overlap.generation_id]
            await self._record_overlap_resolution(overlap, final_decision, generation)
            await self._promote_overlap_to_user_turn(overlap, final_decision, final_text)
            finalized_at = time.perf_counter()
            await self._commit_finalized_user_turn(
                final_text=final_text,
                committed_at=finalization_started_at,
                finalized_at=finalized_at,
                finalization_seconds=finalized_at - finalization_started_at,
            )
            return OverlapResolutionKind.NON_FLOOR_TAKING
        if not resumed_early:
            await self._resume_provisional_overlap(generation, overlap, final_decision)
        await self._record_overlap_resolution(overlap, final_decision, generation)
        self.active_user_overlap = None
        logger.info(
            "ephemeral user overlap resolved: session=%s overlap=%s generation=%d "
            "duration_ms=%d transcript=%r decision=%s",
            self.session_id,
            overlap.overlap_id,
            overlap.generation_id,
            elapsed_ms,
            final_text,
            provisional_decision.kind,
        )
        return OverlapResolutionKind.NON_FLOOR_TAKING

    async def _resume_provisional_overlap(
        self,
        generation: ActiveGeneration,
        overlap: ActiveUserOverlap,
        decision: ProvisionalOverlapDecision,
    ) -> None:
        resume_command = self.playback_controller.issue_resume(
            generation_id=generation.generation_id,
            causal_event_id=overlap.decision_event_id,
            causal_source=decision.causal_source,
            stream_epoch=overlap.stream_epoch,
            turn_epoch=overlap.turn_epoch,
            confidence=decision.confidence,
        )
        generation.continuation_allowed.set()
        generation.synthesis_budget_available.set()
        if resume_command is not None:
            overlap.resume_command_id = resume_command.command_id
            await self._send_event(resume_command)

    async def _record_overlap_resolution(
        self,
        overlap: ActiveUserOverlap,
        decision: ProvisionalOverlapDecision,
        generation: ActiveGeneration,
    ) -> None:
        if overlap.resolution_metrics_recorded:
            return
        decision_time_ns = overlap.decision_monotonic_time_ns
        assert decision_time_ns is not None
        overlap.resolution_metrics_recorded = True
        self.overlap_metrics.record_resolution(
            kind=decision.kind,
            onset_monotonic_time_ns=overlap.onset_monotonic_time_ns,
            decision_monotonic_time_ns=decision_time_ns,
            generated_source_sample_count=max(
                generation.tts_sample_count - overlap.synthesized_source_sample_count_at_onset,
                0,
            ),
        )
        await self._send_event(
            InteractionPolicyDebugEvent(
                decision=decision.kind,
                reason=decision.reason,
                decision_latency_ms=(decision_time_ns - overlap.onset_monotonic_time_ns)
                / 1_000_000,
                first_applicable_prediction_latency_ms=(
                    None
                    if overlap.first_applicable_prediction_monotonic_time_ns is None
                    else (
                        overlap.first_applicable_prediction_monotonic_time_ns
                        - overlap.onset_monotonic_time_ns
                    )
                    / 1_000_000
                ),
                causal_source=decision.causal_source,
            )
        )

    async def _promote_overlap_to_user_turn(
        self,
        overlap: ActiveUserOverlap,
        decision: ProvisionalOverlapDecision,
        transcript: str,
    ) -> None:
        decision_event_id = overlap.decision_event_id
        assert decision_event_id is not None
        self.active_user_overlap = None
        if transcript:
            await self._send_transcript(VoiceServerEventType.TRANSCRIPT_PARTIAL, transcript)
        cancel_command = await self._request_generation_cancellation(
            send_event=True,
            causal_event_id=decision_event_id,
            causal_source=decision.causal_source,
            confidence=decision.confidence,
            tool_invalidation_reason=ToolInvalidationReason.RESPONSE_REQUIRING_OVERLAP,
        )
        overlap.cancel_command_id = None if cancel_command is None else cancel_command.command_id
        logger.info(
            "user overlap promoted: session=%s overlap=%s generation=%d decision=%s "
            "reason=%s fast_path=%s transcript=%r",
            self.session_id,
            overlap.overlap_id,
            overlap.generation_id,
            decision.kind,
            decision.reason,
            decision.fast_path,
            transcript,
        )

    async def _add_speech_understanding_audio(
        self,
        speech_understanding: SpeechUnderstandingSession,
        chunk: CapturedAudioChunk,
    ) -> InteractionPrediction | None:
        if chunk.silero_evidence.is_speech:
            self.latest_user_speech_input_sample = chunk.end_input_sample
        self.interaction_prediction_reducer.observe_audio_chunk(chunk)
        try:
            await speech_understanding.add_audio(chunk)
        except Exception as error:
            raise _component_error(
                error,
                component=VoiceComponent.ASR,
                operation=VoiceOperation.TRANSCRIBE,
            ) from error
        latest_prediction: InteractionPrediction | None = None
        for event in speech_understanding.drain_events():
            match event:
                case TranscriptRevision():
                    self.latest_transcript_revision = event
                    if self.active_user_overlap is None:
                        await self._send_transcript(
                            VoiceServerEventType.TRANSCRIPT_PARTIAL,
                            f"{event.stable_prefix}{event.volatile_suffix}".strip(),
                        )
                    await self._validate_candidate_revision(event)
                case SpeechUnderstandingDegradedEvent():
                    logger.warning(
                        "speech understanding degraded: session=%s component=%s reason=%s "
                        "dropped_observations=%d",
                        self.session_id,
                        event.component,
                        event.reason,
                        event.dropped_observation_count,
                    )
                    await self._send_event(
                        SpeechUnderstandingDebugEvent(
                            adapter_status=TurnAdapterStatus.DEGRADED,
                            silero_speech=chunk.silero_evidence.is_speech,
                            assistant_audible=chunk.playback_condition.assistant_audible,
                            user_yield_probability=None,
                            turn_completion_probability=None,
                            floor_take_probability=None,
                            non_floor_feedback_probability=None,
                            inference_latency_ms=None,
                            observed_audio_time_ms=(
                                event.stamp.observed_through_input_sample
                                * 1_000
                                // INPUT_SAMPLE_RATE
                            ),
                        )
                    )
                case _:
                    prediction = self.interaction_prediction_reducer.reduce(event)
                    if prediction is not None:
                        disposition = self._prediction_disposition(
                            prediction,
                            current_input_sample=chunk.end_input_sample,
                        )
                        await self._send_event(
                            SpeechUnderstandingDebugEvent(
                                adapter_status=TurnAdapterStatus.ACTIVE,
                                silero_speech=chunk.silero_evidence.is_speech,
                                assistant_audible=chunk.playback_condition.assistant_audible,
                                user_yield_probability=prediction.p_user_yield,
                                turn_completion_probability=prediction.p_turn_completion,
                                floor_take_probability=prediction.p_user_interruption,
                                non_floor_feedback_probability=(prediction.p_user_backchannel),
                                inference_latency_ms=(
                                    prediction.stamp.emission_monotonic_time_ns
                                    - prediction.stamp.observation_monotonic_time_ns
                                )
                                / 1_000_000,
                                observed_audio_time_ms=(
                                    prediction.stamp.observed_through_input_sample
                                    * 1_000
                                    // INPUT_SAMPLE_RATE
                                ),
                                prediction_disposition=disposition,
                            )
                        )
                        if disposition is not TurnPredictionDisposition.APPLICABLE:
                            continue
                        overlap = self.active_user_overlap
                        if (
                            overlap is not None
                            and prediction.stamp.input_end_sample >= overlap.onset_input_sample
                            and overlap.first_applicable_prediction_monotonic_time_ns is None
                        ):
                            overlap.first_applicable_prediction_monotonic_time_ns = (
                                time.perf_counter_ns()
                            )
                        latest_prediction = prediction
        return latest_prediction

    async def _send_periodic_speech_debug(self, chunk: CapturedAudioChunk) -> None:
        silero_speech = chunk.silero_evidence.is_speech
        assistant_audible = chunk.playback_condition.assistant_audible
        state_changed = (
            silero_speech != self.last_speech_debug_silero_speech
            or assistant_audible != self.last_speech_debug_assistant_audible
        )
        interval_elapsed = (
            self.last_speech_debug_input_sample is None
            or chunk.end_input_sample - self.last_speech_debug_input_sample
            >= SPEECH_DEBUG_INTERVAL_SAMPLES
        )
        if not state_changed and not interval_elapsed:
            return
        self.last_speech_debug_input_sample = chunk.end_input_sample
        self.last_speech_debug_silero_speech = silero_speech
        self.last_speech_debug_assistant_audible = assistant_audible
        await self._send_event(
            SpeechUnderstandingDebugEvent(
                adapter_status=self.speech_understanding.turn_adapter_status,
                silero_speech=silero_speech,
                assistant_audible=assistant_audible,
                user_yield_probability=None,
                turn_completion_probability=None,
                floor_take_probability=None,
                non_floor_feedback_probability=None,
                inference_latency_ms=None,
                observed_audio_time_ms=chunk.end_input_sample * 1_000 // INPUT_SAMPLE_RATE,
            )
        )

    def _prediction_disposition(
        self,
        prediction: InteractionPrediction,
        current_input_sample: int,
    ) -> TurnPredictionDisposition:
        observed_through_input_sample = prediction.stamp.observed_through_input_sample
        if observed_through_input_sample > current_input_sample:
            raise ValueError("Prediction cannot observe beyond received input audio.")
        maximum_lag_samples = _milliseconds_to_samples(self.policy.maximum_prediction_lag_ms)
        lag_samples = current_input_sample - observed_through_input_sample
        if lag_samples > maximum_lag_samples:
            logger.info(
                "stale interaction prediction discarded: session=%s event=%s lag_samples=%d "
                "maximum_lag_samples=%d",
                self.session_id,
                prediction.stamp.event_id,
                lag_samples,
                maximum_lag_samples,
            )
            return TurnPredictionDisposition.REJECTED_STALE
        latest_user_speech_input_sample = self.latest_user_speech_input_sample
        if (
            self.active_user_overlap is None
            and latest_user_speech_input_sample is not None
            and latest_user_speech_input_sample > observed_through_input_sample
        ):
            logger.info(
                "causally superseded interaction prediction discarded: session=%s event=%s "
                "observed_through_sample=%d later_speech_sample=%d",
                self.session_id,
                prediction.stamp.event_id,
                observed_through_input_sample,
                latest_user_speech_input_sample,
            )
            return TurnPredictionDisposition.REJECTED_SUPERSEDED
        return TurnPredictionDisposition.APPLICABLE

    def _silero_prediction(
        self,
        chunk: CapturedAudioChunk,
        source: CausalSource,
        model_name: str,
    ) -> InteractionPrediction:
        return InteractionPrediction(
            stamp=TraceStamp(
                event_id=str(uuid4()),
                parent_event_ids=(),
                stream_epoch=chunk.stream_epoch,
                turn_epoch=chunk.turn_epoch,
                inference_step=chunk.sequence_number,
                observation_id=f"audio:{chunk.stream_epoch}:{chunk.sequence_number}",
                observation_monotonic_time_ns=chunk.monotonic_observation_time_ns,
                emission_monotonic_time_ns=time.perf_counter_ns(),
                encoder_frame_start=None,
                encoder_frame_end=None,
                input_start_sample=chunk.start_input_sample,
                input_end_sample=chunk.end_input_sample,
                observed_through_input_sample=chunk.end_input_sample,
                input_sample_position=chunk.end_input_sample,
                output_sample_position=chunk.playback_condition.latest_output_sample_position,
                conditioned_transcript_revision_id=None,
                conditioned_playback_event_id=chunk.playback_condition.event_id,
                source=source,
                model_name=model_name,
                model_revision=None,
            ),
            p_user_speech=0.0,
            p_user_yield=self.policy.vad_endpoint_yield_probability,
            p_user_backchannel=0.0,
            p_user_interruption=0.0,
            p_turn_completion=self.policy.vad_endpoint_yield_probability,
            p_continuation_pause=0.0,
            future_user_activity_horizons=(),
            assistant_playback_state=chunk.playback_condition.state,
            confidence=self.policy.vad_endpoint_confidence,
        )

    async def _handle_prediction(self, prediction: InteractionPrediction) -> None:
        generation = self.active_generation
        if generation is not None and generation.speculative:
            if prediction.p_user_interruption >= self.policy.floor_taking_overlap_threshold:
                await self._invalidate_speculative_candidate(
                    CandidateInvalidationReason.FLOOR_TAKING_OVERLAP
                )
            elif (
                generation.causal_prediction is not None
                and generation.causal_prediction.stamp.source
                not in (CausalSource.SILERO_PENDING_SILENCE, CausalSource.SILERO_VAD)
                and prediction.p_user_speech >= self.policy.decisive_hold_threshold
            ):
                await self._invalidate_speculative_candidate(
                    CandidateInvalidationReason.PREDICTION_RETURNED_TO_HOLD
                )
        revision = self.latest_transcript_revision
        prompted_text = (
            "" if revision is None else self._speculative_prompt_text(revision, prediction)
        )
        if (
            self.active_generation is None
            and revision is not None
            and prompted_text
            and prediction.confidence >= self.policy.speculative_minimum_confidence
            and (
                prediction.p_user_yield >= self.policy.speculative_yield_threshold
                or prediction.p_turn_completion >= self.policy.speculative_turn_completion_threshold
            )
        ):
            await self._start_speculative_candidate(revision, prediction)

    async def _validate_candidate_revision(self, revision: TranscriptRevision) -> None:
        generation = self.active_generation
        if generation is None or not generation.speculative:
            return
        prediction = generation.causal_prediction
        assert prediction is not None
        invalidation_reason = candidate_revision_invalidation_reason(
            source=prediction.stamp.source,
            stable_prefix=generation.stable_transcript_prefix,
            prompted_text=generation.anchored_text,
            revised_stable_prefix=revision.stable_prefix,
            revised_volatile_suffix=revision.volatile_suffix,
        )
        if invalidation_reason is not None:
            await self._invalidate_speculative_candidate(invalidation_reason)

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
        generation.latency.final_vad_endpoint_at = self.final_vad_endpoint_at
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
            "input_sample=%d confidence=%.3f source=%s",
            self.session_id,
            generation.generation_id,
            revision.revision_id,
            revision.audio_sample_position,
            prediction.confidence,
            prediction.stamp.source,
        )

    @staticmethod
    def _speculative_prompt_text(
        revision: TranscriptRevision,
        prediction: InteractionPrediction,
    ) -> str:
        if prediction.stamp.source in (
            CausalSource.SILERO_PENDING_SILENCE,
            CausalSource.SILERO_VAD,
        ):
            return f"{revision.stable_prefix}{revision.volatile_suffix}".strip()
        return revision.stable_prefix.strip()

    async def _finalize_turn(
        self,
        speech_understanding: SpeechUnderstandingSession,
    ) -> None:
        committed_at = time.perf_counter()
        generation = self.active_generation
        if generation is not None and generation.speculative:
            self._record_generation_commit_readiness(generation, committed_at)
        logger.info("transcription finalization started: session=%s", self.session_id)
        try:
            finalized_turn = await speech_understanding.finalize_turn()
            final_text = finalized_turn.text.strip()
            for event in speech_understanding.drain_events():
                if isinstance(event, TranscriptRevision):
                    self.latest_transcript_revision = event
                elif isinstance(event, SpeechUnderstandingDegradedEvent):
                    logger.warning(
                        "speech understanding degraded during finalization: "
                        "session=%s component=%s reason=%s dropped_observations=%d",
                        self.session_id,
                        event.component,
                        event.reason,
                        event.dropped_observation_count,
                    )
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
        await self._commit_finalized_user_turn(
            final_text=final_text,
            committed_at=committed_at,
            finalized_at=finalized_at,
            finalization_seconds=finalization_seconds,
        )

    async def _commit_finalized_user_turn(
        self,
        final_text: str,
        committed_at: float,
        finalized_at: float,
        finalization_seconds: float,
    ) -> None:
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
            self.next_user_turn_id += 1
            return
        await self._await_generation_teardown()
        await self._start_authoritative_generation(
            final_text=final_text,
            finalization_seconds=finalization_seconds,
            committed_at=committed_at,
            finalized_at=finalized_at,
        )
        self.next_user_turn_id += 1

    async def _promote_candidate(self, generation: ActiveGeneration) -> None:
        assert self.active_generation is generation
        promoted_at = time.perf_counter()
        if generation.latency.commit_readiness is None:
            self._record_generation_commit_readiness(
                generation,
                _require_timestamp(generation.latency.turn_committed_at, "turn commitment"),
            )
        generation.speculative = False
        self.model_context.commit_turn(generation.model_context_turn)
        for tool in generation.tool_executions:
            if tool.result_commit_status is ToolResultCommitStatus.GENERATION_LOCAL_COMMITTED:
                tool.result_commit_status = ToolResultCommitStatus.SESSION_COMMITTED
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
        self.playback_controller.replace_generation(generation.generation_id)
        self.playback_flow_control.replace_generation(generation.generation_id)
        generation.release_task = asyncio.create_task(self._release_candidate_outputs(generation))
        generation_started_at = generation.latency.generation_started_at
        hidden_work_seconds = (
            0.0 if generation_started_at is None else max(promoted_at - generation_started_at, 0.0)
        )
        self.predictive_metrics.record_promotion(
            hidden_work_seconds=hidden_work_seconds,
            hidden_qwen_tokens=generation.qwen_token_count,
            hidden_tts_samples=generation.release_gate.buffered_pcm_sample_count,
        )
        logger.info(
            "speculative candidate promoted: session=%s generation=%d hidden_work_ms=%.1f "
            "buffered_tts_samples=%d",
            self.session_id,
            generation.generation_id,
            hidden_work_seconds * 1_000,
            generation.release_gate.buffered_pcm_sample_count,
        )

    async def _release_candidate_outputs(self, generation: ActiveGeneration) -> None:
        try:
            await generation.release_gate.release()
            self._record_buffered_first_release(generation)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._handle_generation_failure(generation, error)

    def _record_buffered_first_release(self, generation: ActiveGeneration) -> None:
        if generation.first_release_recorded:
            return
        first_released_pcm_at = generation.release_gate.first_released_pcm_at
        if first_released_pcm_at is None:
            return
        generation.latency.first_audio_sent_at = first_released_pcm_at
        generation.latency.first_released_pcm = MediaLatencyPoint(
            monotonic_time_seconds=first_released_pcm_at,
            input_sample_position=self.audio_sample_count,
            output_sample_position=generation.release_gate.first_released_pcm_start_sample,
            text_offset=None,
        )
        self._record_first_released_pcm(generation)
        self._log_first_audio_latency(generation)

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
                if self.latest_transcript_revision is None
                else self.latest_transcript_revision.revision_id
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
        generation.latency.final_vad_endpoint_at = self.final_vad_endpoint_at
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
        generation.latency.commit_readiness = GenerationCommitReadiness(
            speculative_candidate=False,
            hidden_work_ms=0.0,
            qwen_token_count=0,
            synthesis_word_count=0,
            first_tts_pcm_ready=False,
            buffered_pcm_sample_count=0,
        )
        self._transition_generation(generation, CandidateLifecycle.COMMITTED)
        self.model_context.commit_turn(generation.model_context_turn)
        self.active_generation = generation
        self.playback_controller.replace_generation(generation.generation_id)
        self.playback_flow_control.replace_generation(generation.generation_id)
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
        identity = ModelTurnIdentity(
            assistant_generation_id=generation_id,
            user_turn_id=self.next_user_turn_id,
            transcript_revision_id=transcript_revision_id,
        )
        model_context_turn = ModelConversationTurn(
            identity=identity,
            user_message=ModelUserMessage(content=anchored_text),
        )
        model_context_prefix = self.model_context.snapshot()
        generation = ActiveGeneration(
            candidate_id=generation_id,
            generation_id=generation_id,
            transcript_revision_id=transcript_revision_id,
            prediction_evidence_event_id=(
                None if causal_prediction is None else causal_prediction.stamp.event_id
            ),
            stable_transcript_prefix=stable_transcript_prefix,
            anchored_text=anchored_text,
            input_audio_sample_position=input_audio_sample_position,
            monotonic_creation_time=created_at,
            causal_prediction=causal_prediction,
            prediction_confidence=prediction_confidence,
            prompt_messages=prompt_messages,
            model_context_prefix=model_context_prefix,
            model_context_turn=model_context_turn,
            model_messages=(*model_context_prefix, *model_context_turn.messages()),
            user_turn_id=self.next_user_turn_id,
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
        generation.continuation_allowed.set()
        generation.synthesis_budget_available.set()
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
            await self._handle_generation_failure(generation, error)

    async def _handle_generation_failure(
        self,
        generation: ActiveGeneration,
        error: Exception,
    ) -> None:
        if generation.lifecycle is CandidateLifecycle.FAILED:
            return
        self._transition_generation(generation, CandidateLifecycle.FAILED)
        logger.error(
            "voice response generation failed: session=%s generation=%d",
            self.session_id,
            generation.generation_id,
            exc_info=error,
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
                monotonic_time_seconds=generation.latency.candidate_invalidated_at,
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
            self._invalidate_tool_execution(
                generation,
                reason=ToolInvalidationReason.GENERATION_FAILURE,
                candidate_reason=generation.invalidation_reason,
            )
            self.active_generation = None
            if generation.release_gate.released:
                self._mark_generation_interrupted(generation)
                if self.playback_controller.active_generation_id == generation.generation_id:
                    command = self.playback_controller.issue_cancel(
                        generation_id=generation.generation_id,
                        causal_event_id=str(uuid4()),
                        causal_source=CausalSource.USER_COMMAND,
                        stream_epoch=self.speech_understanding.stream_epoch,
                        turn_epoch=self.speech_understanding.turn_epoch,
                        confidence=1.0,
                    )
                    await self._send_event(command)
        failure = _component_error(
            error,
            component=VoiceComponent.SESSION,
            operation=VoiceOperation.SESSION_RUN,
        )
        await self._send_error(
            VoiceComponentError(
                failure.component,
                failure.operation,
                f"Response generation failed: {failure}",
            ),
            generation_id=generation.generation_id,
            retryable=True,
        )

    async def _generate_response(self, generation: ActiveGeneration) -> None:
        synthesis = SpeechSynthesisSequence(self.speech_synthesizer)
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
        synthesis: SpeechSynthesisSequence,
    ) -> None:
        word_stream = CompleteWordStream()
        model_messages = list(generation.model_messages)
        audible_text_start = 0
        preceding_tool: ToolExecutionJournalEntry | None = None
        recovery_only = False
        if (
            self.tool_executor.search_availability is SearchToolAvailability.UNAVAILABLE
            and required_search_reason(tuple(model_messages)) is not None
        ):
            await self._publish_spoken_text(
                generation,
                synthesis,
                word_stream,
                "I can't access live search right now.",
            )
            await self._flush_synthesis_words(generation, synthesis, word_stream)
            await self._finish_synthesis_input(synthesis)
            return
        while True:
            tool_calls_allowed = (
                not recovery_only
                and len(generation.model_context_turn.committed_tool_exchanges)
                < self.policy.maximum_tool_rounds
            )
            result = await self._run_model_invocation(
                generation=generation,
                synthesis=synthesis,
                word_stream=word_stream,
                messages=tuple(model_messages),
                tools=self.tool_executor.specifications if tool_calls_allowed else (),
                tool_calls_allowed=tool_calls_allowed,
                post_tool_continuation=preceding_tool is not None,
            )
            if preceding_tool is not None:
                preceding_tool.continuation_invocation_id = result.invocation_id
            await self._flush_synthesis_words(generation, synthesis, word_stream)
            audible_text_end = len(generation.response_text)
            if result.tool_request is None and result.tool_failure is None:
                if not generation.response_text.strip():
                    raise VoiceComponentError(
                        VoiceComponent.LANGUAGE_MODEL,
                        VoiceOperation.GENERATE_TEXT,
                        "The language model returned an empty response.",
                    )
                if preceding_tool is not None and audible_text_start == audible_text_end:
                    raise VoiceComponentError(
                        VoiceComponent.LANGUAGE_MODEL,
                        VoiceOperation.GENERATE_TEXT,
                        "Qwen returned no spoken continuation after the tool result.",
                    )
                await self._finish_synthesis_input(synthesis)
                return
            if not tool_calls_allowed:
                raise VoiceComponentError(
                    VoiceComponent.LANGUAGE_MODEL,
                    VoiceOperation.GENERATE_TEXT,
                    "Qwen attempted a tool call after the configured round limit.",
                )
            audible_content = generation.response_text[audible_text_start:audible_text_end].strip()
            tool_call: ToolCall | None = None
            assistant_message: ModelAssistantMessage
            if result.tool_failure is not None:
                tool_outcome = self._tool_failure_outcome(result.tool_failure)
                assistant_message = ModelAssistantMessage(content=audible_content)
                await synthesis.finish_utterance()
            else:
                request = result.tool_request
                assert request is not None
                validated_call = self.tool_executor.validate(request)
                if isinstance(validated_call, ToolCallFailure):
                    tool_outcome = self._tool_failure_outcome(validated_call)
                    assistant_message = ModelAssistantMessage(content=audible_content)
                    await synthesis.finish_utterance()
                else:
                    tool_call = validated_call
                    if not audible_content:
                        await self._publish_spoken_text(
                            generation,
                            synthesis,
                            word_stream,
                            _tool_bridge(validated_call.function.name),
                        )
                        await self._flush_synthesis_words(generation, synthesis, word_stream)
                        audible_text_end = len(generation.response_text)
                        audible_content = generation.response_text[
                            audible_text_start:audible_text_end
                        ].strip()
                    await synthesis.finish_utterance()
                    assistant_message = ModelAssistantMessage(
                        content=audible_content,
                        tool_calls=(tool_call,),
                    )
                    if not self._tool_anchors_are_current(generation, result.invocation_id):
                        return
                    self._stage_assistant_tool_call(
                        generation,
                        result.invocation_id,
                        audible_text_start,
                        audible_text_end,
                        assistant_message,
                    )
                    match generation.tool_circuit_breaker.admit(validated_call):
                        case ToolCallAdmission.ALLOWED:
                            tool_outcome = await self._execute_tool_call(
                                generation,
                                result.invocation_id,
                                validated_call,
                            )
                            generation.tool_circuit_breaker.record(validated_call, tool_outcome)
                        case ToolCallAdmission.DUPLICATE:
                            tool_outcome = ToolExecutionFailure(
                                call_id=validated_call.id,
                                tool_name=validated_call.function.name,
                                reason=ToolCallFailureReason.DUPLICATE_CALL,
                                message=(
                                    "An identical tool call already succeeded in this response."
                                ),
                            )
            tool = self._require_tool_execution(generation, result.invocation_id)
            tool.outcome = tool_outcome
            if isinstance(tool_outcome, ToolExecutionFailure):
                tool.lifecycle = ToolLifecycle.FAILED
            if not self._tool_anchors_are_current(generation, result.invocation_id):
                return
            if tool_call is None:
                model_messages.extend(
                    (
                        assistant_message,
                        ModelToolMessage(
                            tool_call_id=tool_outcome.call_id,
                            outcome=tool_outcome,
                        ),
                    )
                )
                recovery_only = True
            else:
                generation.model_context_turn.commit_tool_result(tool_outcome)
                tool.result_commit_status = (
                    ToolResultCommitStatus.GENERATION_LOCAL_COMMITTED
                    if generation.speculative
                    else ToolResultCommitStatus.SESSION_COMMITTED
                )
                committed_at = time.perf_counter()
                tool.result_committed_at = committed_at
                if generation.latency.tool_result_committed_at is None:
                    generation.latency.tool_result_committed_at = committed_at
                model_messages = [
                    *generation.model_context_prefix,
                    *generation.model_context_turn.messages(),
                ]
                generation.model_messages = tuple(model_messages)
                if (
                    isinstance(tool_outcome, ToolSuccess)
                    and tool_outcome.delivery is ToolResultDelivery.DIRECT_SPEECH
                ):
                    audible_text_start = len(generation.response_text)
                    generation.final_answer_text_start = audible_text_start
                    self._record_first_final_answer_text(generation)
                    await self._publish_spoken_text(
                        generation,
                        synthesis,
                        word_stream,
                        " " + tool_outcome.result,
                    )
                    await self._flush_synthesis_words(generation, synthesis, word_stream)
                    await self._finish_synthesis_input(synthesis)
                    return
                if result.temperature_conversion is not None and isinstance(
                    tool_outcome, ToolSuccess
                ):
                    audible_text_start = len(generation.response_text)
                    generation.final_answer_text_start = audible_text_start
                    self._record_first_final_answer_text(generation)
                    await self._publish_spoken_text(
                        generation,
                        synthesis,
                        word_stream,
                        " "
                        + temperature_conversion_spoken_result(
                            result.temperature_conversion,
                            tool_outcome.result,
                        ),
                    )
                    await self._flush_synthesis_words(generation, synthesis, word_stream)
                    await self._finish_synthesis_input(synthesis)
                    return
            generation.model_messages = tuple(model_messages)
            await generation.continuation_allowed.wait()
            if not self._tool_anchors_are_current(generation, result.invocation_id):
                return
            if generation.response_text and not generation.response_text[-1].isspace():
                await self._publish_spoken_text(generation, synthesis, word_stream, " ")
            if isinstance(tool_outcome, ToolExecutionFailure) and tool_call is not None:
                await self._publish_spoken_text(
                    generation,
                    synthesis,
                    word_stream,
                    tool_failure_spoken_response(tool_outcome),
                )
                await self._flush_synthesis_words(generation, synthesis, word_stream)
                await self._finish_synthesis_input(synthesis)
                return
            audible_text_start = len(generation.response_text)
            if generation.final_answer_text_start is None:
                generation.final_answer_text_start = audible_text_start
                generation.latency.second_invocation_started_at = time.perf_counter()
            preceding_tool = tool

    async def _run_model_invocation(
        self,
        generation: ActiveGeneration,
        synthesis: SpeechSynthesisSequence,
        word_stream: CompleteWordStream,
        messages: tuple[ModelMessage, ...],
        tools: tuple[ToolSpecification, ...],
        tool_calls_allowed: bool,
        post_tool_continuation: bool,
    ) -> ModelInvocationResult:
        request = LanguageModelRequest(
            assistant_generation_id=generation.generation_id,
            messages=messages,
            tools=tools,
        )
        await self._send_event(
            LlmModelRequestEvent(
                generation_id=generation.generation_id,
                invocation_index=len(generation.invocation_ids) + 1,
                speculative=generation.speculative,
                messages=(
                    LlmModelSystemMessage(content=LANGUAGE_MODEL_SYSTEM_PROMPT),
                    *(_llm_model_message(message) for message in request.messages),
                ),
                tools=request.tools,
            )
        )
        language_stream = self.language_model.stream_response(request)
        invocation_id: int | None = None
        tool_request: SerializedToolCall | None = None
        tool_failure: ToolCallFailure | None = None
        completed = False
        routed_at_tool_start = False
        temperature_conversion: TemperatureConversion | None = None
        try:
            async with contextlib.aclosing(language_stream):
                async for event in language_stream:
                    if invocation_id is None:
                        invocation_id = event.invocation_id
                        self._record_invocation_id(generation, invocation_id)
                    elif event.invocation_id != invocation_id:
                        raise ValueError("A Qwen stream changed invocation identity.")
                    match event:
                        case LanguageModelTextDelta():
                            self._update_invocation_tokens(generation, event)
                            if post_tool_continuation:
                                self._record_first_final_answer_text(generation)
                            await self._publish_spoken_text(
                                generation,
                                synthesis,
                                word_stream,
                                event.text,
                            )
                        case LanguageModelToolCallStarted():
                            self._update_invocation_tokens(generation, event)
                            if not tool_calls_allowed:
                                tool_failure = ToolCallFailure(
                                    call_id=event.call_id,
                                    reason=ToolCallFailureReason.MULTIPLE_CALLS,
                                    message="The configured tool round limit was reached.",
                                    attempted_tool_name=None,
                                )
                                continue
                            self._record_tool_call_started(generation, event)
                            routed_search = route_required_search_call(
                                messages,
                                tools,
                                invocation_id,
                                call_id=event.call_id,
                            )
                            if routed_search is not None:
                                tool_request = routed_search.request
                                self._record_tool_call_completed(
                                    generation,
                                    LanguageModelToolCall(
                                        invocation_id=invocation_id,
                                        request=tool_request,
                                        cumulative_token_count=event.cumulative_token_count,
                                    ),
                                )
                                routed_at_tool_start = True
                                logger.info(
                                    "required search dispatched at tool-call start: "
                                    "session=%s generation=%d invocation=%d reason=%s",
                                    self.session_id,
                                    generation.generation_id,
                                    invocation_id,
                                    routed_search.reason,
                                )
                                break
                            routed_calculation = route_required_calculation_call(
                                messages,
                                tools,
                                invocation_id,
                                call_id=event.call_id,
                            )
                            if routed_calculation is not None:
                                tool_request = routed_calculation.request
                                temperature_conversion = routed_calculation.conversion
                                self._record_tool_call_completed(
                                    generation,
                                    LanguageModelToolCall(
                                        invocation_id=invocation_id,
                                        request=tool_request,
                                        cumulative_token_count=event.cumulative_token_count,
                                    ),
                                )
                                routed_at_tool_start = True
                                logger.info(
                                    "required calculation dispatched at tool-call start: "
                                    "session=%s generation=%d invocation=%d reason=%s",
                                    self.session_id,
                                    generation.generation_id,
                                    invocation_id,
                                    routed_calculation.reason,
                                )
                                break
                        case LanguageModelToolCall():
                            self._update_invocation_tokens(generation, event)
                            if not tool_calls_allowed:
                                tool_failure = ToolCallFailure(
                                    call_id=event.request.id,
                                    reason=ToolCallFailureReason.MULTIPLE_CALLS,
                                    message="The configured tool round limit was reached.",
                                    attempted_tool_name=event.request.name,
                                )
                                continue
                            tool_request = event.request
                            self._record_tool_call_completed(generation, event)
                        case LanguageModelToolCallFailure():
                            self._update_invocation_tokens(generation, event)
                            tool_failure = event.failure
                            if tool_calls_allowed:
                                self._record_tool_call_failure(generation, event)
                        case LanguageModelCompleted():
                            self._update_invocation_tokens(generation, event)
                            completed = True
                        case LanguageModelFailed():
                            raise VoiceComponentError(
                                VoiceComponent.LANGUAGE_MODEL,
                                VoiceOperation.GENERATE_TEXT,
                                event.message,
                            )
        except VoiceComponentError:
            raise
        except Exception as error:
            raise VoiceComponentError(
                VoiceComponent.LANGUAGE_MODEL,
                VoiceOperation.GENERATE_TEXT,
                str(error),
            ) from error
        if invocation_id is None:
            raise VoiceComponentError(
                VoiceComponent.LANGUAGE_MODEL,
                VoiceOperation.GENERATE_TEXT,
                "Qwen returned no typed invocation events.",
            )
        if not completed and not routed_at_tool_start:
            logger.warning(
                "language model stream ended without an explicit completion event: "
                "session=%s generation=%d invocation=%d",
                self.session_id,
                generation.generation_id,
                invocation_id,
            )
        if tool_request is None and tool_failure is None and tool_calls_allowed:
            routed_search = route_required_search_call(messages, tools, invocation_id)
            if routed_search is not None:
                tool_request = routed_search.request
                routed_event = LanguageModelToolCall(
                    invocation_id=invocation_id,
                    request=tool_request,
                    cumulative_token_count=generation.invocation_token_counts[invocation_id],
                )
                self._record_tool_call_completed(generation, routed_event)
                logger.warning(
                    "required search call routed after Qwen omitted it: "
                    "session=%s generation=%d invocation=%d reason=%s",
                    self.session_id,
                    generation.generation_id,
                    invocation_id,
                    routed_search.reason,
                )
            else:
                routed_calculation = route_required_calculation_call(
                    messages,
                    tools,
                    invocation_id,
                )
                if routed_calculation is not None:
                    tool_request = routed_calculation.request
                    temperature_conversion = routed_calculation.conversion
                    routed_event = LanguageModelToolCall(
                        invocation_id=invocation_id,
                        request=tool_request,
                        cumulative_token_count=generation.invocation_token_counts[invocation_id],
                    )
                    self._record_tool_call_completed(generation, routed_event)
                    logger.warning(
                        "required calculation call routed after Qwen omitted it: "
                        "session=%s generation=%d invocation=%d reason=%s",
                        self.session_id,
                        generation.generation_id,
                        invocation_id,
                        routed_calculation.reason,
                    )
        return ModelInvocationResult(
            invocation_id=invocation_id,
            tool_request=tool_request,
            tool_failure=tool_failure,
            temperature_conversion=temperature_conversion,
        )

    async def _publish_spoken_text(
        self,
        generation: ActiveGeneration,
        synthesis: SpeechSynthesisSequence,
        word_stream: CompleteWordStream,
        text_delta: str,
    ) -> None:
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
        if text_delta:
            await generation.release_gate.publish(
                ReleasedTextDelta(
                    generation_id=generation.generation_id,
                    text=text_delta,
                )
            )
        for word in word_stream.add_text(text_delta):
            if not generation.continuation_allowed.is_set():
                await generation.continuation_allowed.wait()
            self._record_first_qwen_complete_word(generation, word)
            await self._add_synthesis_word(generation, synthesis, word)

    async def _flush_synthesis_words(
        self,
        generation: ActiveGeneration,
        synthesis: SpeechSynthesisSequence,
        word_stream: CompleteWordStream,
    ) -> None:
        for word in word_stream.finish():
            self._record_first_qwen_complete_word(generation, word)
            await self._add_synthesis_word(generation, synthesis, word)

    @staticmethod
    async def _finish_synthesis_input(synthesis: SpeechSynthesisSequence) -> None:
        try:
            await synthesis.finish_input()
        except Exception as error:
            raise VoiceComponentError(
                VoiceComponent.SPEECH_SYNTHESIS,
                VoiceOperation.STREAM_SYNTHESIS,
                str(error),
            ) from error

    async def _execute_tool_call(
        self,
        generation: ActiveGeneration,
        invocation_id: int,
        call: ToolCall,
    ) -> ToolSuccess | ToolExecutionFailure:
        tool = self._require_tool_execution(generation, invocation_id)
        tool.call_id = call.id
        tool.call = call
        tool.lifecycle = ToolLifecycle.RUNNING
        execution_started_at = time.perf_counter()
        tool.execution_started_at = execution_started_at
        if generation.latency.tool_execution_started_at is None:
            generation.latency.tool_execution_started_at = execution_started_at
        logger.info(
            "tool execution started: session=%s generation=%d invocation=%d tool=%s "
            "tool_call_complete_to_execution_ms=%s first_bridge_pcm_to_execution_ms=%s",
            self.session_id,
            generation.generation_id,
            invocation_id,
            call.function.name,
            _optional_milliseconds_between(tool.call_completed_at, execution_started_at),
            _optional_milliseconds_between(
                None
                if generation.latency.first_bridge_pcm is None
                else generation.latency.first_bridge_pcm.monotonic_time_seconds,
                execution_started_at,
            ),
        )
        task = asyncio.create_task(self.tool_executor.execute(call))
        task.add_done_callback(_consume_tool_task_result)
        tool.task = task
        try:
            outcome = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.policy.tool_timeout_seconds,
            )
        except TimeoutError:
            task.cancel()
            self._observe_tool_cancellation(generation, tool, task)
            outcome = ToolExecutionFailure(
                call_id=call.id,
                tool_name=call.function.name,
                reason=ToolExecutionFailureReason.TIMEOUT,
                message="Tool execution timed out.",
            )
        except asyncio.CancelledError:
            raise
        execution_completed_at = time.perf_counter()
        tool.execution_completed_at = execution_completed_at
        if generation.latency.tool_execution_completed_at is None:
            generation.latency.tool_execution_completed_at = execution_completed_at
        tool.outcome = outcome
        search_debug_trace = self.tool_executor.take_search_debug_trace(call.id)
        if search_debug_trace is not None:
            await self._send_event(
                SearchDebugEvent(
                    generation_id=generation.generation_id,
                    query=search_debug_trace.query,
                    results=tuple(
                        SearchDebugResultEvent(
                            title=result.title,
                            url=result.url,
                            snippet=result.snippet,
                        )
                        for result in search_debug_trace.results
                    ),
                    summarizer_system_prompt=search_debug_trace.summarizer_system_prompt,
                    summarizer_user_prompt=search_debug_trace.summarizer_user_prompt,
                    summary=search_debug_trace.summary,
                    provider_duration_ms=search_debug_trace.provider_duration_ms,
                    summarizer_duration_ms=search_debug_trace.summarizer_duration_ms,
                    total_duration_ms=search_debug_trace.total_duration_ms,
                )
            )
        if isinstance(outcome, ToolSuccess):
            tool.lifecycle = ToolLifecycle.SUCCEEDED
            match call.function:
                case CalculateToolCallFunction(arguments=arguments):
                    logger.info(
                        "calculator trace: session=%s generation=%d expression=%r result=%r",
                        self.session_id,
                        generation.generation_id,
                        arguments.expression,
                        outcome.result,
                    )
                case _:
                    pass
            logger.info(
                "tool execution completed: session=%s generation=%d tool=%s "
                "status=success duration_ms=%.1f",
                self.session_id,
                generation.generation_id,
                call.function.name,
                _milliseconds_between(execution_started_at, execution_completed_at),
            )
        else:
            tool.lifecycle = ToolLifecycle.FAILED
            logger.warning(
                "tool execution completed: session=%s generation=%d tool=%s "
                "status=failure reason=%s duration_ms=%.1f",
                self.session_id,
                generation.generation_id,
                call.function.name,
                outcome.reason,
                _milliseconds_between(execution_started_at, execution_completed_at),
            )
        return outcome

    def _tool_failure_outcome(self, failure: ToolCallFailure) -> ToolExecutionFailure:
        return ToolExecutionFailure(
            call_id=failure.call_id,
            tool_name=None,
            reason=failure.reason,
            message=failure.message,
        )

    def _stage_assistant_tool_call(
        self,
        generation: ActiveGeneration,
        invocation_id: int,
        audible_text_start: int,
        audible_text_end: int,
        assistant_message: ModelAssistantMessage,
    ) -> None:
        tool = self._require_tool_execution(generation, invocation_id)
        if tool.result_commit_status is not ToolResultCommitStatus.NOT_STAGED:
            raise ValueError("The tool call has already been staged.")
        call = assistant_message.tool_calls[0]
        if tool.call_id != call.id:
            raise ValueError("The validated tool call does not match the parsed call envelope.")
        generation.model_context_turn.stage_tool_call(
            invocation_id=invocation_id,
            audible_text_start=audible_text_start,
            audible_text_end=audible_text_end,
            message=assistant_message,
        )
        tool.call = call
        tool.result_commit_status = ToolResultCommitStatus.STAGED

    def _record_invocation_id(
        self,
        generation: ActiveGeneration,
        invocation_id: int,
    ) -> None:
        if generation.invocation_ids and invocation_id <= generation.invocation_ids[-1]:
            raise ValueError("Qwen invocation IDs must be globally monotonic.")
        generation.invocation_ids.append(invocation_id)

    @staticmethod
    def _update_invocation_tokens(
        generation: ActiveGeneration,
        event: (
            LanguageModelTextDelta
            | LanguageModelToolCallStarted
            | LanguageModelToolCall
            | LanguageModelToolCallFailure
            | LanguageModelCompleted
        ),
    ) -> None:
        previous_count = generation.invocation_token_counts.get(event.invocation_id, 0)
        if event.cumulative_token_count < previous_count:
            raise ValueError("Qwen cumulative token counts must be monotonic per invocation.")
        generation.invocation_token_counts[event.invocation_id] = event.cumulative_token_count
        generation.qwen_token_count = sum(generation.invocation_token_counts.values())

    def _record_tool_call_started(
        self,
        generation: ActiveGeneration,
        event: LanguageModelToolCallStarted,
    ) -> None:
        tool = self._find_tool_execution(generation, event.invocation_id)
        if tool is None:
            tool = self._create_tool_execution(
                generation,
                event.invocation_id,
                event.call_id,
            )
        elif tool.call_id != event.call_id:
            raise ValueError("A tool invocation changed call identity while buffering.")
        if tool.lifecycle is not ToolLifecycle.NOT_REQUESTED:
            return
        tool.lifecycle = ToolLifecycle.CALL_BUFFERING
        tool.call_id = event.call_id
        call_started_at = time.perf_counter()
        tool.call_started_at = call_started_at
        if generation.latency.tool_call_started_at is None:
            generation.latency.tool_call_started_at = call_started_at
        logger.info(
            "tool call buffering started: session=%s generation=%d invocation=%d "
            "generation_to_tool_call_start_ms=%s",
            self.session_id,
            generation.generation_id,
            event.invocation_id,
            _optional_milliseconds_between(
                generation.latency.generation_started_at,
                call_started_at,
            ),
        )
        first_delta_at = generation.latency.first_language_delta_at
        if first_delta_at is not None:
            generation.latency.first_bridge_text = MediaLatencyPoint(
                monotonic_time_seconds=first_delta_at,
                input_sample_position=generation.input_audio_sample_position,
                output_sample_position=None,
                text_offset=0,
            )

    def _create_tool_execution(
        self,
        generation: ActiveGeneration,
        invocation_id: int,
        call_id: str,
    ) -> ToolExecutionJournalEntry:
        if self._find_tool_execution(generation, invocation_id) is not None:
            raise ValueError("A tool journal entry already exists for the invocation.")
        tool = ToolExecutionJournalEntry(
            identity=generation.model_context_turn.identity,
            round_index=len(generation.tool_executions) + 1,
            invocation_id=invocation_id,
            created_at=time.perf_counter(),
            call_id=call_id,
        )
        generation.tool_executions.append(tool)
        self.tool_execution_journal.append(tool)
        return tool

    @staticmethod
    def _find_tool_execution(
        generation: ActiveGeneration,
        invocation_id: int,
    ) -> ToolExecutionJournalEntry | None:
        return next(
            (
                tool
                for tool in reversed(generation.tool_executions)
                if tool.invocation_id == invocation_id
            ),
            None,
        )

    def _record_tool_call_completed(
        self,
        generation: ActiveGeneration,
        event: LanguageModelToolCall,
    ) -> None:
        tool = self._find_tool_execution(generation, event.invocation_id)
        if tool is None:
            tool = self._create_tool_execution(
                generation,
                event.invocation_id,
                event.request.id,
            )
        elif tool.call_id != event.request.id:
            raise ValueError("A completed tool call does not match its buffered call ID.")
        tool.lifecycle = ToolLifecycle.READY
        tool.call_id = event.request.id
        tool.request = event.request
        call_completed_at = time.perf_counter()
        tool.call_completed_at = call_completed_at
        if generation.latency.tool_call_completed_at is None:
            generation.latency.tool_call_completed_at = call_completed_at
        logger.info(
            "tool call buffering completed: session=%s generation=%d invocation=%d "
            "tool_syntax_buffering_ms=%s generation_to_tool_call_complete_ms=%s",
            self.session_id,
            generation.generation_id,
            event.invocation_id,
            _optional_milliseconds_between(tool.call_started_at, call_completed_at),
            _optional_milliseconds_between(
                generation.latency.generation_started_at,
                call_completed_at,
            ),
        )

    def _record_tool_call_failure(
        self,
        generation: ActiveGeneration,
        event: LanguageModelToolCallFailure,
    ) -> None:
        tool = self._find_tool_execution(generation, event.invocation_id)
        if tool is None:
            tool = self._create_tool_execution(
                generation,
                event.invocation_id,
                event.failure.call_id,
            )
        elif tool.call_id != event.failure.call_id:
            raise ValueError("A failed tool call does not match its buffered call ID.")
        tool.lifecycle = ToolLifecycle.FAILED
        tool.call_id = event.failure.call_id
        call_completed_at = time.perf_counter()
        tool.call_completed_at = call_completed_at
        if generation.latency.tool_call_completed_at is None:
            generation.latency.tool_call_completed_at = call_completed_at

    @staticmethod
    def _record_first_final_answer_text(generation: ActiveGeneration) -> None:
        if generation.latency.first_final_answer_text is not None:
            return
        generation.latency.first_final_answer_text = MediaLatencyPoint(
            monotonic_time_seconds=time.perf_counter(),
            input_sample_position=generation.input_audio_sample_position,
            output_sample_position=None,
            text_offset=len(generation.response_text),
        )

    def _tool_anchors_are_current(
        self,
        generation: ActiveGeneration,
        invocation_id: int,
    ) -> bool:
        tool = self._require_tool_execution(generation, invocation_id)
        return (
            self.active_generation is generation
            and not generation.cancelled
            and tool.identity == generation.model_context_turn.identity
            and tool.identity.assistant_generation_id == generation.generation_id
            and tool.identity.user_turn_id == generation.user_turn_id
            and tool.identity.transcript_revision_id == generation.transcript_revision_id
            and tool.invocation_id == invocation_id
        )

    def _invalidate_tool_execution(
        self,
        generation: ActiveGeneration,
        reason: ToolInvalidationReason,
        candidate_reason: CandidateInvalidationReason | None,
    ) -> None:
        if not generation.tool_executions:
            return
        invalidated = reason in (
            ToolInvalidationReason.SPECULATIVE_INVALIDATION,
            ToolInvalidationReason.GENERATION_FAILURE,
        )
        invalidated_at = time.perf_counter()
        generation.latency.cancelled_or_invalidated_at = invalidated_at
        if any(
            tool.result_commit_status is ToolResultCommitStatus.GENERATION_LOCAL_COMMITTED
            for tool in generation.tool_executions
        ):
            generation.model_context_turn.discard_tool_exchanges()
        elif any(
            tool.result_commit_status is ToolResultCommitStatus.STAGED
            for tool in generation.tool_executions
        ):
            generation.model_context_turn.discard_staged_tool_call()
        for tool in generation.tool_executions:
            tool.invalidated_at = invalidated_at
            tool.invalidation_reason = reason
            tool.candidate_invalidation_reason = candidate_reason
            if tool.result_commit_status is ToolResultCommitStatus.SESSION_COMMITTED:
                continue
            if tool.result_commit_status is ToolResultCommitStatus.GENERATION_LOCAL_COMMITTED:
                tool.result_commit_status = ToolResultCommitStatus.DISCARDED
                tool.lifecycle = (
                    ToolLifecycle.INVALIDATED if invalidated else ToolLifecycle.CANCELLED
                )
                tool.wasted = True
                continue
            if (
                tool.result_commit_status is ToolResultCommitStatus.NOT_STAGED
                and tool.lifecycle is ToolLifecycle.FAILED
                and tool.outcome is not None
            ):
                continue
            if tool.lifecycle in (ToolLifecycle.CANCELLED, ToolLifecycle.INVALIDATED):
                continue
            tool.lifecycle = ToolLifecycle.INVALIDATED if invalidated else ToolLifecycle.CANCELLED
            if tool.result_commit_status is ToolResultCommitStatus.STAGED:
                tool.result_commit_status = ToolResultCommitStatus.DISCARDED
            tool.outcome = ToolExecutionFailure(
                call_id=tool.call_id,
                tool_name=None if tool.call is None else tool.call.function.name,
                reason=(
                    ToolExecutionFailureReason.INVALIDATED
                    if invalidated
                    else ToolExecutionFailureReason.CANCELLED
                ),
                message=f"Tool execution ended because the generation was {reason.value}.",
            )
            tool.wasted = tool.task is not None
            task = tool.task
            if task is not None and not task.done():
                task.cancel()
                self._observe_tool_cancellation(generation, tool, task)

    def _observe_tool_cancellation(
        self,
        generation: ActiveGeneration,
        tool: ToolExecutionJournalEntry,
        task: asyncio.Task[ToolSuccess | ToolExecutionFailure],
    ) -> None:
        if tool.cancellation_observer is not None:
            return

        async def observe() -> None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=self.policy.tool_cancellation_timeout_seconds,
                )
            except (TimeoutError, asyncio.CancelledError):
                tool.wasted = True
            except Exception:
                logger.debug(
                    "cancelled tool task finished with an error: session=%s generation=%d",
                    self.session_id,
                    generation.generation_id,
                    exc_info=True,
                )

        tool.cancellation_observer = asyncio.create_task(observe())

    @staticmethod
    def _require_tool_execution(
        generation: ActiveGeneration,
        invocation_id: int,
    ) -> ToolExecutionJournalEntry:
        tool = VoiceSession._find_tool_execution(generation, invocation_id)
        if tool is None:
            raise AssertionError("The invocation has no tool lifecycle.")
        return tool

    async def _add_synthesis_word(
        self,
        generation: ActiveGeneration,
        synthesis: SpeechSynthesisSequence,
        word: SynthesisWord,
    ) -> None:
        spoken_word = normalize_synthesis_word(word)
        if spoken_word is None:
            return
        generation.synthesis_word_count += 1
        if generation.latency.first_synthesis_word_at is None:
            generation.latency.first_synthesis_word_at = time.perf_counter()
            generation.latency.tts_first_word = MediaLatencyPoint(
                monotonic_time_seconds=generation.latency.first_synthesis_word_at,
                input_sample_position=generation.input_audio_sample_position,
                output_sample_position=None,
                text_offset=spoken_word.text_end,
            )
            logger.info(
                "speech synthesis first word: session=%s generation=%d text=%r",
                self.session_id,
                generation.generation_id,
                spoken_word.text,
            )
        try:
            await synthesis.add_word(spoken_word)
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
                    if generation.latency.synthesis_metrics is None:
                        generation.latency.synthesis_metrics = event
                case SynthesizedWordBoundary():
                    if event.text_offset > len(generation.response_text):
                        raise ValueError("TTS returned a text boundary beyond generated text.")
                    generation.boundary_samples[event.text_offset] = event.start_sample
                    if (
                        generation.final_answer_text_start is not None
                        and event.text_offset > generation.final_answer_text_start
                        and generation.final_answer_first_sample is None
                    ):
                        generation.final_answer_first_sample = event.start_sample
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
                    await self._wait_for_synthesis_overlap_budget(
                        generation,
                        event.start_sample + _pcm_sample_count(event.pcm_bytes),
                    )
                    if event.start_sample != expected_start_sample:
                        raise ValueError("TTS audio chunks must have contiguous sample offsets.")
                    event_end_sample = event.start_sample + _pcm_sample_count(event.pcm_bytes)
                    if (
                        generation.tool_executions
                        and generation.latency.first_bridge_pcm is None
                        and (
                            generation.final_answer_first_sample is None
                            or event.start_sample < generation.final_answer_first_sample
                        )
                    ):
                        generation.latency.first_bridge_pcm = MediaLatencyPoint(
                            monotonic_time_seconds=time.perf_counter(),
                            input_sample_position=generation.input_audio_sample_position,
                            output_sample_position=event.start_sample,
                            text_offset=None,
                        )
                    if (
                        generation.final_answer_first_sample is not None
                        and generation.latency.first_final_answer_pcm is None
                        and event_end_sample > generation.final_answer_first_sample
                    ):
                        first_final_answer_pcm_at = time.perf_counter()
                        generation.latency.first_final_answer_pcm = MediaLatencyPoint(
                            monotonic_time_seconds=first_final_answer_pcm_at,
                            input_sample_position=generation.input_audio_sample_position,
                            output_sample_position=generation.final_answer_first_sample,
                            text_offset=generation.final_answer_text_start,
                        )
                        preceding_tool = generation.tool_executions[-1]
                        logger.info(
                            "post-tool synthesis first audio: session=%s generation=%d "
                            "invocation=%d tool_call_complete_to_pcm_ms=%s "
                            "tool_execution_complete_to_pcm_ms=%s",
                            self.session_id,
                            generation.generation_id,
                            preceding_tool.invocation_id,
                            _optional_milliseconds_between(
                                preceding_tool.call_completed_at,
                                first_final_answer_pcm_at,
                            ),
                            _optional_milliseconds_between(
                                preceding_tool.execution_completed_at,
                                first_final_answer_pcm_at,
                            ),
                        )
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
                    if generation.release_gate.released:
                        self.playback_controller.validate_synthesized_source_position(
                            generation_id=generation.generation_id,
                            synthesized_source_sample_count=generation.tts_sample_count,
                        )
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

    async def _wait_for_synthesis_overlap_budget(
        self,
        generation: ActiveGeneration,
        next_source_sample_position: int,
    ) -> None:
        overlap = self.active_user_overlap
        if overlap is None or overlap.generation_id != generation.generation_id:
            return
        maximum_ahead_samples = (
            self.speech_synthesizer.sample_rate
            * self.playback_controller.config.maximum_synthesized_ahead_ms
            // 1_000
        )
        maximum_source_sample_position = (
            self.playback_condition.latest_source_sample_position + maximum_ahead_samples
        )
        if next_source_sample_position <= maximum_source_sample_position:
            return
        generation.synthesis_budget_available.clear()
        await generation.synthesis_budget_available.wait()

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

    async def _request_generation_cancellation(
        self,
        send_event: bool,
        tool_invalidation_reason: ToolInvalidationReason,
        causal_event_id: str | None = None,
        causal_source: CausalSource = CausalSource.USER_COMMAND,
        confidence: float = 1.0,
    ) -> PlaybackCommandEvent | None:
        generation = self.active_generation
        if generation is None:
            return None
        if generation.speculative:
            await self._invalidate_speculative_candidate(
                CandidateInvalidationReason.SESSION_CANCELLED
            )
            return None
        if generation.lifecycle in (
            CandidateLifecycle.CANCELLATION_REQUESTED,
            CandidateLifecycle.CANCELLED,
        ):
            return None
        generation.cancelled = True
        self._invalidate_tool_execution(
            generation,
            reason=tool_invalidation_reason,
            candidate_reason=None,
        )
        self._transition_generation(generation, CandidateLifecycle.CANCELLATION_REQUESTED)
        self._mark_generation_interrupted(generation)
        command: PlaybackCommandEvent | None = None
        if send_event and self.playback_controller.active_generation_id == generation.generation_id:
            command = self.playback_controller.issue_cancel(
                generation_id=generation.generation_id,
                causal_event_id=causal_event_id or str(uuid4()),
                causal_source=causal_source,
                stream_epoch=self.speech_understanding.stream_epoch,
                turn_epoch=self.speech_understanding.turn_epoch,
                confidence=confidence,
            )
            await self._send_event(command)
        elif self.playback_controller.active_generation_id == generation.generation_id:
            self.playback_controller.estimate_state(
                generation.generation_id,
                PlaybackState.CANCELLED,
            )
        self.playback_flow_control.invalidate_generation(generation.generation_id)
        self.active_generation = None
        self.pending_generation_teardown = generation
        if generation.release_task is not None and not generation.release_task.done():
            generation.release_task.cancel()
        if generation.task is not None and not generation.task.done():
            generation.task.cancel()
        return command

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
        self._invalidate_tool_execution(
            generation,
            reason=ToolInvalidationReason.SPECULATIVE_INVALIDATION,
            candidate_reason=reason,
        )
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
        if generation.release_task is not None and not generation.release_task.done():
            generation.release_task.cancel()
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
        for task in (generation.task, generation.release_task):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if generation.lifecycle is CandidateLifecycle.CANCELLATION_REQUESTED:
            self._transition_generation(generation, CandidateLifecycle.CANCELLED)
        if self.pending_generation_teardown is generation:
            self.pending_generation_teardown = None

    async def _close_generation_tasks(self) -> None:
        tasks = tuple(
            task
            for generation in self.generations.values()
            for task in (generation.task, generation.release_task)
            if task is not None and not task.done()
        )
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _complete_playback(self, event: PlaybackCompleteEvent) -> None:
        generation = self.active_generation
        if generation is None or generation.generation_id != event.generation_id:
            return
        if not self.playback_controller.record_complete(event):
            return
        self.playback_flow_control.observe(event.generation_id, event.source_sample_position)
        generation.playback_complete = True
        generation.accepts_playback = False
        self._commit_assistant_if_complete(generation)

    async def _record_playback_started(self, event: PlaybackStartedEvent) -> None:
        generation = self.generations.get(event.generation_id)
        if generation is None:
            return
        latency = generation.latency
        if latency.playback_started_at is not None:
            return
        if latency.first_audio_sent_at is None:
            self._record_buffered_first_release(generation)
        if latency.first_audio_sent_at is None:
            return
        if not self.playback_controller.record_started(event):
            return
        self.playback_flow_control.observe(event.generation_id, event.source_sample_position)
        latency.playback_started_at = time.perf_counter()
        latency.first_browser_playback_ack = MediaLatencyPoint(
            monotonic_time_seconds=latency.playback_started_at,
            input_sample_position=self.audio_sample_count,
            output_sample_position=event.rendered_output_sample_position,
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
        commit_readiness = _require_commit_readiness(latency)
        speculation_started_at = latency.speculation_started_at
        first_qwen_word_at = _require_latency_point(
            latency.qwen_first_complete_word,
            "first complete language-model word",
        ).monotonic_time_seconds
        first_tts_pcm_at = _require_timestamp(latency.first_audio_at, "first audio")
        if commit_readiness.speculative_candidate:
            assert speculation_started_at is not None
            self.predictive_metrics.record_promoted_candidate_timing(
                speculation_started_at=speculation_started_at,
                first_endpoint_at=latency.first_endpoint_at,
                final_endpoint_at=latency.final_vad_endpoint_at,
                committed_at=turn_committed_at,
                first_qwen_word_at=first_qwen_word_at,
                first_tts_pcm_at=first_tts_pcm_at,
                qwen_word_ready_at_commit=first_qwen_word_at <= turn_committed_at,
                tts_pcm_ready_at_commit=commit_readiness.first_tts_pcm_ready,
            )
        latency_event = AssistantLatencyEvent(
            generation_id=generation.generation_id,
            first_vad_endpoint_to_turn_commit_ms=(
                None
                if latency.first_endpoint_at is None
                else _milliseconds_between(latency.first_endpoint_at, turn_committed_at)
            ),
            final_vad_endpoint_to_turn_commit_ms=(
                None
                if latency.final_vad_endpoint_at is None
                else _milliseconds_between(latency.final_vad_endpoint_at, turn_committed_at)
            ),
            final_vad_endpoint_to_first_audio_send_ms=(
                None
                if latency.final_vad_endpoint_at is None
                else _milliseconds_between(
                    latency.final_vad_endpoint_at,
                    latency.first_audio_sent_at,
                )
            ),
            asr_finalization_ms=latency.asr_finalization_seconds * 1_000,
            candidate_resolution_ms=(
                None
                if latency.candidate_promoted_at is None
                else _milliseconds_between(turn_committed_at, latency.candidate_promoted_at)
            ),
            speculation_start_relative_to_first_vad_endpoint_ms=(
                None
                if speculation_started_at is None or latency.first_endpoint_at is None
                else (speculation_started_at - latency.first_endpoint_at) * 1_000
            ),
            speculation_start_relative_to_final_vad_endpoint_ms=(
                None
                if speculation_started_at is None or latency.final_vad_endpoint_at is None
                else (speculation_started_at - latency.final_vad_endpoint_at) * 1_000
            ),
            speculation_start_to_turn_commit_ms=(
                None
                if speculation_started_at is None
                else _milliseconds_between(speculation_started_at, turn_committed_at)
            ),
            speculation_start_to_first_qwen_word_ms=(
                None
                if speculation_started_at is None
                else _milliseconds_between(speculation_started_at, first_qwen_word_at)
            ),
            speculation_start_to_first_tts_pcm_ms=(
                None
                if speculation_started_at is None
                else _milliseconds_between(speculation_started_at, first_tts_pcm_at)
            ),
            first_qwen_word_ready_at_commit=(
                speculation_started_at is not None and first_qwen_word_at <= turn_committed_at
            ),
            turn_commit_to_playback_ms=_milliseconds_between(
                turn_committed_at,
                latency.playback_started_at,
            ),
            turn_commit_to_first_audio_send_ms=_milliseconds_between(
                turn_committed_at,
                latency.first_audio_sent_at,
            ),
            generation_to_first_word_ms=_milliseconds_between(
                _require_timestamp(latency.generation_started_at, "generation start"),
                first_qwen_word_at,
            ),
            tts_first_word_to_first_pcm_ms=_milliseconds_between(
                _require_timestamp(latency.first_synthesis_word_at, "first synthesis word"),
                first_tts_pcm_at,
            ),
            first_audio_send_to_playback_ms=_milliseconds_between(
                latency.first_audio_sent_at,
                latency.playback_started_at,
            ),
            speculative_candidate_promoted=commit_readiness.speculative_candidate,
            speculative_hidden_work_ms=commit_readiness.hidden_work_ms,
            prepared_qwen_token_count=commit_readiness.qwen_token_count,
            prepared_word_count=commit_readiness.synthesis_word_count,
            first_tts_pcm_ready_at_commit=commit_readiness.first_tts_pcm_ready,
            buffered_audio_ms=(
                commit_readiness.buffered_pcm_sample_count
                * 1_000
                / self.speech_synthesizer.sample_rate
            ),
        )
        await self._send_event(latency_event)
        if commit_readiness.speculative_candidate:
            logger.info(
                "speculative candidate timing: session=%s generation=%d event=%r",
                self.session_id,
                generation.generation_id,
                latency_event,
            )

    @staticmethod
    def _record_generation_commit_readiness(
        generation: ActiveGeneration,
        committed_at: float,
    ) -> None:
        generation_started_at = generation.latency.generation_started_at
        generation.latency.commit_readiness = GenerationCommitReadiness(
            speculative_candidate=True,
            hidden_work_ms=(
                0.0
                if generation_started_at is None
                else _milliseconds_between(generation_started_at, committed_at)
            ),
            qwen_token_count=generation.qwen_token_count,
            synthesis_word_count=generation.synthesis_word_count,
            first_tts_pcm_ready=(
                generation.latency.first_audio_at is not None
                and generation.latency.first_audio_at <= committed_at
            ),
            buffered_pcm_sample_count=generation.release_gate.buffered_pcm_sample_count,
        )

    def _acknowledge_playback(self, event: PlaybackProgressEvent) -> None:
        generation = self.generations.get(event.generation_id)
        if generation is None or not generation.accepts_playback:
            return
        boundary_sample = generation.boundary_samples.get(event.text_offset)
        if boundary_sample != event.boundary_start_sample:
            return
        if not self._text_boundary_is_fully_played(
            generation,
            event.text_offset,
            event.played_sample_count,
        ):
            return
        if not self.playback_controller.record_progress(event):
            return
        self.playback_flow_control.observe(event.generation_id, event.played_sample_count)
        if event.text_offset <= generation.acknowledged_offset:
            return
        self._update_assistant_history(generation, event.text_offset)

    def _record_playback_clock(self, event: PlaybackClockEvent) -> None:
        generation = self.generations.get(event.generation_id)
        if generation is None or not generation.accepts_playback:
            return
        if not self.playback_controller.record_clock(event):
            return
        self.playback_flow_control.observe(event.generation_id, event.source_sample_position)

    def _stop_playback(self, event: PlaybackStoppedEvent) -> None:
        generation = self.generations.get(event.generation_id)
        if generation is None or not generation.cancelled or not generation.accepts_playback:
            return
        if event.text_offset > 0:
            boundary_sample = generation.boundary_samples.get(event.text_offset)
            if boundary_sample is None or not self._text_boundary_is_fully_played(
                generation,
                event.text_offset,
                event.played_sample_count,
            ):
                return
            if event.text_offset > generation.acknowledged_offset:
                self._update_assistant_history(generation, event.text_offset)
        generation.accepts_playback = False
        generation.playback_stopped.set()
        self.playback_controller.estimate_state(
            generation.generation_id,
            PlaybackState.CANCELLED,
        )
        self._mark_generation_interrupted(generation)

    async def _acknowledge_playback_command(
        self,
        event: PlaybackCommandAcknowledgementEvent,
    ) -> None:
        received_monotonic_time_ns = time.perf_counter_ns()
        disposition = self.playback_controller.acknowledge(
            event,
            received_monotonic_time_ns=received_monotonic_time_ns,
        )
        if disposition is not PlaybackAcknowledgementDisposition.APPLIED:
            return
        self.playback_flow_control.observe(event.generation_id, event.source_sample_position)
        interaction_action_event = self._record_overlap_acknowledgement(
            event,
            received_monotonic_time_ns,
        )
        if interaction_action_event is not None:
            await self._send_event(interaction_action_event)
        generation = self.generations.get(event.generation_id)
        if generation is None:
            return
        if event.resume_rejected and self.active_generation is generation:
            await self._request_generation_cancellation(
                send_event=False,
                causal_event_id=event.command_id,
                causal_source=CausalSource.PLAYBACK_ENGINE,
                confidence=1.0,
                tool_invalidation_reason=ToolInvalidationReason.PLAYBACK_RESUME_REJECTED,
            )
        if event.resulting_state is PlaybackState.CANCELLED:
            if generation.cancelled and generation.accepts_playback:
                text_offset = self._played_text_offset(
                    generation,
                    event.source_sample_position,
                )
                if text_offset > generation.acknowledged_offset:
                    self._update_assistant_history(generation, text_offset)
                generation.accepts_playback = False
                generation.playback_stopped.set()
                self._mark_generation_interrupted(generation)

    def _record_overlap_acknowledgement(
        self,
        event: PlaybackCommandAcknowledgementEvent,
        received_monotonic_time_ns: int,
    ) -> InteractionActionDebugEvent | None:
        overlap = next(
            (
                trace
                for trace in reversed(self.user_overlap_traces)
                if event.command_id
                in (
                    trace.duck_command_id,
                    trace.pause_command_id,
                    trace.resume_command_id,
                    trace.cancel_command_id,
                )
            ),
            None,
        )
        if overlap is None:
            return None
        acknowledgement_recorded = False
        if (
            event.command_id == overlap.duck_command_id
            and event.gain_ramp_complete
            and event.resulting_state is not PlaybackState.CANCELLED
        ):
            self.overlap_metrics.record_duck(
                overlap.onset_monotonic_time_ns,
                received_monotonic_time_ns,
            )
            acknowledgement_recorded = True
        elif (
            event.command_id == overlap.pause_command_id
            and event.resulting_state is PlaybackState.PAUSED_BUFFERED
        ):
            overlap.pause_acknowledged_monotonic_time_ns = received_monotonic_time_ns
            self.overlap_metrics.record_pause(
                overlap.onset_monotonic_time_ns,
                received_monotonic_time_ns,
            )
            acknowledgement_recorded = True
        elif (
            event.command_id == overlap.resume_command_id
            and not event.resume_rejected
            and event.resulting_state in (PlaybackState.RESUMING, PlaybackState.SPEAKING)
        ):
            self.overlap_metrics.record_resume(
                overlap.onset_monotonic_time_ns,
                received_monotonic_time_ns,
                overlap.pause_acknowledged_monotonic_time_ns,
            )
            acknowledgement_recorded = True
        elif event.command_id == overlap.cancel_command_id:
            decision = overlap.decision
            self.overlap_metrics.record_cancel(
                overlap.onset_monotonic_time_ns,
                received_monotonic_time_ns,
                fast_path=False if decision is None else decision.fast_path,
            )
            acknowledgement_recorded = True
        if acknowledgement_recorded:
            return InteractionActionDebugEvent(
                action=event.action,
                onset_to_acknowledgement_ms=(
                    received_monotonic_time_ns - overlap.onset_monotonic_time_ns
                )
                / 1_000_000,
            )
        return None

    @staticmethod
    def _text_boundary_is_fully_played(
        generation: ActiveGeneration,
        text_offset: int,
        source_sample_position: int,
    ) -> bool:
        boundary_sample = generation.boundary_samples[text_offset]
        next_boundary_sample = min(
            (
                candidate_sample
                for candidate_sample in generation.boundary_samples.values()
                if candidate_sample > boundary_sample
            ),
            default=None,
        )
        return next_boundary_sample is not None and source_sample_position >= next_boundary_sample

    @staticmethod
    def _played_text_offset(
        generation: ActiveGeneration,
        source_sample_position: int,
    ) -> int:
        ordered_boundaries = sorted(
            generation.boundary_samples.items(),
            key=lambda boundary: boundary[1],
        )
        return max(
            (
                text_offset
                for (text_offset, _), (_, next_boundary_sample) in zip(
                    ordered_boundaries,
                    ordered_boundaries[1:],
                    strict=False,
                )
                if source_sample_position >= next_boundary_sample
            ),
            default=0,
        )

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
        generation.model_context_turn.update_audible_assistant_content(content)

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
        generation.model_context_turn.update_audible_assistant_content(f"{message.content}...")

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
        generation.model_context_turn.update_audible_assistant_content(complete_message.content)
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
        turn_committed_at = _require_timestamp(latency.turn_committed_at, "turn commitment")
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
            "first_vad_endpoint_to_commit_ms=%s final_vad_endpoint_to_commit_ms=%s "
            "final_vad_endpoint_to_released_audio_ms=%s asr_finalization_ms=%.1f "
            "candidate_resolution_ms=%s turn_commit_ms=%.1f llm_first_delta_ms=%.1f "
            "first_synthesis_word_ms=%.1f first_word_to_audio_ms=%.1f "
            "generation_to_audio_ms=%.1f tts_backend=%s "
            "tts_worker_first_word_to_audio_ms=%s "
            "tts_tokenization_ms=%s tts_lm_step_ms=%s tts_mimi_decode_ms=%s "
            "tts_model_steps=%s tts_first_audio_model_step=%s "
            "tts_prompt_preparation_ms=%s tts_first_frame_generation_ms=%s",
            self.session_id,
            generation.generation_id,
            _optional_milliseconds_between(latency.first_endpoint_at, turn_committed_at),
            _optional_milliseconds_between(latency.final_vad_endpoint_at, turn_committed_at),
            _optional_milliseconds_between(
                latency.final_vad_endpoint_at,
                latency.first_audio_sent_at,
            ),
            latency.asr_finalization_seconds * 1_000,
            _optional_milliseconds_between(turn_committed_at, latency.candidate_promoted_at),
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


def _llm_model_message(message: ModelMessage) -> LlmModelMessage:
    match message:
        case ModelUserMessage(content=content):
            return LlmModelUserMessage(content=content)
        case ModelAssistantMessage(content=content, tool_calls=tool_calls):
            return LlmModelAssistantMessage(content=content, tool_calls=tool_calls)
        case ModelToolMessage(tool_call_id=tool_call_id, outcome=outcome):
            return LlmModelToolMessage(tool_call_id=tool_call_id, outcome=outcome)


def _pcm_sample_count(pcm_bytes: bytes) -> int:
    return len(pcm_bytes) // PCM_BYTES_PER_SAMPLE


def _consume_tool_task_result(
    task: asyncio.Task[ToolSuccess | ToolExecutionFailure],
) -> None:
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()


def _milliseconds_to_samples(duration_ms: int) -> int:
    return duration_ms * INPUT_SAMPLE_RATE // 1_000


def _tool_bridge(tool_name: ToolName) -> str:
    match tool_name:
        case ToolName.SEARCH:
            return "Let me look that up."
        case ToolName.CALCULATE:
            return "Let me calculate that."
        case ToolName.GET_TIME:
            return "Let me check the time."


def _milliseconds_between(started_at: float, finished_at: float) -> float:
    if finished_at < started_at:
        raise AssertionError("Latency timestamps must increase monotonically.")
    return (finished_at - started_at) * 1_000


def _optional_milliseconds_between(
    started_at: float | None,
    finished_at: float | None,
) -> str:
    if started_at is None or finished_at is None:
        return "unknown"
    return f"{_milliseconds_between(started_at, finished_at):.1f}"


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


def _require_latency_point(
    point: MediaLatencyPoint | None,
    label: str,
) -> MediaLatencyPoint:
    if point is None:
        raise AssertionError(f"Missing {label} latency point.")
    return point


def _require_commit_readiness(latency: GenerationLatency) -> GenerationCommitReadiness:
    if latency.commit_readiness is None:
        raise AssertionError("Missing generation readiness at turn commitment.")
    return latency.commit_readiness


def _environment_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    value = environment.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric.") from error


def _environment_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    value = environment.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


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
