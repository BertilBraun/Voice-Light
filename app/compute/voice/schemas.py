from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from app.compute.voice.conversation import ConversationRole
from app.compute.voice.errors import VoiceComponent, VoiceOperation
from app.shared.base_model import FrozenBaseModel


class CausalSource(StrEnum):
    ENERGY_VAD = "energy_vad"
    SILERO_VAD = "silero_vad"
    NEMOTRON_ASR = "nemotron_asr"
    TURN_ADAPTER = "turn_adapter"
    LEXICAL_CLASSIFIER = "lexical_classifier"
    RESPONSE_RANKER = "response_ranker"
    FLOOR_POLICY = "floor_policy"
    PLAYBACK_ENGINE = "playback_engine"
    USER_COMMAND = "user_command"


class TraceStamp(FrozenBaseModel):
    event_id: str
    parent_event_ids: tuple[str, ...]
    stream_epoch: int = Field(ge=1)
    turn_epoch: int = Field(ge=1)
    inference_step: int = Field(ge=0)
    observation_id: str
    observation_monotonic_time_ns: int = Field(ge=0)
    emission_monotonic_time_ns: int = Field(ge=0)
    encoder_frame_start: int | None = Field(ge=0)
    encoder_frame_end: int | None = Field(ge=0)
    input_start_sample: int = Field(ge=0)
    input_end_sample: int = Field(ge=0)
    observed_through_input_sample: int = Field(ge=0)
    input_sample_position: int = Field(ge=0)
    output_sample_position: int | None = Field(default=None, ge=0)
    conditioned_transcript_revision_id: int | None = Field(default=None, ge=0)
    conditioned_playback_event_id: str | None = None
    source: CausalSource
    model_name: str | None
    model_revision: str | None

    @model_validator(mode="after")
    def validate_sample_range(self) -> TraceStamp:
        if self.input_end_sample < self.input_start_sample:
            raise ValueError("Trace input end sample cannot precede its start sample.")
        if (self.encoder_frame_start is None) != (self.encoder_frame_end is None):
            raise ValueError("Trace encoder frame bounds must both be present or both be absent.")
        if (
            self.encoder_frame_start is not None
            and self.encoder_frame_end is not None
            and self.encoder_frame_end < self.encoder_frame_start
        ):
            raise ValueError("Trace encoder frame end cannot precede its start.")
        if self.observed_through_input_sample < self.input_end_sample:
            raise ValueError("Observed-through sample cannot precede the trace input range.")
        if self.input_sample_position != self.input_end_sample:
            raise ValueError("Input sample position must equal the trace input end sample.")
        return self


class ActivityHorizon(FrozenBaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    probability: float = Field(ge=0.0, le=1.0)


class PlaybackState(StrEnum):
    IDLE = "idle"
    QUEUED = "queued"
    SPEAKING = "speaking"
    DUCKING = "ducking"
    PAUSED_BUFFERED = "paused_buffered"
    RESUMING = "resuming"
    DRAINING_TO_BOUNDARY = "draining_to_boundary"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class PlaybackConditionAuthority(StrEnum):
    SERVER_ESTIMATED = "server_estimated"
    BROWSER_AUTHORITATIVE = "browser_authoritative"


class PlaybackCondition(FrozenBaseModel):
    event_id: str
    generation_id: int | None = Field(default=None, gt=0)
    state: PlaybackState
    assistant_audible: bool
    latest_output_sample_position: int = Field(ge=0)
    latest_source_sample_position: int = Field(ge=0)
    output_sample_rate: int | None = Field(default=None, gt=0)
    monotonic_time_ns: int = Field(ge=0)
    authority: PlaybackConditionAuthority


class SileroEvidence(FrozenBaseModel):
    is_speech: bool
    monotonic_time_ns: int = Field(ge=0)


class CapturedAudioChunk(FrozenBaseModel):
    pcm16: bytes
    sequence_number: int = Field(ge=0)
    start_input_sample: int = Field(ge=0)
    end_input_sample: int = Field(ge=0)
    monotonic_observation_time_ns: int = Field(ge=0)
    stream_epoch: int = Field(ge=1)
    turn_epoch: int = Field(ge=1)
    silero_evidence: SileroEvidence
    playback_condition: PlaybackCondition

    @model_validator(mode="after")
    def validate_audio_range(self) -> CapturedAudioChunk:
        if len(self.pcm16) % 2 != 0:
            raise ValueError("Captured PCM16 must contain complete samples.")
        if self.end_input_sample < self.start_input_sample:
            raise ValueError("Captured audio end sample cannot precede its start sample.")
        if self.end_input_sample - self.start_input_sample != len(self.pcm16) // 2:
            raise ValueError("Captured audio sample range must exactly match its PCM16 payload.")
        return self


class InteractionPrediction(FrozenBaseModel):
    type: Literal["interaction.prediction"] = "interaction.prediction"
    stamp: TraceStamp
    p_user_speech: float = Field(ge=0.0, le=1.0)
    p_user_yield: float = Field(ge=0.0, le=1.0)
    p_user_backchannel: float = Field(ge=0.0, le=1.0)
    p_user_interruption: float = Field(ge=0.0, le=1.0)
    future_user_activity_horizons: tuple[ActivityHorizon, ...]
    assistant_playback_state: PlaybackState = Field(
        deprecated=(
            "PlaybackCondition is the causal input. This compatibility snapshot field will be "
            "removed after VoiceSession consumes evidence directly."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptRevision(FrozenBaseModel):
    type: Literal["transcript.revision"] = "transcript.revision"
    stamp: TraceStamp
    revision_id: int = Field(ge=0)
    supersedes_revision_id: int | None = Field(default=None, ge=0)
    stable_prefix: str
    volatile_suffix: str
    audio_sample_position: int = Field(ge=0)
    stable_prefix_end_sample: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class TurnEventKind(StrEnum):
    TURN_COMPLETION = "turn_completion"
    CONTINUATION_PAUSE = "continuation_pause"
    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"
    OTHER = "other"


class TurnEventProbability(FrozenBaseModel):
    event: TurnEventKind
    probability: float = Field(ge=0.0, le=1.0)


class OverlapDisposition(StrEnum):
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    FLOOR_TAKING = "floor_taking"


class OverlapDispositionProbability(FrozenBaseModel):
    disposition: OverlapDisposition
    probability: float = Field(ge=0.0, le=1.0)


class YieldEvidence(FrozenBaseModel):
    type: Literal["speech_understanding.yield_evidence"] = "speech_understanding.yield_evidence"
    stamp: TraceStamp
    evidence_group_id: str
    p_user_yield: float = Field(ge=0.0, le=1.0)
    p_user_speech: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class FutureActivityEvidence(FrozenBaseModel):
    type: Literal["speech_understanding.future_activity_evidence"] = (
        "speech_understanding.future_activity_evidence"
    )
    stamp: TraceStamp
    evidence_group_id: str
    horizons: tuple[ActivityHorizon, ...]
    confidence: float = Field(ge=0.0, le=1.0)


class TurnEventEvidence(FrozenBaseModel):
    type: Literal["speech_understanding.turn_event_evidence"] = (
        "speech_understanding.turn_event_evidence"
    )
    stamp: TraceStamp
    evidence_group_id: str
    probabilities: tuple[TurnEventProbability, ...]
    confidence: float = Field(ge=0.0, le=1.0)


class OverlapDispositionEvidence(FrozenBaseModel):
    type: Literal["speech_understanding.overlap_disposition_evidence"] = (
        "speech_understanding.overlap_disposition_evidence"
    )
    stamp: TraceStamp
    evidence_group_id: str
    probabilities: tuple[OverlapDispositionProbability, ...]
    p_user_interruption: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class SpeechUnderstandingStatus(StrEnum):
    ACTIVE = "active"
    TURN_FINALIZED = "turn_finalized"
    CLOSED = "closed"


class SpeechUnderstandingComponent(StrEnum):
    ASR = "asr"
    STANDALONE_TURN_DETECTOR = "standalone_turn_detector"
    INTEGRATED_NEMOTRON = "integrated_nemotron"


class SpeechUnderstandingStatusEvent(FrozenBaseModel):
    type: Literal["speech_understanding.status"] = "speech_understanding.status"
    stamp: TraceStamp
    status: SpeechUnderstandingStatus


class SpeechUnderstandingDegradedEvent(FrozenBaseModel):
    type: Literal["speech_understanding.degraded"] = "speech_understanding.degraded"
    stamp: TraceStamp
    component: SpeechUnderstandingComponent
    reason: str
    dropped_observation_count: int = Field(ge=0)


class SpeechUnderstandingAbstainedEvent(FrozenBaseModel):
    type: Literal["speech_understanding.abstained"] = "speech_understanding.abstained"
    stamp: TraceStamp
    component: SpeechUnderstandingComponent
    reason: str


SpeechUnderstandingEvent = Annotated[
    TranscriptRevision
    | YieldEvidence
    | FutureActivityEvidence
    | TurnEventEvidence
    | OverlapDispositionEvidence
    | SpeechUnderstandingStatusEvent
    | SpeechUnderstandingDegradedEvent
    | SpeechUnderstandingAbstainedEvent,
    Field(discriminator="type"),
]
speech_understanding_event_adapter: TypeAdapter[SpeechUnderstandingEvent] = TypeAdapter(
    SpeechUnderstandingEvent
)


class VoiceClientEventType(StrEnum):
    SESSION_START = "session.start"
    SESSION_STOP = "session.stop"
    PLAYBACK_STARTED = "playback.started"
    PLAYBACK_COMPLETE = "playback.complete"
    PLAYBACK_PROGRESS = "playback.progress"
    PLAYBACK_STOPPED = "playback.stopped"
    PLAYBACK_ACKNOWLEDGEMENT = "playback.acknowledgement"


class SessionStartEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.SESSION_START] = VoiceClientEventType.SESSION_START
    input_sample_rate: int = Field(gt=0)


class SessionStopEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.SESSION_STOP] = VoiceClientEventType.SESSION_STOP


class PlaybackCompleteEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_COMPLETE] = VoiceClientEventType.PLAYBACK_COMPLETE
    generation_id: int = Field(gt=0)
    browser_monotonic_time_ns: int = Field(ge=0)
    rendered_output_sample_position: int = Field(ge=0)
    source_sample_position: int = Field(ge=0)
    output_sample_rate: int = Field(gt=0)


class PlaybackStartedEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_STARTED] = VoiceClientEventType.PLAYBACK_STARTED
    generation_id: int = Field(gt=0)
    browser_monotonic_time_ns: int = Field(ge=0)
    rendered_output_sample_position: int = Field(ge=0)
    source_sample_position: int = Field(ge=0)
    output_sample_rate: int = Field(gt=0)


class PlaybackProgressEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_PROGRESS] = VoiceClientEventType.PLAYBACK_PROGRESS
    generation_id: int = Field(gt=0)
    text_offset: int = Field(gt=0)
    boundary_start_sample: int = Field(ge=0)
    played_sample_count: int = Field(gt=0)
    browser_monotonic_time_ns: int = Field(ge=0)
    rendered_output_sample_position: int = Field(ge=0)
    output_sample_rate: int = Field(gt=0)


class PlaybackStoppedEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_STOPPED] = VoiceClientEventType.PLAYBACK_STOPPED
    generation_id: int = Field(gt=0)
    text_offset: int = Field(ge=0)
    played_sample_count: int = Field(ge=0)
    browser_monotonic_time_ns: int = Field(ge=0)
    rendered_output_sample_position: int = Field(ge=0)
    output_sample_rate: int = Field(gt=0)


class PlaybackCommandAction(StrEnum):
    DUCK = "duck"
    PAUSE_AT_BOUNDARY = "pause_at_boundary"
    RESUME = "resume"
    CANCEL = "cancel"


class PlaybackPauseResult(StrEnum):
    NOT_REQUESTED = "not_requested"
    WORD_BOUNDARY = "word_boundary"
    FORCED_SAMPLE = "forced_sample"


class PlaybackCommandAcknowledgementEvent(FrozenBaseModel):
    type: Literal[VoiceClientEventType.PLAYBACK_ACKNOWLEDGEMENT] = (
        VoiceClientEventType.PLAYBACK_ACKNOWLEDGEMENT
    )
    command_id: str
    generation_id: int = Field(gt=0)
    action: PlaybackCommandAction
    stream_epoch: int = Field(ge=1)
    turn_epoch: int = Field(ge=1)
    resulting_state: PlaybackState
    browser_monotonic_time_ns: int = Field(ge=0)
    rendered_output_sample_position: int = Field(ge=0)
    source_sample_position: int = Field(ge=0)
    output_sample_rate: int = Field(gt=0)
    pause_result: PlaybackPauseResult
    current_gain: float = Field(ge=0.0, le=1.0)
    gain_ramp_complete: bool
    queued_source_sample_count: int = Field(ge=0)
    discarded_source_sample_count: int = Field(ge=0)
    replayed_source_sample_count: int = Field(ge=0)
    skipped_source_sample_count: int = Field(ge=0)
    resume_rejected: bool


VoiceClientEvent = Annotated[
    SessionStartEvent
    | SessionStopEvent
    | PlaybackStartedEvent
    | PlaybackCompleteEvent
    | PlaybackProgressEvent
    | PlaybackStoppedEvent
    | PlaybackCommandAcknowledgementEvent,
    Field(discriminator="type"),
]
voice_client_event_adapter: TypeAdapter[VoiceClientEvent] = TypeAdapter(VoiceClientEvent)


class VoiceServerEventType(StrEnum):
    SESSION_READY = "session.ready"
    VAD_STARTED = "vad.started"
    VAD_STOPPED = "vad.stopped"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    TURN_COMMITTED = "turn.committed"
    LLM_HISTORY = "llm.history"
    ASSISTANT_TEXT_DELTA = "assistant.text.delta"
    ASSISTANT_AUDIO_START = "assistant.audio.start"
    ASSISTANT_AUDIO_END = "assistant.audio.end"
    ASSISTANT_AUDIO_TEXT_BOUNDARY = "assistant.audio.text_boundary"
    ASSISTANT_CANCEL = "assistant.cancel"
    PLAYBACK_COMMAND = "playback.command"
    ERROR = "error"


class SessionReadyEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.SESSION_READY] = VoiceServerEventType.SESSION_READY
    session_id: str
    input_sample_rate: int = Field(gt=0)
    output_sample_rate: int = Field(gt=0)


class SpeechStateEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.VAD_STARTED, VoiceServerEventType.VAD_STOPPED]
    audio_time_ms: int = Field(ge=0)


class TranscriptEvent(FrozenBaseModel):
    type: Literal[
        VoiceServerEventType.TRANSCRIPT_PARTIAL,
        VoiceServerEventType.TRANSCRIPT_FINAL,
        VoiceServerEventType.TURN_COMMITTED,
    ]
    text: str


class LlmHistoryMessage(FrozenBaseModel):
    role: ConversationRole
    content: str


class LlmHistoryEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.LLM_HISTORY] = VoiceServerEventType.LLM_HISTORY
    generation_id: int = Field(gt=0)
    messages: tuple[LlmHistoryMessage, ...]


class AssistantTextDeltaEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ASSISTANT_TEXT_DELTA] = (
        VoiceServerEventType.ASSISTANT_TEXT_DELTA
    )
    generation_id: int = Field(gt=0)
    text: str


class AssistantAudioBoundaryEvent(FrozenBaseModel):
    type: Literal[
        VoiceServerEventType.ASSISTANT_AUDIO_START,
        VoiceServerEventType.ASSISTANT_AUDIO_END,
        VoiceServerEventType.ASSISTANT_CANCEL,
    ]
    generation_id: int = Field(gt=0)


class AssistantAudioTextBoundaryEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ASSISTANT_AUDIO_TEXT_BOUNDARY] = (
        VoiceServerEventType.ASSISTANT_AUDIO_TEXT_BOUNDARY
    )
    generation_id: int = Field(gt=0)
    text_offset: int = Field(gt=0)
    start_sample: int = Field(ge=0)


class PlaybackCommandEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.PLAYBACK_COMMAND] = VoiceServerEventType.PLAYBACK_COMMAND
    command_id: str
    generation_id: int = Field(gt=0)
    action: PlaybackCommandAction
    issued_monotonic_time_ns: int = Field(ge=0)
    causal_event_id: str
    causal_source: CausalSource
    stream_epoch: int = Field(ge=1)
    turn_epoch: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    requested_boundary_source_sample_position: int | None = Field(default=None, ge=0)
    rendered_output_sample_deadline: int | None = Field(default=None, ge=0)
    target_gain: float | None = Field(default=None, ge=0.0, le=1.0)
    gain_ramp_duration_ms: int | None = Field(default=None, gt=0)
    maximum_paused_age_ms: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_action_parameters(self) -> PlaybackCommandEvent:
        match self.action:
            case PlaybackCommandAction.DUCK:
                if self.target_gain is None or self.gain_ramp_duration_ms is None:
                    raise ValueError("Duck commands require a target gain and ramp duration.")
            case PlaybackCommandAction.PAUSE_AT_BOUNDARY:
                if self.rendered_output_sample_deadline is None:
                    raise ValueError("Pause commands require a rendered-output sample deadline.")
            case PlaybackCommandAction.RESUME:
                if (
                    self.target_gain is None
                    or self.gain_ramp_duration_ms is None
                    or self.maximum_paused_age_ms is None
                ):
                    raise ValueError(
                        "Resume commands require a target gain, ramp duration, and maximum age."
                    )
            case PlaybackCommandAction.CANCEL:
                pass
        return self


class ErrorEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ERROR] = VoiceServerEventType.ERROR
    component: VoiceComponent
    operation: VoiceOperation
    generation_id: int | None = Field(default=None, gt=0)
    retryable: bool
    message: str


VoiceServerEvent = Annotated[
    SessionReadyEvent
    | SpeechStateEvent
    | TranscriptEvent
    | LlmHistoryEvent
    | AssistantTextDeltaEvent
    | AssistantAudioBoundaryEvent
    | AssistantAudioTextBoundaryEvent
    | PlaybackCommandEvent
    | ErrorEvent,
    Field(discriminator="type"),
]
