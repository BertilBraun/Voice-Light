from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.shared.base_model import FrozenBaseModel
from app.shared.quality import AudioMetadata, QualityResult


class HealthStatus(StrEnum):
    LIVE = "live"
    READY = "ready"
    NOT_READY = "not_ready"


class ModelStageStatus(StrEnum):
    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ModelStage(FrozenBaseModel):
    name: str
    status: ModelStageStatus
    load_time_seconds: float | None = None
    error: str | None = None


class GpuMemory(FrozenBaseModel):
    current_allocated_mb: float | None
    peak_allocated_mb: float | None


class LivenessResponse(FrozenBaseModel):
    status: Literal[HealthStatus.LIVE] = HealthStatus.LIVE


class ReadinessResponse(FrozenBaseModel):
    status: Literal[HealthStatus.READY, HealthStatus.NOT_READY]
    stages: tuple[ModelStage, ...]
    gpu_memory: GpuMemory


class QualityAnalysisUpload(FrozenBaseModel):
    sample_id: str
    speaker1_original_metadata: AudioMetadata | None
    speaker2_original_metadata: AudioMetadata | None


class QualityAnalysisResponse(FrozenBaseModel):
    speaker1_metadata: AudioMetadata
    speaker2_metadata: AudioMetadata
    quality_result: QualityResult


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(FrozenBaseModel):
    role: ConversationRole
    content: str = Field(min_length=1)


class VoiceClientEventType(StrEnum):
    ASR_START = "asr.start"
    ASR_AUDIO = "asr.audio"
    ASR_FINISH = "asr.finish"
    ASR_CLOSE = "asr.close"
    LLM_GENERATE = "llm.generate"
    TTS_SYNTHESIZE = "tts.synthesize"
    OPERATION_CANCEL = "operation.cancel"


class AsrStartCommand(FrozenBaseModel):
    type: Literal[VoiceClientEventType.ASR_START] = VoiceClientEventType.ASR_START
    operation_id: str


class AsrAudioCommand(FrozenBaseModel):
    type: Literal[VoiceClientEventType.ASR_AUDIO] = VoiceClientEventType.ASR_AUDIO
    operation_id: str
    audio_base64: str


class AsrFinishCommand(FrozenBaseModel):
    type: Literal[VoiceClientEventType.ASR_FINISH] = VoiceClientEventType.ASR_FINISH
    operation_id: str


class AsrCloseCommand(FrozenBaseModel):
    type: Literal[VoiceClientEventType.ASR_CLOSE] = VoiceClientEventType.ASR_CLOSE
    operation_id: str


class LanguageModelGenerateCommand(FrozenBaseModel):
    type: Literal[VoiceClientEventType.LLM_GENERATE] = VoiceClientEventType.LLM_GENERATE
    operation_id: str
    conversation: tuple[ConversationMessage, ...]


class SpeechSynthesizeCommand(FrozenBaseModel):
    type: Literal[VoiceClientEventType.TTS_SYNTHESIZE] = VoiceClientEventType.TTS_SYNTHESIZE
    operation_id: str
    text: str = Field(min_length=1)


class OperationCancelCommand(FrozenBaseModel):
    type: Literal[VoiceClientEventType.OPERATION_CANCEL] = VoiceClientEventType.OPERATION_CANCEL
    operation_id: str


VoiceClientEvent = Annotated[
    AsrStartCommand
    | AsrAudioCommand
    | AsrFinishCommand
    | AsrCloseCommand
    | LanguageModelGenerateCommand
    | SpeechSynthesizeCommand
    | OperationCancelCommand,
    Field(discriminator="type"),
]
voice_client_event_adapter: TypeAdapter[VoiceClientEvent] = TypeAdapter(VoiceClientEvent)


class VoiceServerEventType(StrEnum):
    READY = "compute.ready"
    ASR_PARTIAL = "asr.partial"
    ASR_FINAL = "asr.final"
    LLM_DELTA = "llm.delta"
    LLM_END = "llm.end"
    TTS_START = "tts.start"
    TTS_AUDIO = "tts.audio"
    TTS_END = "tts.end"
    OPERATION_ERROR = "operation.error"


class ComputeReadyEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.READY] = VoiceServerEventType.READY
    output_sample_rate: int = Field(gt=0)


class AsrPartialEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ASR_PARTIAL] = VoiceServerEventType.ASR_PARTIAL
    operation_id: str
    text: str


class AsrFinalEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.ASR_FINAL] = VoiceServerEventType.ASR_FINAL
    operation_id: str
    text: str
    inference_time_seconds: float = Field(ge=0.0)


class LanguageModelDeltaEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.LLM_DELTA] = VoiceServerEventType.LLM_DELTA
    operation_id: str
    text: str


class LanguageModelEndEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.LLM_END] = VoiceServerEventType.LLM_END
    operation_id: str
    inference_time_seconds: float = Field(ge=0.0)


class SpeechStartEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.TTS_START] = VoiceServerEventType.TTS_START
    operation_id: str


class SpeechAudioEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.TTS_AUDIO] = VoiceServerEventType.TTS_AUDIO
    operation_id: str
    audio_base64: str


class SpeechEndEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.TTS_END] = VoiceServerEventType.TTS_END
    operation_id: str
    inference_time_seconds: float = Field(ge=0.0)
    audio_duration_seconds: float = Field(ge=0.0)


class OperationErrorEvent(FrozenBaseModel):
    type: Literal[VoiceServerEventType.OPERATION_ERROR] = VoiceServerEventType.OPERATION_ERROR
    operation_id: str
    message: str


VoiceServerEvent = Annotated[
    ComputeReadyEvent
    | AsrPartialEvent
    | AsrFinalEvent
    | LanguageModelDeltaEvent
    | LanguageModelEndEvent
    | SpeechStartEvent
    | SpeechAudioEvent
    | SpeechEndEvent
    | OperationErrorEvent,
    Field(discriminator="type"),
]
voice_server_event_adapter: TypeAdapter[VoiceServerEvent] = TypeAdapter(VoiceServerEvent)
