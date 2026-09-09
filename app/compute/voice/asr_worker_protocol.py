from __future__ import annotations

from base64 import b64decode, b64encode
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from app.compute.voice.schemas import InteractionPrediction, PlaybackCondition
from app.shared.base_model import FrozenBaseModel


class AsrWorkerCommandType(str, Enum):
    START = "start"
    AUDIO = "audio"
    FINISH = "finish"
    SHUTDOWN = "shutdown"


class AsrWorkerEventType(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    FINAL = "final"
    ERROR = "error"
    TURN_PREDICTION = "turn_prediction"
    TURN_ADAPTER_DEGRADED = "turn_adapter_degraded"


class StartAsrCommand(FrozenBaseModel):
    type: Literal[AsrWorkerCommandType.START] = AsrWorkerCommandType.START


class AsrAudioCommand(FrozenBaseModel):
    type: Literal[AsrWorkerCommandType.AUDIO] = AsrWorkerCommandType.AUDIO
    pcm_base64: str
    sequence_number: int
    start_input_sample: int
    end_input_sample: int
    observation_monotonic_time_ns: int
    stream_epoch: int
    turn_epoch: int
    silero_is_speech: bool
    playback_condition: PlaybackCondition

    @classmethod
    def from_observation(
        cls,
        pcm_bytes: bytes,
        sequence_number: int,
        start_input_sample: int,
        end_input_sample: int,
        observation_monotonic_time_ns: int,
        stream_epoch: int,
        turn_epoch: int,
        silero_is_speech: bool,
        playback_condition: PlaybackCondition,
    ) -> AsrAudioCommand:
        return cls(
            pcm_base64=b64encode(pcm_bytes).decode("ascii"),
            sequence_number=sequence_number,
            start_input_sample=start_input_sample,
            end_input_sample=end_input_sample,
            observation_monotonic_time_ns=observation_monotonic_time_ns,
            stream_epoch=stream_epoch,
            turn_epoch=turn_epoch,
            silero_is_speech=silero_is_speech,
            playback_condition=playback_condition,
        )

    def pcm_bytes(self) -> bytes:
        return b64decode(self.pcm_base64, validate=True)


class FinishAsrCommand(FrozenBaseModel):
    type: Literal[AsrWorkerCommandType.FINISH] = AsrWorkerCommandType.FINISH


class ShutdownAsrCommand(FrozenBaseModel):
    type: Literal[AsrWorkerCommandType.SHUTDOWN] = AsrWorkerCommandType.SHUTDOWN


AsrWorkerCommand = Annotated[
    StartAsrCommand | AsrAudioCommand | FinishAsrCommand | ShutdownAsrCommand,
    Field(discriminator="type"),
]
asr_worker_command_adapter: TypeAdapter[AsrWorkerCommand] = TypeAdapter(AsrWorkerCommand)


class AsrWorkerReadyEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.READY] = AsrWorkerEventType.READY
    first_prediction_audio_samples: int = Field(gt=0)
    subsequent_prediction_audio_samples: int = Field(gt=0)
    turn_adapter_available: bool = False
    turn_adapter_checkpoint_sha256: str | None = None
    turn_adapter_error: str | None = None


class PartialAsrEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.PARTIAL] = AsrWorkerEventType.PARTIAL
    text: str


class FinalAsrEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.FINAL] = AsrWorkerEventType.FINAL
    text: str


class AsrWorkerErrorEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.ERROR] = AsrWorkerEventType.ERROR
    message: str


class TurnAdapterPredictionEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.TURN_PREDICTION] = AsrWorkerEventType.TURN_PREDICTION
    prediction: InteractionPrediction


class TurnAdapterDegradedEvent(FrozenBaseModel):
    type: Literal[AsrWorkerEventType.TURN_ADAPTER_DEGRADED] = (
        AsrWorkerEventType.TURN_ADAPTER_DEGRADED
    )
    reason: str


AsrWorkerEvent = Annotated[
    AsrWorkerReadyEvent
    | PartialAsrEvent
    | FinalAsrEvent
    | AsrWorkerErrorEvent
    | TurnAdapterPredictionEvent
    | TurnAdapterDegradedEvent,
    Field(discriminator="type"),
]
asr_worker_event_adapter: TypeAdapter[AsrWorkerEvent] = TypeAdapter(AsrWorkerEvent)
