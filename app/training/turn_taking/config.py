from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.shared.base_model import FrozenBaseModel
from app.shared.model_constants import NEMOTRON_ASR_MODEL_NAME, NEMOTRON_ASR_MODEL_REVISION

PINNED_NEMOTRON_REVISION = NEMOTRON_ASR_MODEL_REVISION


class AdapterConfig(FrozenBaseModel):
    feature_dimension: int = 1024
    tap_layer_indices: tuple[int, ...] = (6, 12, 18, 24)
    tap_projection_dimension: int = 32
    fused_dimension: int = 64
    recurrent_dimension: int = 64
    recurrent_layers: int = 1
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class UserYieldObjectiveConfig(FrozenBaseModel):
    kind: Literal["user_yield"] = "user_yield"


class TurnCompletionObjectiveConfig(FrozenBaseModel):
    kind: Literal["turn_completion"] = "turn_completion"
    hold_completion_maximum: float = Field(default=0.2, ge=0.0, le=1.0)
    hold_continuation_minimum: float = Field(default=0.8, ge=0.0, le=1.0)
    eot_completion_minimum: float = Field(default=0.8, ge=0.0, le=1.0)
    conflicting_continuation_minimum: float = Field(default=0.8, ge=0.0, le=1.0)


PrimaryObjectiveConfig = Annotated[
    UserYieldObjectiveConfig | TurnCompletionObjectiveConfig,
    Field(discriminator="kind"),
]


class LossConfig(FrozenBaseModel):
    primary_objective: PrimaryObjectiveConfig = UserYieldObjectiveConfig()
    event_weight: float = Field(default=0.25, ge=0.0)
    future_activity_weight: float = Field(default=0.25, ge=0.0)


class TrainingPrecision(StrEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


class WaveformAugmentationProfile(StrEnum):
    LEGACY = "legacy"
    EXPANDED = "expanded"


class SyntheticAnchorKind(StrEnum):
    EOT = "eot"
    HOLD = "hold"
    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"
    RESPONSE = "response"
    ASSISTANT_STATE = "assistant_state"
    USER_STATE = "user_state"


class SyntheticAnchorSamplingConfig(FrozenBaseModel):
    eot_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    hold_fraction: float = Field(default=0.2, gt=0.0, le=1.0)
    backchannel_fraction: float = Field(default=0.15, gt=0.0, le=1.0)
    interruption_fraction: float = Field(default=0.15, gt=0.0, le=1.0)
    response_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    assistant_state_fraction: float = Field(default=0.075, gt=0.0, le=1.0)
    user_state_fraction: float = Field(default=0.075, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> SyntheticAnchorSamplingConfig:
        if abs(sum(self.fractions()) - 1.0) > 1e-9:
            raise ValueError("Synthetic anchor sampling fractions must sum to one.")
        return self

    def fractions(self) -> tuple[float, ...]:
        return (
            self.eot_fraction,
            self.hold_fraction,
            self.backchannel_fraction,
            self.interruption_fraction,
            self.response_fraction,
            self.assistant_state_fraction,
            self.user_state_fraction,
        )

    def fraction(self, kind: SyntheticAnchorKind) -> float:
        match kind:
            case SyntheticAnchorKind.EOT:
                return self.eot_fraction
            case SyntheticAnchorKind.HOLD:
                return self.hold_fraction
            case SyntheticAnchorKind.BACKCHANNEL:
                return self.backchannel_fraction
            case SyntheticAnchorKind.INTERRUPTION:
                return self.interruption_fraction
            case SyntheticAnchorKind.RESPONSE:
                return self.response_fraction
            case SyntheticAnchorKind.ASSISTANT_STATE:
                return self.assistant_state_fraction
            case SyntheticAnchorKind.USER_STATE:
                return self.user_state_fraction


class WaveformAugmentationConfig(FrozenBaseModel):
    gain_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_gain_db: float = -12.0
    maximum_gain_db: float = 6.0
    noise_probability: float = Field(default=0.4, ge=0.0, le=1.0)
    minimum_signal_to_noise_db: float = 5.0
    maximum_signal_to_noise_db: float = 30.0
    reverberation_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    bandwidth_probability: float = Field(default=0.25, ge=0.0, le=1.0)
    clipping_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    packet_loss_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    minimum_packet_loss_seconds: float = Field(default=0.02, gt=0.0)
    maximum_packet_loss_seconds: float = Field(default=0.12, gt=0.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> WaveformAugmentationConfig:
        if self.minimum_gain_db > self.maximum_gain_db:
            raise ValueError("minimum_gain_db must not exceed maximum_gain_db.")
        if self.minimum_signal_to_noise_db > self.maximum_signal_to_noise_db:
            raise ValueError(
                "minimum_signal_to_noise_db must not exceed maximum_signal_to_noise_db."
            )
        if self.minimum_packet_loss_seconds > self.maximum_packet_loss_seconds:
            raise ValueError(
                "minimum_packet_loss_seconds must not exceed maximum_packet_loss_seconds."
            )
        return self


def waveform_augmentation_config(
    profile: WaveformAugmentationProfile,
) -> WaveformAugmentationConfig:
    match profile:
        case WaveformAugmentationProfile.LEGACY:
            return WaveformAugmentationConfig(
                noise_probability=0.3,
                reverberation_probability=0.0,
                bandwidth_probability=0.0,
                clipping_probability=0.0,
                packet_loss_probability=0.1,
                minimum_packet_loss_seconds=0.04,
                maximum_packet_loss_seconds=0.04,
            )
        case WaveformAugmentationProfile.EXPANDED:
            return WaveformAugmentationConfig()


class TrainingConfig(FrozenBaseModel):
    model_identifier: str = NEMOTRON_ASR_MODEL_NAME
    model_revision: str = PINNED_NEMOTRON_REVISION
    sample_rate_hz: int = 16_000
    context_seconds: float = 20.0
    burn_in_seconds: float = 4.0
    encoder_frame_seconds: float = 0.08
    lookahead_tokens: int = 1
    batch_size: int = 2
    validation_batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = 8
    data_loader_workers: int = Field(default=4, ge=0)
    data_loader_prefetch_factor: int = Field(default=2, gt=0)
    precision: TrainingPrecision = TrainingPrecision.BFLOAT16
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 350
    max_steps: int = 3_500
    target_optimizer_step: int | None = Field(default=None, gt=0)
    checkpoint_interval_steps: int = Field(default=250, gt=0)
    progress_interval_steps: int = Field(default=10, gt=0)
    validation_interval_steps: int = Field(default=125, gt=0)
    minimum_steps_before_stopping: int = Field(default=1_000, ge=0)
    early_stopping_patience: int = Field(default=4, gt=0)
    gradient_clip_norm: float = 1.0
    random_seed: int = 17
    unmeasured_reliability_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    augmentation: WaveformAugmentationConfig = WaveformAugmentationConfig()
    adapter: AdapterConfig = AdapterConfig()
    loss: LossConfig = LossConfig()
