from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from app.shared.base_model import FrozenBaseModel


class AdapterConfig(FrozenBaseModel):
    feature_dimension: int = 1024
    tap_layer_indices: tuple[int, ...] = (6, 12, 18, 24)
    tap_projection_dimension: int = 32
    fused_dimension: int = 64
    recurrent_dimension: int = 64
    recurrent_layers: int = 1
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class LossConfig(FrozenBaseModel):
    event_weight: float = Field(default=0.25, ge=0.0)
    future_activity_weight: float = Field(default=0.25, ge=0.0)


class TrainingPrecision(StrEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


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

    @model_validator(mode="after")
    def validate_ranges(self) -> WaveformAugmentationConfig:
        if self.minimum_gain_db > self.maximum_gain_db:
            raise ValueError("minimum_gain_db must not exceed maximum_gain_db.")
        if self.minimum_signal_to_noise_db > self.maximum_signal_to_noise_db:
            raise ValueError(
                "minimum_signal_to_noise_db must not exceed maximum_signal_to_noise_db."
            )
        return self


class TrainingConfig(FrozenBaseModel):
    model_identifier: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    sample_rate_hz: int = 16_000
    context_seconds: float = 20.0
    burn_in_seconds: float = 4.0
    encoder_frame_seconds: float = 0.08
    lookahead_tokens: int = 1
    batch_size: int = 2
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
    validation_interval_steps: int = 250
    gradient_clip_norm: float = 1.0
    random_seed: int = 17
    unmeasured_reliability_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    augmentation: WaveformAugmentationConfig = WaveformAugmentationConfig()
    adapter: AdapterConfig = AdapterConfig()
    loss: LossConfig = LossConfig()
