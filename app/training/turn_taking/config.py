from __future__ import annotations

from pydantic import Field

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


class TrainingConfig(FrozenBaseModel):
    model_identifier: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    sample_rate_hz: int = 16_000
    context_seconds: float = 20.0
    burn_in_seconds: float = 4.0
    encoder_frame_seconds: float = 0.08
    lookahead_tokens: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 2_000
    max_steps: int = 50_000
    validation_interval_steps: int = 1_000
    gradient_clip_norm: float = 1.0
    random_seed: int = 17
    unmeasured_reliability_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    adapter: AdapterConfig = AdapterConfig()
    loss: LossConfig = LossConfig()
