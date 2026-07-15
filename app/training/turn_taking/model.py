from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from app.training.turn_taking.config import AdapterConfig


@dataclass(frozen=True)
class AdapterOutput:
    yield_logits: Tensor
    future_activity_logits: Tensor
    event_logits: Tensor
    recurrent_state: Tensor


class CausalConvolutionBlock(nn.Module):
    def __init__(self, dimension: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            dimension,
            dimension,
            kernel_size,
            groups=dimension,
            dilation=dilation,
        )
        self.pointwise = nn.Conv1d(dimension, dimension, 1)
        self.normalization = nn.LayerNorm(dimension)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        residual = features
        temporal = features.transpose(1, 2)
        temporal = nn.functional.pad(temporal, (self.left_padding, 0))
        temporal = self.pointwise(self.depthwise(temporal)).transpose(1, 2)
        return self.normalization(residual + self.dropout(self.activation(temporal)))


class TurnTakingAdapter(nn.Module):
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.tap_normalizations = nn.ModuleList(
            nn.LayerNorm(config.feature_dimension) for _ in config.tap_layer_indices
        )
        self.tap_projections = nn.ModuleList(
            nn.Linear(config.feature_dimension, config.tap_projection_dimension)
            for _ in config.tap_layer_indices
        )
        concatenated_dimension = config.tap_projection_dimension * len(config.tap_layer_indices)
        self.fusion = nn.Sequential(
            nn.Linear(concatenated_dimension, config.fused_dimension),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.convolution_blocks = nn.ModuleList(
            CausalConvolutionBlock(
                config.fused_dimension, kernel_size=5, dilation=dilation, dropout=config.dropout
            )
            for dilation in (1, 2)
        )
        self.recurrent = nn.GRU(
            input_size=config.fused_dimension,
            hidden_size=config.recurrent_dimension,
            num_layers=config.recurrent_layers,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
            batch_first=True,
        )
        self.yield_head = nn.Linear(config.recurrent_dimension, 1)
        self.future_activity_head = nn.Linear(config.recurrent_dimension, 4)
        self.event_head = nn.Linear(config.recurrent_dimension, 5)

    def forward(
        self, feature_taps: tuple[Tensor, ...], recurrent_state: Tensor | None = None
    ) -> AdapterOutput:
        if len(feature_taps) != len(self.tap_projections):
            raise ValueError(
                f"Expected {len(self.tap_projections)} feature taps, received {len(feature_taps)}."
            )
        projected = [
            projection(normalization(features))
            for features, normalization, projection in zip(
                feature_taps, self.tap_normalizations, self.tap_projections, strict=True
            )
        ]
        temporal = self.fusion(torch.cat(projected, dim=-1))
        for block in self.convolution_blocks:
            temporal = block(temporal)
        temporal, next_state = self.recurrent(temporal, recurrent_state)
        return AdapterOutput(
            yield_logits=self.yield_head(temporal).squeeze(-1),
            future_activity_logits=self.future_activity_head(temporal),
            event_logits=self.event_head(temporal),
            recurrent_state=next_state,
        )


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
