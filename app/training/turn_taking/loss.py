from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from app.training.turn_taking.config import LossConfig
from app.training.turn_taking.data import FrameTargets
from app.training.turn_taking.model import AdapterOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    primary: Tensor
    events: Tensor
    future_activity: Tensor


def compute_loss(
    output: AdapterOutput,
    targets: FrameTargets,
    frame_mask: Tensor,
    config: LossConfig,
) -> LossBreakdown:
    targets = align_targets(targets, output.yield_logits.shape[1])
    primary_weights = targets.primary_weight * targets.primary_mask.float() * frame_mask.float()
    primary = _weighted_mean(
        nn.functional.binary_cross_entropy_with_logits(
            output.yield_logits, targets.yield_probability, reduction="none"
        ),
        primary_weights,
    )
    event_values = -(
        targets.event_distribution * nn.functional.log_softmax(output.event_logits, dim=-1)
    ).sum(dim=-1)
    event_weights = targets.event_weight * targets.event_mask.float() * frame_mask.float()
    events = _weighted_mean(event_values, event_weights)
    future_mask = targets.future_activity_mask & frame_mask.unsqueeze(-1)
    future_activity = _weighted_mean(
        nn.functional.binary_cross_entropy_with_logits(
            output.future_activity_logits, targets.future_activity, reduction="none"
        ),
        future_mask.float(),
    )
    total = primary + config.event_weight * events + config.future_activity_weight * future_activity
    return LossBreakdown(
        total=total,
        primary=primary,
        events=events,
        future_activity=future_activity,
    )


def align_targets(targets: FrameTargets, frame_count: int) -> FrameTargets:
    source_count = targets.yield_probability.shape[1]
    indices = (
        torch.linspace(0, source_count - 1, frame_count, device=targets.yield_probability.device)
        .round()
        .long()
    )
    return FrameTargets(
        yield_probability=targets.yield_probability[:, indices],
        primary_weight=targets.primary_weight[:, indices],
        primary_mask=targets.primary_mask[:, indices],
        event_distribution=targets.event_distribution[:, indices],
        event_weight=targets.event_weight[:, indices],
        event_mask=targets.event_mask[:, indices],
        future_activity=targets.future_activity[:, indices],
        future_activity_mask=targets.future_activity_mask[:, indices],
    )


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)
