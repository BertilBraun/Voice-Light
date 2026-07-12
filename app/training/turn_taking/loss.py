from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from app.training.turn_taking.data import FrameTargets
from app.training.turn_taking.model import AdapterOutput


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    policy: Tensor
    future_activity: Tensor
    end_horizons: Tensor
    time_to_shift: Tensor
    events: Tensor
    monotonic: Tensor


def compute_loss(output: AdapterOutput, targets: FrameTargets, frame_mask: Tensor) -> LossBreakdown:
    targets = align_targets(targets, output.policy_logits.shape[1])
    mask = targets.loss_mask & frame_mask
    weights = targets.confidence * mask.float()
    policy_values = nn.functional.cross_entropy(
        output.policy_logits.transpose(1, 2), targets.policy, reduction="none"
    )
    policy = _weighted_mean(policy_values, weights)
    future_activity = _weighted_mean(
        nn.functional.binary_cross_entropy_with_logits(
            output.future_activity_logits, targets.future_activity, reduction="none"
        ).mean(dim=-1),
        weights,
    )
    end_horizons = _weighted_mean(
        nn.functional.binary_cross_entropy_with_logits(
            output.end_horizon_logits, targets.end_horizons, reduction="none"
        ).mean(dim=-1),
        weights,
    )
    time_to_shift = _weighted_mean(
        nn.functional.cross_entropy(
            output.time_to_shift_logits.transpose(1, 2),
            targets.time_to_shift_bucket,
            reduction="none",
        ),
        weights,
    )
    events = _weighted_mean(
        nn.functional.binary_cross_entropy_with_logits(
            output.event_logits, targets.events, reduction="none"
        ).mean(dim=-1),
        weights,
    )
    probabilities = output.end_horizon_logits.sigmoid()
    monotonic_values = nn.functional.relu(
        probabilities[..., 0] - probabilities[..., 1]
    ) + nn.functional.relu(probabilities[..., 1] - probabilities[..., 2])
    monotonic = _weighted_mean(monotonic_values, weights)
    total = (
        policy
        + 0.5 * future_activity
        + 0.5 * end_horizons
        + 0.25 * time_to_shift
        + 0.2 * events
        + 0.05 * monotonic
    )
    return LossBreakdown(
        total=total,
        policy=policy,
        future_activity=future_activity,
        end_horizons=end_horizons,
        time_to_shift=time_to_shift,
        events=events,
        monotonic=monotonic,
    )


def align_targets(targets: FrameTargets, frame_count: int) -> FrameTargets:
    source_count = targets.policy.shape[1]
    indices = (
        torch.linspace(0, source_count - 1, frame_count, device=targets.policy.device)
        .round()
        .long()
    )
    return FrameTargets(
        policy=targets.policy[:, indices],
        future_activity=targets.future_activity[:, indices],
        end_horizons=targets.end_horizons[:, indices],
        time_to_shift_bucket=targets.time_to_shift_bucket[:, indices],
        events=targets.events[:, indices],
        confidence=targets.confidence[:, indices],
        loss_mask=targets.loss_mask[:, indices],
    )


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)
