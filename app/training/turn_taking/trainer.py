from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from app.training.turn_taking.backbone import FeatureBackbone
from app.training.turn_taking.config import TrainingConfig
from app.training.turn_taking.data import FrameTargets, TrainingBatch
from app.training.turn_taking.loss import LossBreakdown, compute_loss
from app.training.turn_taking.model import TurnTakingAdapter


@dataclass(frozen=True)
class TrainingResult:
    optimizer_steps: int
    final_loss: float
    checkpoint_path: Path


def train(
    backbone: FeatureBackbone,
    adapter: TurnTakingAdapter,
    batches: Iterable[TrainingBatch],
    config: TrainingConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> TrainingResult:
    adapter.to(device)
    adapter.train()
    optimizer = AdamW(
        adapter.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = LambdaLR(optimizer, _learning_rate_multiplier(config))
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    final_loss = math.nan
    while optimizer_step < config.max_steps:
        observed_batch = False
        for batch in batches:
            observed_batch = True
            loss = train_micro_batch(backbone, adapter, batch, config, device)
            (loss.total / config.gradient_accumulation_steps).backward()
            final_loss = float(loss.total.detach().cpu())
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps != 0:
                continue
            nn.utils.clip_grad_norm_(adapter.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step >= config.max_steps:
                break
        if not observed_batch:
            raise ValueError("Training data loader produced no batches.")
    save_checkpoint(checkpoint_path, adapter, optimizer, optimizer_step, config)
    return TrainingResult(
        optimizer_steps=optimizer_step,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
    )


def train_micro_batch(
    backbone: FeatureBackbone,
    adapter: TurnTakingAdapter,
    batch: TrainingBatch,
    config: TrainingConfig,
    device: torch.device,
) -> LossBreakdown:
    features = backbone.extract(batch.waveforms, batch.waveform_lengths)
    taps = tuple(tap.to(device) for tap in features.taps)
    assistant_speaking = _align_frame_input(
        values=batch.assistant_speaking.to(device),
        frame_count=taps[0].shape[1],
    )
    output = adapter(taps, assistant_speaking)
    targets = _targets_to_device(batch.targets, device)
    return compute_loss(output, targets, features.frame_mask.to(device), config.loss)


def save_checkpoint(
    path: Path,
    adapter: TurnTakingAdapter,
    optimizer: Optimizer,
    optimizer_step: int,
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_step": optimizer_step,
            "training_config": config.model_dump(mode="json"),
        },
        path,
    )


def _targets_to_device(targets: FrameTargets, device: torch.device) -> FrameTargets:
    return FrameTargets(
        yield_probability=targets.yield_probability.to(device),
        primary_weight=targets.primary_weight.to(device),
        primary_mask=targets.primary_mask.to(device),
        event_targets=targets.event_targets.to(device),
        event_mask=targets.event_mask.to(device),
        future_activity=targets.future_activity.to(device),
        future_activity_mask=targets.future_activity_mask.to(device),
    )


def _align_frame_input(values: torch.Tensor, frame_count: int) -> torch.Tensor:
    source_count = values.shape[1]
    indices = torch.linspace(0, source_count - 1, frame_count, device=values.device).round().long()
    return values[:, indices]


def _learning_rate_multiplier(config: TrainingConfig) -> Callable[[int], float]:
    minimum_ratio = config.minimum_learning_rate / config.learning_rate

    def multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return max(1, step) / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return multiplier
