from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from app.training.turn_taking.backbone import FeatureBackbone
from app.training.turn_taking.config import TrainingConfig, TrainingPrecision
from app.training.turn_taking.data import FrameTargets, TrainingBatch
from app.training.turn_taking.loss import LossBreakdown, compute_loss
from app.training.turn_taking.model import AdapterOutput, TurnTakingAdapter


@dataclass(frozen=True)
class TrainingResult:
    starting_optimizer_step: int
    optimizer_steps: int
    final_loss: float
    elapsed_seconds: float
    optimizer_steps_per_second: float
    peak_device_memory_bytes: int | None
    peak_reserved_device_memory_bytes: int | None
    checkpoint_path: Path


def train(
    backbone: FeatureBackbone,
    adapter: TurnTakingAdapter,
    batches: Iterable[TrainingBatch],
    config: TrainingConfig,
    checkpoint_path: Path,
    device: torch.device,
    resume_checkpoint_path: Path | None = None,
) -> TrainingResult:
    _validate_precision(config.precision, device)
    adapter.to(device)
    adapter.train()
    optimizer = AdamW(
        adapter.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    optimizer_step = 0
    schedule_config = config
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        checkpoint_config = TrainingConfig.model_validate(checkpoint["training_config"])
        _validate_resume_config(config, checkpoint_config)
        adapter.load_state_dict(checkpoint["adapter_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_step = checkpoint["optimizer_step"]
        schedule_config = TrainingConfig.model_validate(
            checkpoint.get("schedule_config", checkpoint["training_config"])
        )
    target_optimizer_step = config.target_optimizer_step or config.max_steps
    if target_optimizer_step <= optimizer_step:
        raise ValueError(
            f"Target optimizer step {target_optimizer_step} must exceed saved step "
            f"{optimizer_step}."
        )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        schedule_config=schedule_config,
        optimizer_step=optimizer_step,
    )
    if resume_checkpoint_path is not None and "scheduler_state" in checkpoint:
        saved_scheduler_step = checkpoint["scheduler_state"]["last_epoch"]
        if saved_scheduler_step != optimizer_step:
            raise ValueError("Checkpoint scheduler step does not match its optimizer step.")
    optimizer.zero_grad(set_to_none=True)
    starting_optimizer_step = optimizer_step
    micro_step = 0
    final_loss = math.nan
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    while optimizer_step < target_optimizer_step:
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
            if optimizer_step % config.checkpoint_interval_steps == 0:
                save_checkpoint(
                    checkpoint_path,
                    adapter,
                    optimizer,
                    scheduler,
                    optimizer_step,
                    config,
                    schedule_config,
                )
                print(
                    f"step={optimizer_step}; loss={final_loss:.6f}; "
                    f"learning_rate={scheduler.get_last_lr()[0]:.8f}",
                    flush=True,
                )
            if optimizer_step >= target_optimizer_step:
                break
        if not observed_batch:
            raise ValueError("Training data loader produced no batches.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    peak_device_memory_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    peak_reserved_device_memory_bytes = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    save_checkpoint(
        checkpoint_path,
        adapter,
        optimizer,
        scheduler,
        optimizer_step,
        config,
        schedule_config,
    )
    performed_optimizer_steps = optimizer_step - starting_optimizer_step
    return TrainingResult(
        starting_optimizer_step=starting_optimizer_step,
        optimizer_steps=optimizer_step,
        final_loss=final_loss,
        elapsed_seconds=elapsed_seconds,
        optimizer_steps_per_second=performed_optimizer_steps / elapsed_seconds,
        peak_device_memory_bytes=peak_device_memory_bytes,
        peak_reserved_device_memory_bytes=peak_reserved_device_memory_bytes,
        checkpoint_path=checkpoint_path,
    )


def train_micro_batch(
    backbone: FeatureBackbone,
    adapter: TurnTakingAdapter,
    batch: TrainingBatch,
    config: TrainingConfig,
    device: torch.device,
) -> LossBreakdown:
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and config.precision is TrainingPrecision.BFLOAT16,
    ):
        features = backbone.extract(batch.waveforms, batch.waveform_lengths)
        taps = tuple(tap.to(device) for tap in features.taps)
        assistant_speaking = _align_frame_input(
            values=batch.assistant_speaking.to(device),
            frame_count=taps[0].shape[1],
        )
        output = adapter(taps, assistant_speaking)
    targets = _targets_to_device(batch.targets, device)
    return compute_loss(
        _output_to_float32(output),
        targets,
        features.frame_mask.to(device),
        config.loss,
    )


def save_checkpoint(
    path: Path,
    adapter: TurnTakingAdapter,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    optimizer_step: int,
    config: TrainingConfig,
    schedule_config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(
        {
            "checkpoint_version": 2,
            "adapter_state": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "optimizer_step": optimizer_step,
            "training_config": config.model_dump(mode="json"),
            "schedule_config": schedule_config.model_dump(mode="json"),
        },
        temporary_path,
    )
    temporary_path.replace(path)


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


def _output_to_float32(output: AdapterOutput) -> AdapterOutput:
    return AdapterOutput(
        yield_logits=output.yield_logits.float(),
        future_activity_logits=output.future_activity_logits.float(),
        event_logits=output.event_logits.float(),
        recurrent_state=output.recurrent_state.float(),
    )


def _validate_precision(precision: TrainingPrecision, device: torch.device) -> None:
    if (
        device.type == "cuda"
        and precision is TrainingPrecision.BFLOAT16
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("The selected CUDA device does not support bfloat16 training.")


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


def _build_scheduler(
    optimizer: Optimizer,
    schedule_config: TrainingConfig,
    optimizer_step: int,
) -> LambdaLR:
    if optimizer_step == 0:
        return LambdaLR(optimizer, _learning_rate_multiplier(schedule_config))
    for parameter_group in optimizer.param_groups:
        parameter_group.setdefault("initial_lr", schedule_config.learning_rate)
    return LambdaLR(
        optimizer,
        _learning_rate_multiplier(schedule_config),
        last_epoch=optimizer_step - 1,
    )


def _validate_resume_config(config: TrainingConfig, checkpoint: TrainingConfig) -> None:
    if (
        config.model_copy(
            update={
                "target_optimizer_step": checkpoint.target_optimizer_step,
                "checkpoint_interval_steps": checkpoint.checkpoint_interval_steps,
                "data_loader_workers": checkpoint.data_loader_workers,
                "data_loader_prefetch_factor": checkpoint.data_loader_prefetch_factor,
            }
        )
        != checkpoint
    ):
        raise ValueError(
            "Resume configuration changes the saved model, optimizer, loss, batch, or schedule."
        )
