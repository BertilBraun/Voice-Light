from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, Self

import torch
from torch import Tensor

from app.training.turn_taking.backbone import FeatureBackbone
from app.training.turn_taking.data import TrainingBatch
from app.training.turn_taking.loss import align_targets
from app.training.turn_taking.model import AdapterOutput, TurnTakingAdapter


class CompletionScoringAdapter(Protocol):
    def to(self, device: torch.device) -> Self: ...

    def eval(self) -> Self: ...

    def train(self) -> Self: ...

    def __call__(
        self,
        feature_taps: tuple[Tensor, ...],
        assistant_speaking: Tensor,
    ) -> AdapterOutput: ...


@dataclass(frozen=True)
class CompletionValidationMetrics:
    auroc: float
    average_precision: float
    binary_cross_entropy: float
    brier_score: float
    support: int


class CompletionValidator:
    def __init__(
        self,
        backbone: FeatureBackbone,
        batches: Iterable[TrainingBatch],
        device: torch.device,
    ) -> None:
        self.backbone = backbone
        self.batches = batches
        self.device = device

    def __call__(self, adapter: TurnTakingAdapter, optimizer_step: int) -> float:
        metrics = evaluate_completion_boundaries(
            backbone=self.backbone,
            adapter=adapter,
            batches=self.batches,
            device=self.device,
        )
        print(
            f"validation_step={optimizer_step}; completion_auroc={metrics.auroc:.6f}; "
            f"completion_ap={metrics.average_precision:.6f}; "
            f"completion_bce={metrics.binary_cross_entropy:.6f}; "
            f"completion_brier={metrics.brier_score:.6f}; support={metrics.support}",
            flush=True,
        )
        return metrics.auroc


def evaluate_completion_boundaries(
    backbone: FeatureBackbone,
    adapter: CompletionScoringAdapter,
    batches: Iterable[TrainingBatch],
    device: torch.device,
) -> CompletionValidationMetrics:
    scored: list[tuple[float, float, int]] = []
    adapter.to(device).eval()
    with torch.no_grad():
        for batch in batches:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = backbone.extract(batch.waveforms, batch.waveform_lengths)
                taps = tuple(tap.to(device) for tap in features.taps)
                assistant_speaking = _align_frame_input(
                    batch.assistant_speaking.to(device),
                    taps[0].shape[1],
                )
                probabilities = adapter(taps, assistant_speaking).event_logits[..., 0].sigmoid()
            targets = align_targets(batch.targets, probabilities.shape[1])
            valid = targets.event_mask[..., 0] & features.frame_mask.cpu().bool()
            for batch_index in range(probabilities.shape[0]):
                indices = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
                if indices.numel() == 0:
                    raise ValueError("Completion validation item has no aligned boundary target.")
                frame_index = int(indices[0])
                target = float(targets.event_targets[batch_index, frame_index, 0])
                probability = float(probabilities[batch_index, frame_index].float().cpu())
                scored.append((probability, target, int(target >= 0.5)))
    adapter.train()
    if not scored:
        raise ValueError("Completion validation produced no scores.")
    probabilities = torch.tensor([probability for probability, _, _ in scored])
    soft_targets = torch.tensor([target for _, target, _ in scored])
    labeled = tuple((probability, label) for probability, _, label in scored)
    return CompletionValidationMetrics(
        auroc=_auroc(labeled),
        average_precision=_average_precision(labeled),
        binary_cross_entropy=float(
            torch.nn.functional.binary_cross_entropy(probabilities, soft_targets)
        ),
        brier_score=float(torch.mean((probabilities - soft_targets) ** 2)),
        support=len(scored),
    )


def _align_frame_input(values: Tensor, frame_count: int) -> Tensor:
    indices = (
        torch.linspace(0, values.shape[1] - 1, frame_count, device=values.device).round().long()
    )
    return values[:, indices]


def _auroc(scored: tuple[tuple[float, int], ...]) -> float:
    positive_count = sum(label for _, label in scored)
    negative_count = len(scored) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Completion AUROC requires both HOLD and EOT labels.")
    ordered = sorted(scored, key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(ordered[end][0], ordered[index][0]):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _average_precision(scored: tuple[tuple[float, int], ...]) -> float:
    positive_count = sum(label for _, label in scored)
    if positive_count == 0:
        raise ValueError("Completion average precision requires EOT labels.")
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(
        sorted(scored, key=lambda item: item[0], reverse=True),
        start=1,
    ):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positive_count
