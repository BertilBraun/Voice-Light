from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import torch
from pydantic import Field
from torch import Tensor

from app.local.training_corpus.export import MaterializedTrainingSample
from app.shared.base_model import FrozenBaseModel
from app.training.turn_taking.backbone import FeatureBackbone
from app.training.turn_taking.config import LossConfig
from app.training.turn_taking.data import FrameTargets, TrainingBatch
from app.training.turn_taking.hub import frame_targets_from_sample
from app.training.turn_taking.loss import align_targets
from app.training.turn_taking.model import AdapterOutput, TurnTakingAdapter


class EvaluationHead(StrEnum):
    USER_YIELD = "user_yield"
    TURN_COMPLETION = "turn_completion"
    CONTINUATION_PAUSE = "continuation_pause"
    ASSISTANT_BACKCHANNEL = "assistant_backchannel"
    NON_FLOOR_FEEDBACK = "non_floor_feedback"
    FLOOR_TAKE = "floor_take"
    FUTURE_ACTIVITY_0_200 = "future_activity_0_200"
    FUTURE_ACTIVITY_200_500 = "future_activity_200_500"
    FUTURE_ACTIVITY_500_1000 = "future_activity_500_1000"
    FUTURE_ACTIVITY_1000_1500 = "future_activity_1000_1500"


class EvaluationModel(StrEnum):
    TRAINED_ADAPTER = "trained_adapter"
    RANDOM_ADAPTER = "random_adapter"
    CLASS_PRIOR = "class_prior"
    ASSISTANT_FLOOR_HEURISTIC = "assistant_floor_heuristic"
    ORACLE_VAD = "oracle_vad"


class HeadMetrics(FrozenBaseModel):
    head: EvaluationHead
    binary_cross_entropy: float = Field(ge=0.0)
    brier_score: float = Field(ge=0.0)
    target_mean: float = Field(ge=0.0, le=1.0)
    prediction_mean: float = Field(ge=0.0, le=1.0)
    effective_support: float = Field(gt=0.0)


class ModelEvaluation(FrozenBaseModel):
    model: EvaluationModel
    total_loss: float = Field(ge=0.0)
    primary_loss: float = Field(ge=0.0)
    event_loss: float = Field(ge=0.0)
    future_activity_loss: float = Field(ge=0.0)
    heads: tuple[HeadMetrics, ...]


class EvaluationReport(FrozenBaseModel):
    schema_version: str = "voice-light-turn-taking-evaluation-v1"
    generated_at: datetime
    hub_revision: str
    checkpoint_sha256: str
    optimizer_step: int = Field(ge=0)
    validation_sample_count: int = Field(gt=0)
    random_seed: int
    class_priors: tuple[float, ...]
    models: tuple[ModelEvaluation, ...]


class EvaluationComparison(FrozenBaseModel):
    reference_optimizer_step: int = Field(ge=0)
    candidate_optimizer_step: int = Field(gt=0)
    reference_total_loss: float = Field(ge=0.0)
    candidate_total_loss: float = Field(ge=0.0)
    absolute_improvement: float
    improved: bool


@dataclass(frozen=True)
class PredictionProbabilities:
    user_yield: Tensor
    events: Tensor
    future_activity: Tensor


@dataclass
class _HeadAccumulator:
    binary_cross_entropy_sum: float = 0.0
    brier_sum: float = 0.0
    target_sum: float = 0.0
    prediction_sum: float = 0.0
    weight_sum: float = 0.0

    def update(self, predictions: Tensor, targets: Tensor, weights: Tensor) -> None:
        probabilities = predictions.float().clamp(1e-7, 1.0 - 1e-7)
        target_values = targets.float()
        weight_values = weights.float()
        self.binary_cross_entropy_sum += float(
            (
                -(
                    target_values * probabilities.log()
                    + (1.0 - target_values) * (1.0 - probabilities).log()
                )
                * weight_values
            )
            .sum()
            .item()
        )
        self.brier_sum += float(
            (((probabilities - target_values) ** 2) * weight_values).sum().item()
        )
        self.target_sum += float((target_values * weight_values).sum().item())
        self.prediction_sum += float((probabilities * weight_values).sum().item())
        self.weight_sum += float(weight_values.sum().item())

    def finalize(self, head: EvaluationHead) -> HeadMetrics:
        if self.weight_sum <= 0.0:
            raise ValueError(f"Evaluation head {head.value!r} has no valid targets.")
        return HeadMetrics(
            head=head,
            binary_cross_entropy=self.binary_cross_entropy_sum / self.weight_sum,
            brier_score=self.brier_sum / self.weight_sum,
            target_mean=self.target_sum / self.weight_sum,
            prediction_mean=self.prediction_sum / self.weight_sum,
            effective_support=self.weight_sum,
        )


class EvaluationAccumulator:
    def __init__(self) -> None:
        self.accumulators = tuple(_HeadAccumulator() for _ in EvaluationHead)

    def update(
        self,
        predictions: PredictionProbabilities,
        targets: FrameTargets,
        frame_mask: Tensor,
    ) -> None:
        aligned_targets = align_targets(targets, predictions.user_yield.shape[1])
        valid_frames = frame_mask.cpu().bool()
        primary_weights = (
            aligned_targets.primary_weight.cpu()
            * aligned_targets.primary_mask.cpu().float()
            * valid_frames.float()
        )
        self.accumulators[0].update(
            predictions.user_yield.cpu(),
            aligned_targets.yield_probability.cpu(),
            primary_weights,
        )
        for index in range(5):
            self.accumulators[index + 1].update(
                predictions.events[..., index].cpu(),
                aligned_targets.event_targets[..., index].cpu(),
                aligned_targets.event_mask[..., index].cpu() * valid_frames,
            )
        for index in range(4):
            self.accumulators[index + 6].update(
                predictions.future_activity[..., index].cpu(),
                aligned_targets.future_activity[..., index].cpu(),
                aligned_targets.future_activity_mask[..., index].cpu() * valid_frames,
            )

    def finalize(self, model: EvaluationModel, loss_config: LossConfig) -> ModelEvaluation:
        heads = tuple(
            accumulator.finalize(head)
            for head, accumulator in zip(EvaluationHead, self.accumulators, strict=True)
        )
        primary_loss = heads[0].binary_cross_entropy
        event_loss = _weighted_head_mean(heads[1:6])
        future_activity_loss = _weighted_head_mean(heads[6:10])
        return ModelEvaluation(
            model=model,
            total_loss=(
                primary_loss
                + loss_config.event_weight * event_loss
                + loss_config.future_activity_weight * future_activity_loss
            ),
            primary_loss=primary_loss,
            event_loss=event_loss,
            future_activity_loss=future_activity_loss,
            heads=heads,
        )


def adapter_probabilities(output: AdapterOutput) -> PredictionProbabilities:
    return PredictionProbabilities(
        user_yield=output.yield_logits.sigmoid(),
        events=output.event_logits.sigmoid(),
        future_activity=output.future_activity_logits.sigmoid(),
    )


def constant_probabilities(
    priors: tuple[float, ...],
    batch_size: int,
    frame_count: int,
) -> PredictionProbabilities:
    if len(priors) != len(EvaluationHead):
        raise ValueError(f"Expected {len(EvaluationHead)} class priors, received {len(priors)}.")
    values = torch.tensor(priors, dtype=torch.float32)
    return PredictionProbabilities(
        user_yield=values[0].expand(batch_size, frame_count),
        events=values[1:6].expand(batch_size, frame_count, 5),
        future_activity=values[6:10].expand(batch_size, frame_count, 4),
    )


def assistant_floor_probabilities(
    assistant_speaking: Tensor,
    priors: tuple[float, ...],
    frame_count: int,
) -> PredictionProbabilities:
    indices = torch.linspace(0, assistant_speaking.shape[1] - 1, frame_count).round().long()
    aligned_speaking = assistant_speaking[:, indices].bool()
    probabilities = constant_probabilities(priors, assistant_speaking.shape[0], frame_count)
    return PredictionProbabilities(
        user_yield=torch.where(aligned_speaking, 0.99, 0.01),
        events=probabilities.events,
        future_activity=probabilities.future_activity,
    )


def oracle_vad_probabilities(
    user_has_floor: Tensor,
    priors: tuple[float, ...],
) -> PredictionProbabilities:
    if user_has_floor.ndim != 2:
        raise ValueError("Oracle VAD input must contain batch and frame dimensions.")
    valid = user_has_floor >= 0.0
    user_yield = torch.where(valid, 1.0 - user_has_floor.clamp(0.0, 1.0), 0.5)
    probabilities = constant_probabilities(priors, user_has_floor.shape[0], user_has_floor.shape[1])
    return PredictionProbabilities(
        user_yield=user_yield,
        events=probabilities.events,
        future_activity=probabilities.future_activity,
    )


def fit_class_priors(target_batches: Iterable[FrameTargets]) -> tuple[float, ...]:
    accumulator = EvaluationAccumulator()
    for raw_targets in target_batches:
        targets = _batched_targets(raw_targets)
        batch_size, frame_count = targets.yield_probability.shape
        accumulator.update(
            constant_probabilities((0.5,) * len(EvaluationHead), batch_size, frame_count),
            targets,
            torch.ones((batch_size, frame_count), dtype=torch.bool),
        )
    priors: list[float] = []
    for head_accumulator in accumulator.accumulators:
        if head_accumulator.weight_sum <= 0.0:
            raise ValueError("Cannot fit a class prior without valid training targets.")
        priors.append(head_accumulator.target_sum / head_accumulator.weight_sum)
    if any(not math.isfinite(prior) for prior in priors):
        raise ValueError("Class priors must be finite.")
    return tuple(priors)


def evaluate_models(
    backbone: FeatureBackbone,
    trained_adapter: TurnTakingAdapter,
    random_adapter: TurnTakingAdapter,
    batches: Iterable[TrainingBatch],
    priors: tuple[float, ...],
    loss_config: LossConfig,
    device: torch.device,
) -> tuple[ModelEvaluation, ...]:
    trained_adapter.to(device).eval()
    random_adapter.to(device).eval()
    trained_accumulator = EvaluationAccumulator()
    random_accumulator = EvaluationAccumulator()
    prior_accumulator = EvaluationAccumulator()
    heuristic_accumulator = EvaluationAccumulator()
    with torch.no_grad():
        for batch in batches:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = backbone.extract(batch.waveforms, batch.waveform_lengths)
                taps = tuple(tap.to(device) for tap in features.taps)
                assistant_speaking = _align_assistant_speaking(
                    batch.assistant_speaking.to(device), taps[0].shape[1]
                )
                trained_output = trained_adapter(taps, assistant_speaking)
                random_output = random_adapter(taps, assistant_speaking)
            frame_count = trained_output.yield_logits.shape[1]
            frame_mask = features.frame_mask.cpu()
            trained_accumulator.update(
                adapter_probabilities(trained_output), batch.targets, frame_mask
            )
            random_accumulator.update(
                adapter_probabilities(random_output), batch.targets, frame_mask
            )
            prior_accumulator.update(
                constant_probabilities(priors, batch.waveforms.shape[0], frame_count),
                batch.targets,
                frame_mask,
            )
            heuristic_accumulator.update(
                assistant_floor_probabilities(batch.assistant_speaking, priors, frame_count),
                batch.targets,
                frame_mask,
            )
    return (
        trained_accumulator.finalize(EvaluationModel.TRAINED_ADAPTER, loss_config),
        random_accumulator.finalize(EvaluationModel.RANDOM_ADAPTER, loss_config),
        prior_accumulator.finalize(EvaluationModel.CLASS_PRIOR, loss_config),
        heuristic_accumulator.finalize(EvaluationModel.ASSISTANT_FLOOR_HEURISTIC, loss_config),
    )


def evaluate_oracle_vad(
    samples: Iterable[MaterializedTrainingSample],
    priors: tuple[float, ...],
    loss_config: LossConfig,
) -> ModelEvaluation:
    accumulator = EvaluationAccumulator()
    for sample in samples:
        targets = _batched_targets(frame_targets_from_sample(sample))
        user_has_floor = torch.tensor(sample.p_user_has_floor, dtype=torch.float32).unsqueeze(0)
        frame_count = user_has_floor.shape[1]
        accumulator.update(
            oracle_vad_probabilities(user_has_floor, priors),
            targets,
            torch.ones((1, frame_count), dtype=torch.bool),
        )
    return accumulator.finalize(EvaluationModel.ORACLE_VAD, loss_config)


def compare_evaluations(
    reference: EvaluationReport,
    candidate: EvaluationReport,
) -> EvaluationComparison:
    if reference.hub_revision != candidate.hub_revision:
        raise ValueError("Evaluation reports use different Hub revisions.")
    if reference.validation_sample_count != candidate.validation_sample_count:
        raise ValueError("Evaluation reports use different validation inventories.")
    if reference.class_priors != candidate.class_priors:
        raise ValueError("Evaluation reports use different class-prior baselines.")
    if candidate.optimizer_step <= reference.optimizer_step:
        raise ValueError("Candidate evaluation must have a later optimizer step.")
    reference_loss = _trained_evaluation(reference).total_loss
    candidate_loss = _trained_evaluation(candidate).total_loss
    improvement = reference_loss - candidate_loss
    return EvaluationComparison(
        reference_optimizer_step=reference.optimizer_step,
        candidate_optimizer_step=candidate.optimizer_step,
        reference_total_loss=reference_loss,
        candidate_total_loss=candidate_loss,
        absolute_improvement=improvement,
        improved=improvement > 0.0,
    )


def _weighted_head_mean(heads: tuple[HeadMetrics, ...]) -> float:
    total_weight = sum(head.effective_support for head in heads)
    return sum(head.binary_cross_entropy * head.effective_support for head in heads) / total_weight


def _batched_targets(targets: FrameTargets) -> FrameTargets:
    if targets.yield_probability.ndim == 2:
        return targets
    if targets.yield_probability.ndim != 1:
        raise ValueError("Frame targets must have one or two dimensions.")
    return FrameTargets(
        yield_probability=targets.yield_probability.unsqueeze(0),
        primary_weight=targets.primary_weight.unsqueeze(0),
        primary_mask=targets.primary_mask.unsqueeze(0),
        event_targets=targets.event_targets.unsqueeze(0),
        event_mask=targets.event_mask.unsqueeze(0),
        future_activity=targets.future_activity.unsqueeze(0),
        future_activity_mask=targets.future_activity_mask.unsqueeze(0),
    )


def _align_assistant_speaking(values: Tensor, frame_count: int) -> Tensor:
    indices = (
        torch.linspace(0, values.shape[1] - 1, frame_count, device=values.device).round().long()
    )
    return values[:, indices]


def _trained_evaluation(report: EvaluationReport) -> ModelEvaluation:
    trained = tuple(
        model for model in report.models if model.model is EvaluationModel.TRAINED_ADAPTER
    )
    if len(trained) != 1:
        raise ValueError("Evaluation report must contain exactly one trained adapter result.")
    return trained[0]
