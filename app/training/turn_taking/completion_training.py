from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor
from torch.utils.data import Dataset

from app.local.training_corpus.export import MaterializedTrainingSample
from app.training.turn_taking.benchmark_models import TurnCompletionInventory
from app.training.turn_taking.config import TurnCompletionObjectiveConfig
from app.training.turn_taking.data import FrameTargets, TrainingItem

FRAME_SECONDS = 0.08


class CompletionClass(StrEnum):
    HOLD = "hold"
    EOT = "eot"


@dataclass(frozen=True)
class CompletionBoundaryIndex:
    sample_index: int
    frame_index: int
    completion_class: CompletionClass


class CompletionBoundaryDataset(Dataset[TrainingItem]):
    def __init__(
        self,
        source: Dataset[TrainingItem],
        boundaries: tuple[CompletionBoundaryIndex, ...],
    ) -> None:
        if not boundaries:
            raise ValueError("Completion-boundary training requires at least one clean boundary.")
        self.source = source
        self.boundaries = boundaries

    def __len__(self) -> int:
        return len(self.boundaries)

    def __getitem__(self, index: int) -> TrainingItem:
        boundary = self.boundaries[index]
        item = self.source[boundary.sample_index]
        event_mask = item.targets.event_mask.clone()
        event_mask[:, 0] = False
        event_mask[boundary.frame_index, 0] = True
        targets = FrameTargets(
            yield_probability=item.targets.yield_probability,
            primary_weight=item.targets.primary_weight,
            primary_mask=item.targets.primary_mask,
            event_targets=item.targets.event_targets,
            event_mask=event_mask,
            future_activity=item.targets.future_activity,
            future_activity_mask=item.targets.future_activity_mask,
        )
        return TrainingItem(
            sample_id=f"{item.sample_id}:{boundary.frame_index}",
            waveform=item.waveform,
            assistant_speaking=item.assistant_speaking,
            targets=targets,
        )


def build_completion_boundaries(
    samples: tuple[MaterializedTrainingSample, ...],
    config: TurnCompletionObjectiveConfig,
) -> tuple[CompletionBoundaryIndex, ...]:
    boundaries: list[CompletionBoundaryIndex] = []
    for sample_index, sample in enumerate(samples):
        for frame_index, completion_probability in enumerate(sample.turn_completion):
            if completion_probability < 0.0:
                continue
            continuation_probability = sample.continuation_pause[frame_index]
            completion_class = _completion_class(
                completion_probability,
                continuation_probability,
                config,
            )
            if completion_class is None:
                continue
            if sample.assistant_has_floor[frame_index] >= 0.5:
                continue
            if (
                completion_class is CompletionClass.EOT
                and sample.p_user_has_floor[frame_index] >= 0.5
            ):
                continue
            boundaries.append(
                CompletionBoundaryIndex(
                    sample_index=sample_index,
                    frame_index=frame_index,
                    completion_class=completion_class,
                )
            )
    return tuple(boundaries)


def filter_completion_boundaries_by_dataset(
    samples: tuple[MaterializedTrainingSample, ...],
    boundaries: tuple[CompletionBoundaryIndex, ...],
    dataset_names: frozenset[str],
) -> tuple[CompletionBoundaryIndex, ...]:
    if not dataset_names:
        raise ValueError("At least one human training dataset name is required.")
    filtered = tuple(
        boundary
        for boundary in boundaries
        if samples[boundary.sample_index].dataset_name in dataset_names
    )
    if not filtered:
        raise ValueError(
            f"No completion boundaries matched human training datasets {sorted(dataset_names)}."
        )
    return filtered


def build_inventory_completion_boundaries(
    samples: tuple[MaterializedTrainingSample, ...],
    inventory: TurnCompletionInventory,
    config: TurnCompletionObjectiveConfig,
) -> tuple[CompletionBoundaryIndex, ...]:
    samples_by_window_id = {
        sample.window_id: (sample_index, sample) for sample_index, sample in enumerate(samples)
    }
    boundaries: list[CompletionBoundaryIndex] = []
    for candidate in inventory.candidates:
        completion_probability = candidate.target_points[0].completion_probability
        continuation_probability = candidate.continuation_probability
        completion_class = _completion_class(
            completion_probability,
            continuation_probability if continuation_probability is not None else -1.0,
            config,
        )
        if completion_class is None:
            continue
        matching: list[tuple[int, int]] = []
        for window_id in candidate.source_window_ids:
            sample_entry = samples_by_window_id.get(window_id)
            if sample_entry is None:
                continue
            sample_index, sample = sample_entry
            frame_index = round((candidate.anchor_seconds - sample.start_seconds) / FRAME_SECONDS)
            if not 0 <= frame_index < len(sample.turn_completion):
                continue
            if math.isclose(
                sample.turn_completion[frame_index],
                completion_probability,
                abs_tol=1e-9,
            ):
                matching.append((sample_index, frame_index))
        if len(matching) != 1:
            raise ValueError(
                f"Completion candidate {candidate.candidate_id} maps to {len(matching)} "
                "supervised windows; expected exactly one."
            )
        sample_index, frame_index = matching[0]
        boundaries.append(
            CompletionBoundaryIndex(
                sample_index=sample_index,
                frame_index=frame_index,
                completion_class=completion_class,
            )
        )
    return tuple(boundaries)


def _completion_class(
    completion_probability: float,
    continuation_probability: float,
    config: TurnCompletionObjectiveConfig,
) -> CompletionClass | None:
    if (
        completion_probability <= config.hold_completion_maximum
        and continuation_probability >= config.hold_continuation_minimum
    ):
        return CompletionClass.HOLD
    if completion_probability >= config.eot_completion_minimum and not (
        continuation_probability >= config.conflicting_continuation_minimum
    ):
        return CompletionClass.EOT
    return None


def balanced_completion_weights(
    boundaries: tuple[CompletionBoundaryIndex, ...],
) -> Tensor:
    hold_count = sum(boundary.completion_class is CompletionClass.HOLD for boundary in boundaries)
    eot_count = sum(boundary.completion_class is CompletionClass.EOT for boundary in boundaries)
    if hold_count == 0 or eot_count == 0:
        raise ValueError("Completion-boundary training requires both HOLD and EOT examples.")
    return torch.tensor(
        [
            0.5 / hold_count
            if boundary.completion_class is CompletionClass.HOLD
            else 0.5 / eot_count
            for boundary in boundaries
        ],
        dtype=torch.double,
    )
