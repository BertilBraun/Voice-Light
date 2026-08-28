from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from app.training.turn_taking.data import TrainingItem


@dataclass(frozen=True)
class WeightedTrainingSource:
    dataset: Dataset[TrainingItem]
    sampling_weights: Tensor
    fraction: float

    def __post_init__(self) -> None:
        if len(self.dataset) != len(self.sampling_weights):
            raise ValueError("Sampling weights must match the source dataset length.")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("Training source fraction must be between zero and one.")
        if torch.any(self.sampling_weights < 0.0):
            raise ValueError("Sampling weights must be nonnegative.")
        if not torch.isclose(self.sampling_weights.sum(), torch.tensor(1.0, dtype=torch.double)):
            raise ValueError("Training source sampling weights must sum to one.")


class WeightedTrainingDatasetCollection(Dataset[TrainingItem]):
    def __init__(self, sources: tuple[WeightedTrainingSource, ...]) -> None:
        if len(sources) < 2:
            raise ValueError("Weighted training collection requires at least two sources.")
        if abs(sum(source.fraction for source in sources) - 1.0) > 1e-9:
            raise ValueError("Training source fractions must sum to one.")
        self.sources = sources

    def __len__(self) -> int:
        return sum(len(source.dataset) for source in self.sources)

    def __getitem__(self, index: int) -> TrainingItem:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        for source in self.sources:
            if index < len(source.dataset):
                return source.dataset[index]
            index -= len(source.dataset)
        raise AssertionError("Weighted collection index resolution failed.")

    def sampling_weights(self) -> Tensor:
        return torch.cat(
            tuple(source.sampling_weights * source.fraction for source in self.sources)
        )
