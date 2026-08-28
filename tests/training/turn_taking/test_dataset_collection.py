from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from app.training.turn_taking.data import FrameTargets, TrainingItem
from app.training.turn_taking.dataset_collection import (
    WeightedTrainingDatasetCollection,
    WeightedTrainingSource,
)


class ItemDataset(Dataset[TrainingItem]):
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> TrainingItem:
        targets = FrameTargets(
            yield_probability=torch.zeros(1),
            primary_weight=torch.ones(1),
            primary_mask=torch.ones(1, dtype=torch.bool),
            event_targets=torch.zeros((1, 5)),
            event_mask=torch.zeros((1, 5), dtype=torch.bool),
            future_activity=torch.zeros((1, 4)),
            future_activity_mask=torch.zeros((1, 4), dtype=torch.bool),
        )
        return TrainingItem(
            sample_id=self.names[index],
            waveform=torch.zeros(1),
            assistant_speaking=torch.zeros(1),
            targets=targets,
        )


def test_weighted_collection_preserves_source_fractions() -> None:
    human = ItemDataset(("human-a", "human-b"))
    synthetic = ItemDataset(("synthetic-a",))
    collection = WeightedTrainingDatasetCollection(
        (
            WeightedTrainingSource(human, torch.tensor((0.5, 0.5), dtype=torch.double), 0.85),
            WeightedTrainingSource(synthetic, torch.tensor((1.0,), dtype=torch.double), 0.15),
        )
    )

    assert tuple(collection[index].sample_id for index in range(3)) == (
        "human-a",
        "human-b",
        "synthetic-a",
    )
    assert collection.sampling_weights().tolist() == pytest.approx((0.425, 0.425, 0.15))


def test_weighted_source_rejects_unormalized_weights() -> None:
    with pytest.raises(ValueError, match="must sum to one"):
        WeightedTrainingSource(
            ItemDataset(("human-a", "human-b")),
            torch.tensor((0.5, 0.25), dtype=torch.double),
            0.8,
        )
