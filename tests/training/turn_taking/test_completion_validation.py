from __future__ import annotations

import pytest
import torch
from torch import Tensor

from app.training.turn_taking.backbone import BackboneFeatures
from app.training.turn_taking.completion_validation import evaluate_completion_boundaries
from app.training.turn_taking.data import FrameTargets, TrainingBatch
from app.training.turn_taking.model import AdapterOutput


class _Backbone:
    def extract(self, waveforms: Tensor, waveform_lengths: Tensor) -> BackboneFeatures:
        del waveform_lengths
        batch_size = waveforms.shape[0]
        return BackboneFeatures(
            taps=(torch.zeros((batch_size, 2, 3)),),
            frame_mask=torch.ones((batch_size, 2), dtype=torch.bool),
        )


class _Adapter:
    def to(self, device: torch.device) -> _Adapter:
        del device
        return self

    def eval(self) -> _Adapter:
        return self

    def train(self) -> _Adapter:
        return self

    def __call__(self, taps: tuple[Tensor, ...], assistant_speaking: Tensor) -> AdapterOutput:
        del taps, assistant_speaking
        probabilities = torch.tensor(((0.1, 0.2), (0.8, 0.9)))
        event_logits = torch.zeros((2, 2, 5))
        event_logits[..., 0] = torch.logit(probabilities)
        return AdapterOutput(
            yield_logits=torch.zeros((2, 2)),
            event_logits=event_logits,
            future_activity_logits=torch.zeros((2, 2, 4)),
            recurrent_state=torch.zeros((1, 2, 1)),
        )


def test_completion_validation_scores_one_boundary_per_item() -> None:
    event_targets = torch.zeros((2, 2, 5))
    event_targets[1, 1, 0] = 1.0
    event_mask = torch.zeros((2, 2, 5), dtype=torch.bool)
    event_mask[0, 0, 0] = True
    event_mask[1, 1, 0] = True
    batch = TrainingBatch(
        sample_ids=("hold", "eot"),
        waveforms=torch.zeros((2, 16)),
        waveform_lengths=torch.tensor((16, 16)),
        assistant_speaking=torch.zeros((2, 2)),
        targets=FrameTargets(
            yield_probability=torch.zeros((2, 2)),
            primary_weight=torch.ones((2, 2)),
            primary_mask=torch.ones((2, 2), dtype=torch.bool),
            event_targets=event_targets,
            event_mask=event_mask,
            future_activity=torch.zeros((2, 2, 4)),
            future_activity_mask=torch.zeros((2, 2, 4), dtype=torch.bool),
        ),
    )

    metrics = evaluate_completion_boundaries(
        backbone=_Backbone(),
        adapter=_Adapter(),
        batches=(batch,),
        device=torch.device("cpu"),
    )

    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.binary_cross_entropy == pytest.approx(-torch.log(torch.tensor(0.9)).item())
    assert metrics.support == 2
