from pathlib import Path

import torch
from torch import Tensor

from app.training.turn_taking.backbone import BackboneFeatures
from app.training.turn_taking.config import AdapterConfig, LossConfig, TrainingConfig
from app.training.turn_taking.data import FrameTargets, TrainingBatch
from app.training.turn_taking.loss import compute_loss
from app.training.turn_taking.model import TurnTakingAdapter
from app.training.turn_taking.trainer import train


class FakeBackbone:
    def __init__(self, tap_count: int, feature_dimension: int, frame_count: int) -> None:
        self.tap_count = tap_count
        self.feature_dimension = feature_dimension
        self.frame_count = frame_count

    def extract(self, waveforms: Tensor, waveform_lengths: Tensor) -> BackboneFeatures:
        del waveform_lengths
        batch_size = waveforms.shape[0]
        taps = tuple(
            torch.randn(batch_size, self.frame_count, self.feature_dimension)
            for _ in range(self.tap_count)
        )
        return BackboneFeatures(
            taps=taps,
            frame_mask=torch.ones(batch_size, self.frame_count, dtype=torch.bool),
        )


def test_adapter_and_multitask_loss_backpropagate() -> None:
    config = _adapter_config()
    adapter = TurnTakingAdapter(config)
    taps = tuple(torch.randn(2, 8, 16) for _ in config.tap_layer_indices)

    output = adapter(taps, torch.ones(2, 8, dtype=torch.bool))
    loss = compute_loss(
        output,
        _targets(batch_size=2, frame_count=8),
        torch.ones(2, 8).bool(),
        LossConfig(),
    )
    loss.total.backward()

    assert output.yield_logits.shape == (2, 8)
    assert output.event_logits.shape == (2, 8, 5)
    assert output.future_activity_logits.shape == (2, 8, 4)
    assert output.recurrent_state.shape == (2, 2, 12)
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_default_adapter_stays_below_parameter_budget() -> None:
    config = AdapterConfig()
    adapter = TurnTakingAdapter(config)

    trainable_parameter_count = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )

    assert len(config.tap_layer_indices) == 4
    assert config.recurrent_dimension == 64
    assert trainable_parameter_count < 200_000


def test_training_loop_saves_checkpoint(tmp_path: Path) -> None:
    adapter_config = _adapter_config()
    config = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=2,
        warmup_steps=1,
        adapter=adapter_config,
    )
    batch = TrainingBatch(
        sample_ids=("one", "two"),
        waveforms=torch.randn(2, 320),
        waveform_lengths=torch.tensor([320, 320]),
        assistant_speaking=torch.randint(0, 2, (2, 8), dtype=torch.bool),
        targets=_targets(batch_size=2, frame_count=8),
    )
    path = tmp_path / "adapter.pt"

    result = train(
        backbone=FakeBackbone(3, 16, 8),
        adapter=TurnTakingAdapter(adapter_config),
        batches=[batch],
        config=config,
        checkpoint_path=path,
        device=torch.device("cpu"),
    )

    assert result.optimizer_steps == 2
    assert path.exists()


def _adapter_config() -> AdapterConfig:
    return AdapterConfig(
        feature_dimension=16,
        tap_layer_indices=(1, 2, 3),
        tap_projection_dimension=8,
        fused_dimension=12,
        recurrent_dimension=12,
        recurrent_layers=2,
    )


def _targets(batch_size: int, frame_count: int) -> FrameTargets:
    return FrameTargets(
        yield_probability=torch.rand(batch_size, frame_count),
        primary_weight=torch.ones(batch_size, frame_count),
        primary_mask=torch.ones(batch_size, frame_count, dtype=torch.bool),
        event_distribution=torch.softmax(torch.rand(batch_size, frame_count, 5), dim=-1),
        event_weight=torch.ones(batch_size, frame_count),
        event_mask=torch.ones(batch_size, frame_count, dtype=torch.bool),
        future_activity=torch.rand(batch_size, frame_count, 4),
        future_activity_mask=torch.ones(batch_size, frame_count, 4, dtype=torch.bool),
    )
