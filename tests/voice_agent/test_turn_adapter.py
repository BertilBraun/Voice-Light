from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.compute.voice.turn_adapter import (
    StreamingTurnAdapter,
    load_streaming_turn_adapter,
)
from app.shared.model_constants import NEMOTRON_ASR_MODEL_NAME, NEMOTRON_ASR_MODEL_REVISION
from app.training.turn_taking.config import AdapterConfig, TrainingConfig
from app.training.turn_taking.model import TurnTakingAdapter

MODEL_IDENTIFIER = NEMOTRON_ASR_MODEL_NAME
MODEL_REVISION = NEMOTRON_ASR_MODEL_REVISION


def test_checkpoint_loading_and_streaming_prediction(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.pt"
    training_config = _training_config()
    adapter = TurnTakingAdapter(training_config.adapter)
    torch.save(
        {
            "optimizer_step": 750,
            "training_config": training_config.model_dump(mode="json"),
            "adapter_state": adapter.state_dict(),
        },
        checkpoint,
    )

    loaded = load_streaming_turn_adapter(
        checkpoint_path=checkpoint,
        expected_model_identifier=MODEL_IDENTIFIER,
        expected_model_revision=MODEL_REVISION,
        expected_lookahead_tokens=1,
        device=torch.device("cpu"),
    )
    runtime = StreamingTurnAdapter(loaded)
    probabilities = runtime.predict(
        tuple(torch.zeros(1, 1, 8) for _ in range(2)),
        assistant_speaking=True,
    )

    assert loaded.optimizer_step == 750
    assert len(loaded.checkpoint_sha256) == 64
    assert 0.0 <= probabilities.turn_completion <= 1.0
    assert 0.0 <= probabilities.floor_take <= 1.0
    assert len(probabilities.future_activity) == 4


def test_streaming_adapter_reset_clears_recurrent_history(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.pt"
    training_config = _training_config()
    adapter = TurnTakingAdapter(training_config.adapter)
    torch.save(
        {
            "optimizer_step": 750,
            "training_config": training_config.model_dump(mode="json"),
            "adapter_state": adapter.state_dict(),
        },
        checkpoint,
    )
    runtime = StreamingTurnAdapter(
        load_streaming_turn_adapter(
            checkpoint_path=checkpoint,
            expected_model_identifier=MODEL_IDENTIFIER,
            expected_model_revision=MODEL_REVISION,
            expected_lookahead_tokens=1,
            device=torch.device("cpu"),
        )
    )
    features = tuple(torch.randn(1, 3, 8) for _ in range(2))

    first = runtime.predict(features, assistant_speaking=False)
    runtime.predict(features, assistant_speaking=True)
    runtime.reset()
    reset = runtime.predict(features, assistant_speaking=False)

    assert reset == first


def test_streaming_adapter_warmup_does_not_change_first_prediction(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.pt"
    training_config = _training_config()
    adapter = TurnTakingAdapter(training_config.adapter)
    torch.save(
        {
            "optimizer_step": 750,
            "training_config": training_config.model_dump(mode="json"),
            "adapter_state": adapter.state_dict(),
        },
        checkpoint,
    )
    runtime = StreamingTurnAdapter(
        load_streaming_turn_adapter(
            checkpoint_path=checkpoint,
            expected_model_identifier=MODEL_IDENTIFIER,
            expected_model_revision=MODEL_REVISION,
            expected_lookahead_tokens=1,
            device=torch.device("cpu"),
        )
    )
    features = tuple(torch.randn(1, 3, 8) for _ in range(2))

    expected = runtime.predict(features, assistant_speaking=False)
    runtime.reset()
    runtime.warm_up()
    actual = runtime.predict(features, assistant_speaking=False)

    assert actual == expected


@pytest.mark.parametrize(
    ("model_identifier", "model_revision", "lookahead_tokens", "expected_message"),
    [
        ("wrong/model", MODEL_REVISION, 1, "different encoder model"),
        (MODEL_IDENTIFIER, "wrong-revision", 1, "different encoder revision"),
        (MODEL_IDENTIFIER, MODEL_REVISION, 2, "different lookahead"),
    ],
)
def test_checkpoint_loading_rejects_incompatible_encoder_configuration(
    tmp_path: Path,
    model_identifier: str,
    model_revision: str,
    lookahead_tokens: int,
    expected_message: str,
) -> None:
    checkpoint = tmp_path / "adapter.pt"
    training_config = _training_config()
    adapter = TurnTakingAdapter(training_config.adapter)
    torch.save(
        {
            "optimizer_step": 1,
            "training_config": training_config.model_dump(mode="json"),
            "adapter_state": adapter.state_dict(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match=expected_message):
        load_streaming_turn_adapter(
            checkpoint_path=checkpoint,
            expected_model_identifier=model_identifier,
            expected_model_revision=model_revision,
            expected_lookahead_tokens=lookahead_tokens,
            device=torch.device("cpu"),
        )


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        model_identifier=MODEL_IDENTIFIER,
        model_revision=MODEL_REVISION,
        lookahead_tokens=1,
        adapter=AdapterConfig(
            feature_dimension=8,
            tap_layer_indices=(1, 2),
            tap_projection_dimension=4,
            fused_dimension=6,
            recurrent_dimension=5,
            recurrent_layers=1,
            dropout=0.0,
        ),
    )
