from __future__ import annotations

import pytest
import torch

from app.training.turn_taking.config import AdapterConfig
from app.training.turn_taking.model import TurnTakingAdapter


def test_incremental_adapter_matches_whole_sequence_inference() -> None:
    torch.manual_seed(7)
    adapter = TurnTakingAdapter(_adapter_config()).eval()
    taps = tuple(torch.randn(1, 13, 8) for _ in range(2))
    assistant_speaking = torch.randint(0, 2, (1, 13), dtype=torch.float32)

    with torch.no_grad():
        whole = adapter(taps, assistant_speaking)
        state = None
        outputs = []
        for start, end in ((0, 1), (1, 4), (4, 9), (9, 13)):
            output, state = adapter.forward_incremental(
                tuple(tap[:, start:end] for tap in taps),
                assistant_speaking[:, start:end],
                state,
            )
            outputs.append(output)

    torch.testing.assert_close(
        torch.cat([output.yield_logits for output in outputs], dim=1),
        whole.yield_logits,
        rtol=1e-5,
        atol=1e-7,
    )
    torch.testing.assert_close(
        torch.cat([output.future_activity_logits for output in outputs], dim=1),
        whole.future_activity_logits,
        rtol=1e-5,
        atol=1e-7,
    )
    torch.testing.assert_close(
        torch.cat([output.event_logits for output in outputs], dim=1),
        whole.event_logits,
        rtol=1e-5,
        atol=1e-7,
    )
    assert state is not None
    torch.testing.assert_close(state.recurrent_state, whole.recurrent_state)


def test_incremental_adapter_state_is_reset_by_starting_without_state() -> None:
    torch.manual_seed(11)
    adapter = TurnTakingAdapter(_adapter_config()).eval()
    taps = tuple(torch.randn(1, 5, 8) for _ in range(2))
    assistant_speaking = torch.zeros(1, 5)

    with torch.no_grad():
        first, _ = adapter.forward_incremental(taps, assistant_speaking)
        reset, _ = adapter.forward_incremental(taps, assistant_speaking)

    torch.testing.assert_close(reset.yield_logits, first.yield_logits)
    torch.testing.assert_close(reset.future_activity_logits, first.future_activity_logits)
    torch.testing.assert_close(reset.event_logits, first.event_logits)


def test_incremental_adapter_rejects_state_from_another_batch_shape() -> None:
    adapter = TurnTakingAdapter(_adapter_config()).eval()
    taps = tuple(torch.randn(1, 2, 8) for _ in range(2))
    _, state = adapter.forward_incremental(taps, torch.zeros(1, 2))

    with pytest.raises(ValueError, match="history must match"):
        adapter.forward_incremental(
            tuple(torch.randn(2, 1, 8) for _ in range(2)),
            torch.zeros(2, 1),
            state,
        )


def _adapter_config() -> AdapterConfig:
    return AdapterConfig(
        feature_dimension=8,
        tap_layer_indices=(1, 2),
        tap_projection_dimension=4,
        fused_dimension=6,
        recurrent_dimension=5,
        recurrent_layers=1,
        dropout=0.0,
    )
