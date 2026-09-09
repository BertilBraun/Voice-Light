from __future__ import annotations

import threading

import pytest
import torch

from app.compute.voice.qwen_config import (
    QwenAdapterConfiguration,
    QwenBackend,
    QwenModelConfiguration,
)
from app.compute.voice.qwen_transformers_runtime import (
    CancellationLogitsProcessor,
    QwenTransformersRuntime,
)


def test_cancellation_logits_processor_forces_eos_after_cancellation() -> None:
    cancellation_event = threading.Event()
    processor = CancellationLogitsProcessor(cancellation_event, eos_token_id=2)
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    assert torch.equal(processor(torch.tensor([[1]]), scores), scores)

    cancellation_event.set()
    cancelled_scores = processor(torch.tensor([[1]]), scores)

    assert cancelled_scores[0, 2] == 0.0
    assert torch.isneginf(cancelled_scores[0, [0, 1, 3]]).all()


def test_transformers_configuration_rejects_dynamic_adapter() -> None:
    configuration = QwenModelConfiguration(
        model_name="owner/model",
        model_revision="revision",
        adapter=QwenAdapterConfiguration("owner/adapter", "revision"),
        gpu_memory_utilization=0.5,
        maximum_model_length=1_024,
        enforce_eager=True,
        backend=QwenBackend.TRANSFORMERS,
    )

    with pytest.raises(ValueError, match="merged model checkpoint"):
        QwenTransformersRuntime(configuration)
