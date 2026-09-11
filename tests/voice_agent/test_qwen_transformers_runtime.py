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
    GenerationFlowControl,
    GenerationFlowControlLogitsProcessor,
    QwenTransformersRuntime,
)


def test_cancellation_logits_processor_forces_eos_after_cancellation() -> None:
    flow_control = GenerationFlowControl()
    processor = GenerationFlowControlLogitsProcessor(flow_control, eos_token_id=2)
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    assert torch.equal(processor(torch.tensor([[1]]), scores), scores)

    flow_control.cancel()
    cancelled_scores = processor(torch.tensor([[1]]), scores)

    assert cancelled_scores[0, 2] == 0.0
    assert torch.isneginf(cancelled_scores[0, [0, 1, 3]]).all()


def test_transformers_flow_control_blocks_between_tokens_until_resumed() -> None:
    flow_control = GenerationFlowControl()
    processor = GenerationFlowControlLogitsProcessor(flow_control, eos_token_id=2)
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    processing_started = threading.Event()
    processing_completed = threading.Event()

    def process_next_token() -> None:
        processing_started.set()
        processor(torch.tensor([[1]]), scores)
        processing_completed.set()

    flow_control.pause()
    generation_thread = threading.Thread(target=process_next_token)
    generation_thread.start()
    assert processing_started.wait(timeout=1)
    assert not processing_completed.wait(timeout=0.05)

    flow_control.resume()
    generation_thread.join(timeout=1)

    assert processing_completed.is_set()


def test_transformers_cancellation_releases_paused_generation_and_forces_eos() -> None:
    flow_control = GenerationFlowControl()
    processor = GenerationFlowControlLogitsProcessor(flow_control, eos_token_id=2)
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    processed_scores: list[torch.FloatTensor] = []

    def process_next_token() -> None:
        processed_scores.append(processor(torch.tensor([[1]]), scores))

    flow_control.pause()
    generation_thread = threading.Thread(target=process_next_token)
    generation_thread.start()
    flow_control.cancel()
    generation_thread.join(timeout=1)

    assert not generation_thread.is_alive()
    assert processed_scores[0][0, 2] == 0.0


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
