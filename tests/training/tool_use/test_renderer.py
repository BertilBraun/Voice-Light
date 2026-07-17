from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from app.training.tool_use.generation import (
    GeneratedRecord,
    RolloutGenerationConfig,
    generate_record,
)
from app.training.tool_use.renderer import (
    ChatTemplateEncoding,
    QwenChatMessage,
    QwenToolSpecification,
    qwen_message,
    render_qwen_training_example,
)
from app.training.tool_use.schema import AssistantMessage, ToolResultMessage
from tests.training.tool_use.test_generation import (
    ScriptedGenerator,
    scripted_two_call_values,
    two_call_scenario,
)

QWEN_MODEL_IDENTIFIER = "Qwen/Qwen3-1.7B"
QWEN_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


class PrefixStableTokenizer:
    chat_template = (
        '{%- elif message.role == "assistant" %}\n'
        "        {{- '<|im_end|>\\n' }}\n"
        '    {%- elif message.role == "tool" %}'
    )

    def apply_chat_template(
        self,
        conversation: Sequence[QwenChatMessage],
        *,
        tools: Sequence[QwenToolSpecification],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        chat_template: str | None = None,
        return_dict: bool = True,
        return_assistant_tokens_mask: bool = False,
    ) -> ChatTemplateEncoding:
        assert tokenize is True
        assert add_generation_prompt is False
        assert enable_thinking is False
        assert len(tools) == 3
        role_tokens = {
            "system": 11,
            "user": 12,
            "assistant": 13,
            "tool": 14,
        }
        input_ids = [99, *(role_tokens[message["role"]] for message in conversation)]
        assert return_dict is True
        if return_assistant_tokens_mask:
            assert chat_template is not None
            assert return_assistant_tokens_mask is True
            return ChatTemplateEncoding(
                input_ids=input_ids,
                assistant_masks=[
                    0,
                    *(int(message["role"] == "assistant") for message in conversation),
                ],
            )
        return ChatTemplateEncoding(input_ids=input_ids)


def test_qwen_renderer_masks_every_non_assistant_message() -> None:
    async def exercise() -> GeneratedRecord:
        return await generate_record(
            scenario=two_call_scenario(),
            generator=ScriptedGenerator(scripted_two_call_values()),
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
            ),
        )

    generated = asyncio.run(exercise())
    example = render_qwen_training_example(PrefixStableTokenizer(), generated.record)

    expected_mask = (False,) + tuple(
        isinstance(message, AssistantMessage) for message in generated.record.messages
    )
    assert example.assistant_loss_mask == expected_mask
    assert all(
        label == token_id if target else label == -100
        for label, token_id, target in zip(
            example.labels,
            example.input_ids,
            example.assistant_loss_mask,
            strict=True,
        )
    )


def test_qwen_renderer_keeps_spoken_content_calls_and_string_results_separate() -> None:
    async def exercise() -> GeneratedRecord:
        return await generate_record(
            scenario=two_call_scenario(),
            generator=ScriptedGenerator(scripted_two_call_values()),
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
            ),
        )

    generated = asyncio.run(exercise())
    assistant_call = next(
        message
        for message in generated.record.messages
        if isinstance(message, AssistantMessage) and message.tool_calls
    )
    tool_result = next(
        message for message in generated.record.messages if isinstance(message, ToolResultMessage)
    )

    rendered_assistant = qwen_message(assistant_call)
    rendered_result = qwen_message(tool_result)

    assert rendered_assistant["content"] == "I'll check the current price."
    assert rendered_assistant["tool_calls"][0]["function"]["name"] == "search"
    assert rendered_assistant["tool_calls"][0]["function"]["arguments"] == (
        '{"query":"current example ticket price"}'
    )
    assert rendered_result["role"] == "tool"
    assert rendered_result["content"] == "The current ticket price is 24 euros."


@pytest.mark.integration
def test_pinned_qwen_template_preserves_tokens_and_returns_assistant_mask() -> None:
    async def exercise() -> GeneratedRecord:
        return await generate_record(
            scenario=two_call_scenario(),
            generator=ScriptedGenerator(scripted_two_call_values()),
            config=RolloutGenerationConfig(
                model_identifier="Qwen/Qwen3.6-27B-FP8",
                model_revision="test-revision",
                quantization="fp8",
            ),
        )

    cache_path = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Qwen--Qwen3-1.7B"
        / "snapshots"
        / QWEN_MODEL_REVISION
    )
    if not cache_path.exists():
        pytest.skip("Pinned Qwen tokenizer is not present in the local Hugging Face cache.")
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_IDENTIFIER,
        revision=QWEN_MODEL_REVISION,
        local_files_only=True,
    )
    generated = asyncio.run(exercise())

    example = render_qwen_training_example(tokenizer, generated.record)

    assert example.input_ids
    assert any(example.assistant_loss_mask)
    assert any(not target for target in example.assistant_loss_mask)
