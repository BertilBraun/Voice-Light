from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import vllm
from huggingface_hub import model_info
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from app.local.synthetic_generation.conversation_prompts import (
    ConversationGenerationBrief,
    EnglishConversationContentDraft,
    conversation_generation_instruction,
    validate_conversation_prompt_set_id,
)
from app.local.synthetic_generation.generate_conversation_prompts import (
    run_conversation_prompt_generation,
)


class VllmConversationTextGenerator:
    def __init__(
        self,
        model: LLM,
        tokenizer: PreTrainedTokenizerBase,
        maximum_new_tokens: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._maximum_new_tokens = maximum_new_tokens

    def generate_batch(
        self,
        briefs: tuple[ConversationGenerationBrief, ...],
    ) -> tuple[str, ...]:
        prompts = tuple(
            self._prompt(conversation_generation_instruction(brief)) for brief in briefs
        )
        sampling_parameters = tuple(
            SamplingParams(
                n=1,
                temperature=0.8,
                top_p=0.92,
                max_tokens=self._maximum_new_tokens,
                seed=brief.seed,
                structured_outputs=StructuredOutputsParams(
                    json=EnglishConversationContentDraft.model_json_schema(),
                    disable_additional_properties=True,
                ),
            )
            for brief in briefs
        )
        outputs = self._model.generate(
            prompts,
            sampling_params=sampling_parameters,
            use_tqdm=False,
        )
        return tuple(output.outputs[0].text for output in outputs)

    def _prompt(self, instruction: str) -> str:
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(prompt, str):
            raise ValueError("Tokenizer did not return one rendered chat prompt.")
        return prompt


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    set_id = validate_conversation_prompt_set_id(parsed.set_id)
    revision = model_info(parsed.model).sha
    if revision is None:
        raise ValueError(f"Hugging Face did not return a revision for {parsed.model}.")
    tokenizer = AutoTokenizer.from_pretrained(parsed.model, revision=revision)
    model = LLM(
        model=parsed.model,
        revision=revision,
        dtype="bfloat16",
        gpu_memory_utilization=parsed.gpu_memory_utilization,
        max_model_len=parsed.maximum_model_length,
        max_num_seqs=parsed.generation_batch_size,
        enable_prefix_caching=True,
    )
    generator = VllmConversationTextGenerator(
        model=model,
        tokenizer=tokenizer,
        maximum_new_tokens=parsed.maximum_new_tokens,
    )
    run_conversation_prompt_generation(
        output_path=parsed.output,
        set_id=set_id,
        count=parsed.count,
        target_conversation_hours=parsed.target_conversation_hours,
        seed=parsed.seed,
        model_id=parsed.model,
        model_revision=revision,
        runtime_version=vllm.__version__,
        generation_batch_size=parsed.generation_batch_size,
        generate_batch=generator.generate_batch,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate English conversation plans with offline continuous vLLM batching."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--count", type=int, choices=range(5, 2_001), default=10)
    parser.add_argument("--target-conversation-hours", type=float)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--maximum-model-length", type=int, default=8_192)
    parser.add_argument("--maximum-new-tokens", type=int, default=1_100)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
