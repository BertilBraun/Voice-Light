from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import transformers
from huggingface_hub import model_info
from pydantic import ValidationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.local.synthetic_generation.conversation_prompts import (
    CompletionUserPrompt,
    ConversationGenerationBrief,
    ConversationPromptGeneratorProvenance,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    HoldUserPrompt,
    InterruptionFloorClaimUserPrompt,
    NonFloorFeedbackUserPrompt,
    ResponseFloorClaimUserPrompt,
    UserPrompt,
    conversation_generation_instruction,
    representative_conversation_briefs,
    validate_conversation_prompt_set_id,
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    set_id = validate_conversation_prompt_set_id(parsed.set_id)
    revision = model_info(parsed.model).sha
    if revision is None:
        raise ValueError(f"Hugging Face did not return a revision for {parsed.model}.")
    tokenizer = AutoTokenizer.from_pretrained(parsed.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        parsed.model,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    provenance = ConversationPromptGeneratorProvenance(
        model_id=parsed.model,
        model_revision=revision,
        runtime_version=transformers.__version__,
        seed=parsed.seed,
        requested_plan_count=parsed.count,
    )

    def checkpoint(plans: tuple[EnglishConversationPromptPlan, ...]) -> None:
        if not plans:
            return
        prompt_set = EnglishConversationPromptSet(
            set_id=set_id,
            provenance=provenance,
            plans=plans,
        )
        _write_atomically(parsed.output.with_suffix(f"{parsed.output.suffix}.partial"), prompt_set)
        print(f"Checkpointed {len(plans)}/{parsed.count} conversation plans", flush=True)

    prompt_set = generate_conversation_prompt_set(
        model=model,
        tokenizer=tokenizer,
        set_id=set_id,
        provenance=provenance,
        on_progress=checkpoint,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(parsed.output, prompt_set)
    parsed.output.with_suffix(f"{parsed.output.suffix}.partial").unlink(missing_ok=True)
    print(f"Wrote {len(prompt_set.plans)} conversation plans to {parsed.output}", flush=True)


def generate_conversation_prompt_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    set_id: str,
    provenance: ConversationPromptGeneratorProvenance,
    on_progress: Callable[[tuple[EnglishConversationPromptPlan, ...]], None],
) -> EnglishConversationPromptSet:
    briefs = representative_conversation_briefs(
        count=provenance.requested_plan_count,
        seed=provenance.seed,
    )
    plans = []
    normalized_user_texts: set[str] = set()
    for brief in briefs:
        final_error: ValueError | None = None
        for attempt in range(4):
            try:
                plan = _generate_conversation_plan(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=conversation_generation_instruction(brief),
                    seed=brief.seed + attempt,
                )
                _validate_plan_against_brief(plan, brief)
                plan_texts = tuple(
                    _normalized_user_text(text)
                    for prompt in plan.user_prompts
                    for text in _prompt_texts(prompt)
                )
                if normalized_user_texts.intersection(plan_texts):
                    raise ValueError("Generated conversation repeats user text from another plan.")
            except ValueError as error:
                final_error = error
                print(
                    f"Discarding invalid realization for {brief.plan_id}: {error}",
                    flush=True,
                )
                continue
            plans.append(plan)
            normalized_user_texts.update(plan_texts)
            on_progress(tuple(plans))
            break
        else:
            assert final_error is not None
            raise ValueError(
                f"Conversation generation failed four times for {brief.plan_id}."
            ) from final_error
    return EnglishConversationPromptSet(
        set_id=set_id,
        provenance=provenance,
        plans=tuple(plans),
    )


def _generate_conversation_plan(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    seed: int,
) -> EnglishConversationPromptPlan:
    messages = [{"role": "user", "content": instruction}]
    final_error: ValidationError | None = None
    for repair_index in range(3):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        torch.manual_seed(seed + repair_index)
        output = model.generate(
            **inputs,
            max_new_tokens=2400,
            do_sample=True,
            temperature=0.8 if repair_index == 0 else 0.4,
            top_p=0.92,
        )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        try:
            return EnglishConversationPromptPlan.model_validate_json(_json_object(generated))
        except ValidationError as error:
            final_error = error
            messages.extend(
                (
                    {"role": "assistant", "content": generated},
                    {"role": "user", "content": _repair_instruction(error)},
                )
            )
    assert final_error is not None
    raise ValueError(f"Prompt model returned an invalid conversation plan: {final_error}")


def _repair_instruction(error: ValidationError) -> str:
    return f"""The JSON object failed typed validation.
Return only a corrected complete JSON object. Preserve the requested plan identity, topic, and
conversation meaning. Do not explain the correction.

Assign every assistant turn and user prompt a distinct sequence_index. Renumber all elements with
consecutive integers 0, 1, 2, and so on in chronological conversational order. A condition such as
response_floor_claim is never a speech_act; use only a speech_act value allowed by the schema.

Validation errors:
{error}
"""


def _validate_plan_against_brief(
    plan: EnglishConversationPromptPlan,
    brief: ConversationGenerationBrief,
) -> None:
    if plan.plan_id != brief.plan_id or plan.seed != brief.seed:
        raise ValueError("Generated conversation changed its fixed plan identity.")
    if plan.domain is not brief.domain:
        raise ValueError("Generated conversation changed its required topic domain.")
    realized_conditions = {prompt.condition for prompt in plan.user_prompts}
    missing_conditions = set(brief.required_conditions) - realized_conditions
    if missing_conditions:
        raise ValueError(
            f"Generated conversation omitted conditions: {', '.join(sorted(missing_conditions))}."
        )
    if not any(
        prompt.delivery.pace is brief.pace and prompt.delivery.affect is brief.affect
        for prompt in plan.user_prompts
    ):
        raise ValueError("Generated conversation omitted its anchor pace and affect.")


def _prompt_texts(prompt: UserPrompt) -> tuple[str, ...]:
    match prompt:
        case HoldUserPrompt(text_before_pause=before, text_after_pause=after):
            return (before, after)
        case (
            CompletionUserPrompt(text=text)
            | NonFloorFeedbackUserPrompt(text=text)
            | ResponseFloorClaimUserPrompt(text=text)
            | InterruptionFloorClaimUserPrompt(text=text)
        ):
            return (text,)


def _normalized_user_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _write_atomically(path: Path, prompt_set: EnglishConversationPromptSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(prompt_set.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Prompt model output did not contain a JSON object.")
    return text[start : end + 1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate coherent English synthetic conversation prompt plans."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--count", type=int, choices=range(10, 21), default=20)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
