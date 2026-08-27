from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Sequence
from itertools import islice
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
    ConversationGenerationBrief,
    ConversationPromptGeneratorProvenance,
    EnglishConversationContentDraft,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    assemble_conversation_prompt_plan,
    conversation_generation_instruction,
    representative_conversation_briefs,
    user_prompt_texts_for_uniqueness,
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
        target_conversation_hours=parsed.target_conversation_hours,
    )
    partial_path = parsed.output.with_suffix(f"{parsed.output.suffix}.partial")
    initial_plans = _load_partial_plans(partial_path, set_id, provenance)

    def checkpoint(plans: tuple[EnglishConversationPromptPlan, ...]) -> None:
        if not plans:
            return
        prompt_set = EnglishConversationPromptSet(
            set_id=set_id,
            provenance=provenance,
            plans=plans,
        )
        _write_atomically(partial_path, prompt_set)
        print(f"Checkpointed {len(plans)}/{parsed.count} conversation plans", flush=True)

    prompt_set = generate_conversation_prompt_set(
        model=model,
        tokenizer=tokenizer,
        set_id=set_id,
        provenance=provenance,
        on_progress=checkpoint,
        initial_plans=initial_plans,
        generation_batch_size=parsed.generation_batch_size,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(parsed.output, prompt_set)
    partial_path.unlink(missing_ok=True)
    print(f"Wrote {len(prompt_set.plans)} conversation plans to {parsed.output}", flush=True)


def generate_conversation_prompt_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    set_id: str,
    provenance: ConversationPromptGeneratorProvenance,
    on_progress: Callable[[tuple[EnglishConversationPromptPlan, ...]], None],
    initial_plans: tuple[EnglishConversationPromptPlan, ...] = (),
    generation_batch_size: int = 1,
) -> EnglishConversationPromptSet:
    if generation_batch_size <= 0:
        raise ValueError("Conversation generation batch size must be positive.")
    briefs = representative_conversation_briefs(
        count=provenance.requested_plan_count,
        seed=provenance.seed,
    )
    _validate_initial_plans(initial_plans, briefs)
    plans = list(initial_plans)
    planned_duration_seconds = sum(plan.target_duration_seconds for plan in plans)
    normalized_user_texts = {
        _normalized_user_text(text)
        for plan in plans
        for prompt in plan.user_prompts
        for text in user_prompt_texts_for_uniqueness(prompt)
    }
    normalized_reference_texts = {
        _normalized_user_text(plan.voice_reference_text) for plan in plans
    }
    target_hours = provenance.target_conversation_hours
    remaining_briefs = (
        ()
        if target_hours is not None and planned_duration_seconds >= target_hours * 3600.0
        else briefs[len(initial_plans) :]
    )
    remaining_iterator = iter(remaining_briefs)
    while brief_batch := tuple(islice(remaining_iterator, generation_batch_size)):
        initial_texts = _generate_initial_batch(model, tokenizer, brief_batch)
        for brief, initial_text in zip(brief_batch, initial_texts, strict=True):
            plan, plan_texts = _validated_unique_plan(
                model,
                tokenizer,
                brief,
                initial_text,
                normalized_user_texts,
                normalized_reference_texts,
            )
            plans.append(plan)
            planned_duration_seconds += plan.target_duration_seconds
            normalized_user_texts.update(plan_texts)
            normalized_reference_texts.add(_normalized_user_text(plan.voice_reference_text))
            on_progress(tuple(plans))
            if target_hours is not None and planned_duration_seconds >= target_hours * 3600.0:
                break
        if target_hours is not None and planned_duration_seconds >= target_hours * 3600.0:
            break
    prompt_set = EnglishConversationPromptSet(
        set_id=set_id,
        provenance=provenance,
        plans=tuple(plans),
    )
    if (
        provenance.target_conversation_hours is None
        and len(plans) != provenance.requested_plan_count
    ):
        raise ValueError("Prompt generation did not produce its requested conversation count.")
    return prompt_set


def _generate_initial_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    briefs: tuple[ConversationGenerationBrief, ...],
) -> tuple[str, ...]:
    conversations = [
        [{"role": "user", "content": conversation_generation_instruction(brief)}]
        for brief in briefs
    ]
    tokenizer.padding_side = "left"
    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    torch.manual_seed(_batch_seed(briefs))
    output = model.generate(
        **inputs,
        max_new_tokens=1100,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    return tuple(tokenizer.decode(row[prompt_length:], skip_special_tokens=True) for row in output)


def _validated_unique_plan(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    brief: ConversationGenerationBrief,
    initial_text: str,
    normalized_user_texts: set[str],
    normalized_reference_texts: set[str],
) -> tuple[EnglishConversationPromptPlan, tuple[str, ...]]:
    final_error: ValueError | None = None
    for attempt in range(4):
        try:
            plan = (
                _plan_from_generated_text(initial_text, brief)
                if attempt == 0
                else _generate_conversation_plan(
                    model=model,
                    tokenizer=tokenizer,
                    brief=brief,
                    seed=brief.seed + attempt,
                )
            )
            plan_texts = tuple(
                _normalized_user_text(text)
                for prompt in plan.user_prompts
                for text in user_prompt_texts_for_uniqueness(prompt)
            )
            if normalized_user_texts.intersection(plan_texts):
                raise ValueError("Generated conversation repeats user text from another plan.")
            normalized_reference_text = _normalized_user_text(plan.voice_reference_text)
            if normalized_reference_text in normalized_reference_texts:
                raise ValueError(
                    "Generated conversation repeats a voice reference from another plan."
                )
            return plan, plan_texts
        except ValueError as error:
            final_error = error
            print(
                f"Discarding invalid realization for {brief.plan_id}: {error}",
                flush=True,
            )
    assert final_error is not None
    raise ValueError(
        f"Conversation generation failed four times for {brief.plan_id}."
    ) from final_error


def _batch_seed(briefs: tuple[ConversationGenerationBrief, ...]) -> int:
    content = ":".join(f"{brief.plan_id}:{brief.seed}" for brief in briefs)
    return int.from_bytes(hashlib.sha256(content.encode()).digest()[:4], "big")


def _load_partial_plans(
    path: Path,
    set_id: str,
    provenance: ConversationPromptGeneratorProvenance,
) -> tuple[EnglishConversationPromptPlan, ...]:
    if not path.exists():
        return ()
    prompt_set = EnglishConversationPromptSet.model_validate_json(path.read_text(encoding="utf-8"))
    if prompt_set.set_id != set_id or prompt_set.provenance != provenance:
        raise ValueError("Partial prompt checkpoint does not match the requested corpus.")
    print(f"Resuming {len(prompt_set.plans)} checkpointed conversation plans", flush=True)
    return prompt_set.plans


def _validate_initial_plans(
    plans: tuple[EnglishConversationPromptPlan, ...],
    briefs: tuple[ConversationGenerationBrief, ...],
) -> None:
    if len(plans) > len(briefs):
        raise ValueError("Partial prompt checkpoint exceeds the requested plan count.")
    for plan, brief in zip(plans, briefs, strict=False):
        if (
            plan.plan_id != brief.plan_id
            or plan.seed != brief.seed
            or plan.domain is not brief.domain
        ):
            raise ValueError("Partial prompt checkpoint does not match its deterministic brief.")


def _generate_conversation_plan(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    brief: ConversationGenerationBrief,
    seed: int,
) -> EnglishConversationPromptPlan:
    messages = [{"role": "user", "content": conversation_generation_instruction(brief)}]
    final_error: ValidationError | ValueError | None = None
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
            max_new_tokens=1100,
            do_sample=True,
            temperature=0.8 if repair_index == 0 else 0.4,
            top_p=0.92,
        )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        try:
            return _plan_from_generated_text(generated, brief)
        except (ValidationError, ValueError) as error:
            final_error = error
            messages.extend(
                (
                    {"role": "assistant", "content": generated},
                    {"role": "user", "content": _repair_instruction(error)},
                )
            )
    assert final_error is not None
    raise ValueError(f"Prompt model returned an invalid conversation plan: {final_error}")


def _plan_from_generated_text(
    generated: str,
    brief: ConversationGenerationBrief,
) -> EnglishConversationPromptPlan:
    draft = EnglishConversationContentDraft.model_validate_json(_json_object(generated))
    plan = assemble_conversation_prompt_plan(draft, brief)
    _validate_plan_against_brief(plan, brief)
    return plan


def _repair_instruction(error: ValidationError | ValueError) -> str:
    return f"""The compact conversation-content JSON failed typed validation.
Return only a corrected complete JSON object with the same topic and conversation meaning. Do not
explain the correction. Keep exactly the schema fields from the original request. Do not add IDs,
sequence numbers, semantic labels, timing, pauses, numeric parameters, voice metadata, delivery
metadata, or backchannels. Correct the reported word count or missing natural-language field while
preserving coherence between the four user turns and three assistant turns. Keep the extended user
turn to 2-4 natural sentences, with no sentence longer than 35 words.

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
    parser.add_argument("--count", type=int, choices=range(5, 2_001), default=10)
    parser.add_argument("--target-conversation-hours", type=float)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
