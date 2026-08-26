from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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

from app.local.synthetic_generation.completion_dataset import SyntheticEnglishSpeechPrompt
from app.local.synthetic_generation.completion_prompts import (
    PromptDeliveryProfile,
    PromptGeneratorProvenance,
    SpeechPromptDraftBatch,
    SyntheticSpeechPromptSet,
    apply_prompt_draft_profile,
    validate_prompt_set_id,
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    set_id = validate_prompt_set_id(parsed.set_id)
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
    provenance = PromptGeneratorProvenance(
        model_id=parsed.model,
        model_revision=revision,
        runtime_version=transformers.__version__,
        seed=parsed.seed,
        requested_prompt_count=parsed.count,
        delivery_profile=parsed.delivery_profile,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = parsed.output.with_suffix(f"{parsed.output.suffix}.partial")

    def checkpoint(prompts: tuple[SyntheticEnglishSpeechPrompt, ...]) -> None:
        if not prompts:
            return
        artifact = SyntheticSpeechPromptSet(
            set_id=set_id,
            provenance=provenance,
            prompts=prompts,
        )
        _write_atomically(partial_output, artifact)
        print(f"Checkpointed {len(prompts)}/{parsed.count} prompts", flush=True)

    prompts = generate_speech_prompts(
        model=model,
        tokenizer=tokenizer,
        prompt_count=parsed.count,
        batch_size=parsed.batch_size,
        seed=parsed.seed,
        delivery_profile=parsed.delivery_profile,
        on_progress=checkpoint,
    )
    artifact = SyntheticSpeechPromptSet(
        set_id=set_id,
        provenance=provenance,
        prompts=prompts,
    )
    _write_atomically(parsed.output, artifact)
    partial_output.unlink(missing_ok=True)
    print(f"Wrote {len(prompts)} prompts to {parsed.output}", flush=True)


def generate_speech_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_count: int,
    batch_size: int,
    seed: int,
    delivery_profile: PromptDeliveryProfile,
    on_progress: Callable[[tuple[SyntheticEnglishSpeechPrompt, ...]], None],
) -> tuple[SyntheticEnglishSpeechPrompt, ...]:
    prompts = []
    used_texts: set[str] = set()
    attempt = 0
    while len(prompts) < prompt_count:
        requested_batch_size = min(batch_size, prompt_count - len(prompts))
        try:
            draft_batch = _generate_draft_batch(
                model=model,
                tokenizer=tokenizer,
                batch_size=requested_batch_size,
                seed=seed + attempt,
                delivery_profile=delivery_profile,
            )
        except ValueError as error:
            print(f"Discarding invalid prompt batch: {error}", flush=True)
            attempt += 1
            if attempt > prompt_count * 4:
                raise ValueError(
                    "Prompt generation failed to produce enough unique valid drafts."
                ) from error
            continue
        for draft in draft_batch.prompts:
            try:
                profiled_draft = apply_prompt_draft_profile(
                    draft,
                    delivery_profile,
                    seed + attempt,
                )
            except ValueError as error:
                print(f"Discarding off-profile prompt draft: {error}", flush=True)
                continue
            normalized_text = " ".join(profiled_draft.text.lower().split())
            if normalized_text in used_texts:
                continue
            prompt_index = len(prompts)
            try:
                prompt = SyntheticEnglishSpeechPrompt(
                    prompt_id=f"prompt_{prompt_index:05d}",
                    text=profiled_draft.text,
                    voice_instruction=profiled_draft.voice_instruction,
                    topic=profiled_draft.topic,
                    seed=seed + prompt_index,
                )
            except ValidationError as error:
                print(f"Discarding invalid prompt draft: {error}", flush=True)
                continue
            prompts.append(prompt)
            used_texts.add(normalized_text)
            if len(prompts) == prompt_count:
                break
        attempt += 1
        on_progress(tuple(prompts))
        if attempt > prompt_count * 4:
            raise ValueError("Prompt generation failed to produce enough unique valid drafts.")
    return tuple(prompts)


def _write_atomically(path: Path, artifact: SyntheticSpeechPromptSet) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _generate_draft_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    seed: int,
    delivery_profile: PromptDeliveryProfile,
) -> SpeechPromptDraftBatch:
    instruction = _generation_instruction(batch_size, delivery_profile)
    messages = [{"role": "user", "content": instruction}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    torch.manual_seed(seed)
    output = model.generate(
        **inputs,
        max_new_tokens=1100,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
    )
    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    json_text = _json_object(generated)
    try:
        return SpeechPromptDraftBatch.model_validate_json(json_text)
    except ValidationError as error:
        raise ValueError(f"Prompt model returned invalid structured output: {error}") from error


@dataclass(frozen=True)
class _ProfileRequirements:
    word_count_requirement: str
    delivery_requirements: str


def _generation_instruction(
    batch_size: int,
    delivery_profile: PromptDeliveryProfile,
) -> str:
    profile_requirements = _profile_requirements(delivery_profile)
    return f"""Create {batch_size} distinct long-form English text-to-speech prompts.
Return only one JSON object shaped exactly as:
{{"prompts":[{{"text":"...","voice_instruction":"...","topic":"..."}}]}}

Requirements for every prompt:
- Text contains {profile_requirements.word_count_requirement} and sounds like one natural
  conversational turn, not a list.
- Use ordinary punctuation to create two or three plausible conversational pauses. At least one
  pause should plausibly last over 500 ms when spoken, using a sentence boundary, an em dash, or
  an explicit hesitation such as "uh".
- End with a clearly complete statement or question. No ellipsis at the end.
- Vary syntax, sentence length, topic, emotion, age presentation, regional accent, pitch, energy,
  and vocal texture across items.
- voice_instruction is a detailed natural-language instruction for Qwen3-TTS VoiceDesign and must
  specify perceived age, voice character, accent or dialect, pace, emotion, and how pauses sound.
- Require a clean, close-mic studio recording with normal voiced projection. Do not request
  whispering, breathiness, hushed delivery, ambient sound, room tone, or background noise.
- Do not use the words whisper, breathy, hushed, ambient sound, room tone, or background noise,
  even in a negative instruction.
- Avoid quotations, unsafe content, copyrighted passages, names of real public figures, stage
  directions inside text, and repeated templates.
- topic is a short descriptive phrase.
{profile_requirements.delivery_requirements}
"""


def _profile_requirements(delivery_profile: PromptDeliveryProfile) -> _ProfileRequirements:
    match delivery_profile:
        case PromptDeliveryProfile.BALANCED:
            return _ProfileRequirements(
                word_count_requirement="55-115 words",
                delivery_requirements=(
                    "- Cover a balanced mix of slow, moderate, and fast delivery across the batch."
                ),
            )
        case PromptDeliveryProfile.BRISK_ENGAGED:
            return _ProfileRequirements(
                word_count_requirement="90-115 words",
                delivery_requirements=(
                    "- Every voice is brisk, engaged, energetic, and normally projected.\n"
                    "- Request a brisk, flowing pace without rushing or slurring. The pipeline "
                    "adds an exact rate from 200-240 spoken words per minute when the instruction "
                    "does not already contain one.\n"
                    "- Use lively everyday topics and active language; do not choose silence, "
                    "stillness, grief, meditation, nostalgia, loneliness, or quiet reflection as "
                    "topics.\n"
                    "- Vary perceived age, pitch, vocal weight, and English accent across North "
                    "American, British, Irish, Australian, and New Zealand voices.\n"
                    "- Include one clearly planned 500-800 ms internal pause, but keep all other "
                    "phrasing flowing and responsive. Do not request slow, subdued, soft-spoken, "
                    "intimate, contemplative, or deliberate delivery."
                ),
            )


def _json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Prompt model output did not contain a JSON object.")
    return text[start : end + 1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate varied long-form TTS prompts with Qwen3."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument(
        "--delivery-profile",
        type=PromptDeliveryProfile,
        choices=tuple(PromptDeliveryProfile),
        default=PromptDeliveryProfile.BALANCED,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
