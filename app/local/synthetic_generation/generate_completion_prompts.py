from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
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

from app.local.synthetic_generation.completion_dataset import (
    PromptLanguage,
    SyntheticSpeechPrompt,
)
from app.local.synthetic_generation.completion_prompts import (
    PromptGeneratorProvenance,
    SpeechPromptDraftBatch,
    SyntheticSpeechPromptSet,
)

LANGUAGE_WEIGHTS = (
    (PromptLanguage.ENGLISH, 0.55),
    (PromptLanguage.GERMAN, 0.20),
    (PromptLanguage.FRENCH, 0.15),
    (PromptLanguage.SPANISH, 0.10),
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
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
    prompts = generate_speech_prompts(
        model=model,
        tokenizer=tokenizer,
        prompt_count=parsed.count,
        batch_size=parsed.batch_size,
        seed=parsed.seed,
    )
    artifact = SyntheticSpeechPromptSet(
        set_id=parsed.set_id,
        provenance=PromptGeneratorProvenance(
            model_id=parsed.model,
            model_revision=revision,
            runtime_version=transformers.__version__,
            seed=parsed.seed,
            requested_prompt_count=parsed.count,
        ),
        prompts=prompts,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {len(prompts)} prompts to {parsed.output}", flush=True)


def generate_speech_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_count: int,
    batch_size: int,
    seed: int,
) -> tuple[SyntheticSpeechPrompt, ...]:
    generator = random.Random(seed)
    prompts = []
    used_texts: set[str] = set()
    attempt = 0
    while len(prompts) < prompt_count:
        language = _sample_language(generator)
        requested_batch_size = min(batch_size, prompt_count - len(prompts))
        try:
            draft_batch = _generate_draft_batch(
                model=model,
                tokenizer=tokenizer,
                language=language,
                batch_size=requested_batch_size,
                seed=seed + attempt,
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
            normalized_text = " ".join(draft.text.lower().split())
            if normalized_text in used_texts:
                continue
            prompt_index = len(prompts)
            prompts.append(
                SyntheticSpeechPrompt(
                    prompt_id=f"prompt_{prompt_index:05d}",
                    language=language,
                    text=draft.text,
                    voice_instruction=draft.voice_instruction,
                    topic=draft.topic,
                    seed=seed + prompt_index,
                )
            )
            used_texts.add(normalized_text)
            if len(prompts) == prompt_count:
                break
        attempt += 1
        if attempt > prompt_count * 4:
            raise ValueError("Prompt generation failed to produce enough unique valid drafts.")
    return tuple(prompts)


def _generate_draft_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    language: PromptLanguage,
    batch_size: int,
    seed: int,
) -> SpeechPromptDraftBatch:
    instruction = _generation_instruction(language, batch_size)
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


def _generation_instruction(language: PromptLanguage, batch_size: int) -> str:
    return f"""Create {batch_size} distinct long-form text-to-speech prompts in {language.value}.
Return only one JSON object shaped exactly as:
{{"prompts":[{{"text":"...","voice_instruction":"...","topic":"..."}}]}}

Requirements for every prompt:
- Text contains 55-115 words and sounds like one natural conversational turn, not a list.
- Use ordinary punctuation to create two or three plausible thoughtful pauses. At least one pause
  should plausibly last over 500 ms when spoken, using a sentence boundary, an em dash, or an
  explicit hesitation such as "uh" or its natural {language.value} equivalent.
- End with a clearly complete statement or question. No ellipsis at the end.
- Vary syntax, sentence length, topic, emotion, age presentation, regional accent, speaking pace,
  pitch, energy, and vocal texture across items.
- voice_instruction is a detailed natural-language instruction for Qwen3-TTS VoiceDesign and must
  specify perceived age, voice character, accent or dialect, pace, emotion, and how pauses sound.
- Avoid quotations, unsafe content, copyrighted passages, names of real public figures, stage
  directions inside text, and repeated templates.
- topic is a short descriptive phrase.
"""


def _json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Prompt model output did not contain a JSON object.")
    return text[start : end + 1]


def _sample_language(generator: random.Random) -> PromptLanguage:
    draw = generator.random()
    cumulative = 0.0
    for language, weight in LANGUAGE_WEIGHTS:
        cumulative += weight
        if draw <= cumulative:
            return language
    return PromptLanguage.SPANISH


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate varied long-form TTS prompts with Qwen3."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
