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

from app.local.synthetic_generation.completion_dataset import SyntheticSpeechPrompt
from app.local.synthetic_generation.completion_prompts import (
    PromptGeneratorProvenance,
    SpeechPromptDraftBatch,
    SyntheticSpeechPromptSet,
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
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = parsed.output.with_suffix(f"{parsed.output.suffix}.partial")

    def checkpoint(prompts: tuple[SyntheticSpeechPrompt, ...]) -> None:
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
    on_progress: Callable[[tuple[SyntheticSpeechPrompt, ...]], None],
) -> tuple[SyntheticSpeechPrompt, ...]:
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
) -> SpeechPromptDraftBatch:
    instruction = _generation_instruction(batch_size)
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


def _generation_instruction(batch_size: int) -> str:
    return f"""Create {batch_size} distinct long-form English text-to-speech prompts.
Return only one JSON object shaped exactly as:
{{"prompts":[{{"text":"...","voice_instruction":"...","topic":"..."}}]}}

Requirements for every prompt:
- Text contains 55-115 words and sounds like one natural conversational turn, not a list.
- Use ordinary punctuation to create two or three plausible thoughtful pauses. At least one pause
  should plausibly last over 500 ms when spoken, using a sentence boundary, an em dash, or an
  explicit hesitation such as "uh".
- End with a clearly complete statement or question. No ellipsis at the end.
- Vary syntax, sentence length, topic, emotion, age presentation, regional accent, speaking pace,
  pitch, energy, and vocal texture across items.
- voice_instruction is a detailed natural-language instruction for Qwen3-TTS VoiceDesign and must
  specify perceived age, voice character, accent or dialect, pace, emotion, and how pauses sound.
- Require a clean, close-mic studio recording with normal voiced projection. Do not request
  whispering, breathiness, hushed delivery, ambient sound, room tone, or background noise.
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
