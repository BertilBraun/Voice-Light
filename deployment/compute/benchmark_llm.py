from __future__ import annotations

import argparse
import re
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict

import torch
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
    SEARCH_SUMMARIZER_MODEL_NAME,
    SEARCH_SUMMARIZER_MODEL_REVISION,
)
from app.compute.voice.qwen_worker import LANGUAGE_MODEL_SYSTEM_PROMPT
from app.shared.base_model import FrozenBaseModel

COMPLETE_WORD_PATTERN = re.compile(r"\S+\s+")


class PromptStyle(StrEnum):
    CURRENT = "current"
    SPOKEN_CONVERSATION = "spoken-conversation"
    COOPERATIVE_LISTENER = "cooperative-listener"


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    messages: tuple[ChatMessage, ...]


class BenchmarkMetadata(FrozenBaseModel):
    record_type: Literal["metadata"] = "metadata"
    model_name: str
    revision: str
    prompt_style: PromptStyle
    load_seconds: float = Field(ge=0.0)
    peak_allocated_mb: float = Field(ge=0.0)
    runs_per_case: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(gt=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    repetition_penalty: float = Field(gt=0.0)


class BenchmarkResult(FrozenBaseModel):
    record_type: Literal["result"] = "result"
    model_name: str
    prompt_style: PromptStyle
    case_name: str
    run_number: int = Field(gt=0)
    seed: int = Field(ge=0)
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    output_words: int = Field(ge=0)
    first_token_seconds: float = Field(ge=0.0)
    first_text_seconds: float = Field(ge=0.0)
    first_complete_word_seconds: float = Field(ge=0.0)
    total_seconds: float = Field(gt=0.0)
    tokens_per_second: float = Field(gt=0.0)
    output_text: str


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LANGUAGE_MODEL_NAME)
    parser.add_argument("--revision")
    parser.add_argument(
        "--prompt-style",
        choices=tuple(PromptStyle),
        default=PromptStyle.CURRENT,
        type=PromptStyle,
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-jsonl", type=Path)
    options = parser.parse_args(arguments)
    if options.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if options.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if options.temperature <= 0.0:
        raise ValueError("--temperature must be positive.")
    if not 0.0 < options.top_p <= 1.0:
        raise ValueError("--top-p must be greater than zero and at most one.")
    if options.repetition_penalty <= 0.0:
        raise ValueError("--repetition-penalty must be positive.")

    revision = options.revision or _default_revision(options.model)
    run_benchmark(
        model_name=options.model,
        revision=revision,
        prompt_style=options.prompt_style,
        runs=options.runs,
        max_new_tokens=options.max_new_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        repetition_penalty=options.repetition_penalty,
        initial_seed=options.seed,
        output_jsonl=options.output_jsonl,
    )


def run_benchmark(
    model_name: str,
    revision: str,
    prompt_style: PromptStyle,
    runs: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    initial_seed: int,
    output_jsonl: Path | None,
) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started_at = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started_at
    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    metadata = BenchmarkMetadata(
        model_name=model_name,
        revision=revision,
        prompt_style=prompt_style,
        load_seconds=load_seconds,
        peak_allocated_mb=peak_allocated_mb,
        runs_per_case=runs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    _emit_record(metadata, output_jsonl, overwrite=True)

    system_prompt = _system_prompt(prompt_style)
    _warm_model(model, tokenizer, system_prompt)
    for case_index, benchmark_case in enumerate(_benchmark_cases()):
        for run_index in range(runs):
            seed = initial_seed + case_index * runs + run_index
            result = _run_case(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                prompt_style=prompt_style,
                system_prompt=system_prompt,
                benchmark_case=benchmark_case,
                run_number=run_index + 1,
                seed=seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            _emit_record(result, output_jsonl, overwrite=False)


def _warm_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str,
) -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(
            **model_inputs,
            max_new_tokens=8,
            do_sample=False,
        )
    torch.cuda.synchronize()


def _run_case(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    prompt_style: PromptStyle,
    system_prompt: str,
    benchmark_case: BenchmarkCase,
    run_number: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> BenchmarkResult:
    messages: list[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        *benchmark_case.messages,
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(model_inputs.input_ids.shape[1])
    streamer = TimedTextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_errors: list[Exception] = []
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(
                    **model_inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
        except Exception as error:
            generation_errors.append(error)
            streamer.on_finalized_text("", stream_end=True)

    torch.cuda.synchronize()
    started_at = time.perf_counter()
    streamer.mark_started(started_at)
    generation_thread = threading.Thread(target=generate, daemon=True)
    generation_thread.start()
    output_parts: list[str] = []
    first_text_seconds: float | None = None
    first_complete_word_seconds: float | None = None
    for text_delta in _nonempty_deltas(streamer):
        output_parts.append(text_delta)
        elapsed_seconds = time.perf_counter() - started_at
        if first_text_seconds is None:
            first_text_seconds = elapsed_seconds
        if (
            first_complete_word_seconds is None
            and COMPLETE_WORD_PATTERN.search("".join(output_parts)) is not None
        ):
            first_complete_word_seconds = elapsed_seconds
    generation_thread.join()
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started_at
    if generation_errors:
        raise RuntimeError(f"Language model benchmark failed: {generation_errors[0]}")

    output_text = "".join(output_parts).strip()
    if streamer.first_token_seconds is None:
        raise RuntimeError("Language model benchmark produced no generated token.")
    if first_text_seconds is None or not output_text:
        raise RuntimeError("Language model benchmark produced no text.")
    if first_complete_word_seconds is None:
        first_complete_word_seconds = total_seconds
    output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
    if output_tokens < 1:
        raise RuntimeError("Language model benchmark produced no output tokens.")
    return BenchmarkResult(
        model_name=model_name,
        prompt_style=prompt_style,
        case_name=benchmark_case.name,
        run_number=run_number,
        seed=seed,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        output_words=len(output_text.split()),
        first_token_seconds=streamer.first_token_seconds,
        first_text_seconds=first_text_seconds,
        first_complete_word_seconds=first_complete_word_seconds,
        total_seconds=total_seconds,
        tokens_per_second=output_tokens / total_seconds,
        output_text=output_text,
    )


def _nonempty_deltas(streamer: TextIteratorStreamer) -> Iterator[str]:
    for text_delta in streamer:
        if text_delta:
            yield text_delta


class TimedTextIteratorStreamer(TextIteratorStreamer):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        skip_prompt: bool,
        skip_special_tokens: bool,
    ) -> None:
        super().__init__(
            tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens,
        )
        self.started_at: float | None = None
        self.first_token_seconds: float | None = None

    def mark_started(self, started_at: float) -> None:
        self.started_at = started_at

    def put(self, value: torch.Tensor) -> None:
        is_prompt = self.skip_prompt and self.next_tokens_are_prompt
        super().put(value)
        if is_prompt or self.first_token_seconds is not None:
            return
        if self.started_at is None:
            raise AssertionError("Timed streamer received a token before generation started.")
        self.first_token_seconds = time.perf_counter() - self.started_at


def _system_prompt(prompt_style: PromptStyle) -> str:
    match prompt_style:
        case PromptStyle.CURRENT:
            return LANGUAGE_MODEL_SYSTEM_PROMPT
        case PromptStyle.SPOKEN_CONVERSATION:
            return (
                "You are speaking in a live voice conversation. Respond to the user's latest "
                "meaning, not merely its wording. Usually speak in one or two short sentences. "
                "Prefer natural spoken language, contractions, and concrete reactions. Avoid "
                "lists, headings, summaries, generic praise, and customer-service phrases. Do "
                "not restate what the user just said. If a brief acknowledgement is socially "
                "useful, make it specific and continue directly with substance. End once you "
                "have made the useful conversational move. Use plain text without Markdown or "
                "emoji."
            )
        case PromptStyle.COOPERATIVE_LISTENER:
            return (
                "You are a thoughtful participant in a live spoken conversation. Act like a "
                "cooperative listener rather than a chatbot. Keep most turns short enough to be "
                "comfortably interrupted, normally one or two clauses. React specifically to "
                "what matters, then add one useful thought or question. Do not recap the user's "
                "message, announce what you are doing, offer a menu of options, or use generic "
                "filler such as 'Absolutely' or 'That's a great question.' Use natural repair "
                "language when correcting a misunderstanding. Use plain text without Markdown "
                "or emoji."
            )


def _benchmark_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase(
            name="direct_question",
            messages=(
                {
                    "role": "user",
                    "content": "Why do conversations with voice assistants often feel unnatural?",
                },
            ),
        ),
        BenchmarkCase(
            name="personal_disclosure",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "I finally told my manager the project is burning me out. I expected an "
                        "argument, but she actually listened."
                    ),
                },
            ),
        ),
        BenchmarkCase(
            name="uncertain_thought",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "I don't know. Maybe I'm overthinking it, but something about the whole "
                        "conversation still feels off."
                    ),
                },
            ),
        ),
        BenchmarkCase(
            name="repair_after_misunderstanding",
            messages=(
                {
                    "role": "user",
                    "content": "I might leave the company after this launch.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "It sounds like you have already decided to resign once the launch is over."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "No, that's not what I said. I'm considering it, but I haven't decided."
                    ),
                },
            ),
        ),
        BenchmarkCase(
            name="multi_turn_coherence",
            messages=(
                {
                    "role": "user",
                    "content": "I'm trying to make this voice system feel less turn-based.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Then the interaction policy matters as much as raw response latency."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Right. The audio already sounds good, but it still feels like a chatbot."
                    ),
                },
            ),
        ),
        BenchmarkCase(
            name="concision_pressure",
            messages=(
                {
                    "role": "user",
                    "content": (
                        "Give me the most important next step, but keep it conversational and "
                        "don't turn it into a list."
                    ),
                },
            ),
        ),
    )


def _default_revision(model_name: str) -> str:
    if model_name == LANGUAGE_MODEL_NAME:
        return LANGUAGE_MODEL_REVISION
    if model_name == SEARCH_SUMMARIZER_MODEL_NAME:
        return SEARCH_SUMMARIZER_MODEL_REVISION
    raise ValueError("--revision is required for an unrecognized model.")


def _emit_record(
    record: BenchmarkMetadata | BenchmarkResult,
    output_jsonl: Path | None,
    overwrite: bool,
) -> None:
    serialized_record = record.model_dump_json()
    print(serialized_record, flush=True)
    if output_jsonl is None:
        return
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with output_jsonl.open(mode, encoding="utf-8") as output_stream:
        output_stream.write(serialized_record + "\n")


if __name__ == "__main__":
    main()
