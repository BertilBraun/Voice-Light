from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

from app.compute.voice.llm_worker_protocol import GenerateTextLlmCommand, StartLlmCommand
from app.compute.voice.qwen_config import QwenModelConfiguration
from app.compute.voice.qwen_worker import (
    GeneratedTextDelta,
    QwenChatTemplateTokenizer,
    QwenGenerationCommand,
    render_qwen_prompt,
)


class QwenTransformersTokenizer(QwenChatTemplateTokenizer, Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, *, return_tensors: str) -> BatchEncoding: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class CancellationLogitsProcessor(LogitsProcessor):
    def __init__(self, cancellation_event: threading.Event, eos_token_id: int) -> None:
        self.cancellation_event = cancellation_event
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if not self.cancellation_event.is_set():
            return scores
        cancelled_scores = torch.full_like(scores, -torch.inf)
        cancelled_scores[:, self.eos_token_id] = 0.0
        return cancelled_scores


@dataclass
class GenerationThreadState:
    error: Exception | None = None


class QwenTransformersRuntime:
    def __init__(self, configuration: QwenModelConfiguration) -> None:
        if configuration.adapter is not None:
            raise ValueError("The Transformers Qwen backend requires a merged model checkpoint.")
        tokenizer = AutoTokenizer.from_pretrained(
            configuration.model_name,
            revision=configuration.model_revision,
        )
        self.tokenizer = cast(QwenTransformersTokenizer, tokenizer)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The Qwen tokenizer must define an EOS token.")
        self.model = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                configuration.model_name,
                revision=configuration.model_revision,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ),
        ).to("cuda")
        self.model.eval()

    async def stream_text(
        self,
        command: QwenGenerationCommand,
    ) -> AsyncIterator[GeneratedTextDelta]:
        prompt = render_qwen_prompt(self.tokenizer, command)
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            cast(PreTrainedTokenizerBase, self.tokenizer),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        cancellation_event = threading.Event()
        cancellation_processor = CancellationLogitsProcessor(
            cancellation_event,
            self.tokenizer.eos_token_id,
        )
        thread_state = GenerationThreadState()
        generation_thread = threading.Thread(
            target=self._generate,
            args=(command, model_inputs, streamer, cancellation_processor, thread_state),
            daemon=True,
            name=f"qwen-transformers-{command.invocation_id}",
        )
        generation_thread.start()
        generated_text = ""
        try:
            while True:
                text = await asyncio.to_thread(_next_text, streamer)
                if text is None:
                    break
                generated_text += text
                yield GeneratedTextDelta(
                    text=text,
                    cumulative_token_count=len(
                        self.tokenizer.encode(generated_text, add_special_tokens=False)
                    ),
                )
            await asyncio.to_thread(generation_thread.join)
            if thread_state.error is not None:
                raise thread_state.error
        finally:
            cancellation_event.set()
            await asyncio.shield(asyncio.to_thread(generation_thread.join))

    def _generate(
        self,
        command: QwenGenerationCommand,
        model_inputs: BatchEncoding,
        streamer: TextIteratorStreamer,
        cancellation_processor: CancellationLogitsProcessor,
        thread_state: GenerationThreadState,
    ) -> None:
        try:
            with torch.inference_mode():
                match command:
                    case StartLlmCommand():
                        self.model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=model_inputs.attention_mask,
                            streamer=streamer,
                            logits_processor=LogitsProcessorList([cancellation_processor]),
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,
                        )
                    case GenerateTextLlmCommand(max_new_tokens=max_new_tokens):
                        self.model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=model_inputs.attention_mask,
                            streamer=streamer,
                            logits_processor=LogitsProcessorList([cancellation_processor]),
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
        except Exception as error:
            thread_state.error = error
            streamer.on_finalized_text("", stream_end=True)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    async def sleep(self) -> None:
        raise RuntimeError("The Transformers Qwen backend does not support memory snapshots.")

    async def wake(self) -> None:
        raise RuntimeError("The Transformers Qwen backend does not support memory snapshots.")


def _next_text(streamer: TextIteratorStreamer) -> str | None:
    try:
        return next(streamer)
    except StopIteration:
        return None
