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


class GenerationFlowControl:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.paused = False
        self.cancelled = False

    def pause(self) -> None:
        with self.condition:
            self.paused = True

    def resume(self) -> None:
        with self.condition:
            self.paused = False
            self.condition.notify_all()

    def cancel(self) -> None:
        with self.condition:
            self.cancelled = True
            self.paused = False
            self.condition.notify_all()

    def wait_until_ready(self) -> bool:
        with self.condition:
            while self.paused and not self.cancelled:
                self.condition.wait()
            return not self.cancelled


class GenerationFlowControlLogitsProcessor(LogitsProcessor):
    def __init__(self, flow_control: GenerationFlowControl, eos_token_id: int) -> None:
        self.flow_control = flow_control
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        if self.flow_control.wait_until_ready():
            return scores
        cancelled_scores = torch.full_like(scores, -torch.inf)
        cancelled_scores[:, self.eos_token_id] = 0.0
        return cancelled_scores


@dataclass
class GenerationThreadState:
    error: Exception | None = None


@dataclass(frozen=True)
class ActiveTransformersGeneration:
    invocation_id: int
    flow_control: GenerationFlowControl


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
        self.sampling = configuration.sampling
        self.active_generation: ActiveTransformersGeneration | None = None

    async def stream_text(
        self,
        command: QwenGenerationCommand,
    ) -> AsyncIterator[GeneratedTextDelta]:
        if self.active_generation is not None:
            raise RuntimeError("The Transformers Qwen runtime already has an active generation.")
        prompt = render_qwen_prompt(self.tokenizer, command)
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            cast(PreTrainedTokenizerBase, self.tokenizer),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        flow_control = GenerationFlowControl()
        flow_control_processor = GenerationFlowControlLogitsProcessor(
            flow_control,
            self.tokenizer.eos_token_id,
        )
        active_generation = ActiveTransformersGeneration(command.invocation_id, flow_control)
        self.active_generation = active_generation
        thread_state = GenerationThreadState()
        generation_thread = threading.Thread(
            target=self._generate,
            args=(command, model_inputs, streamer, flow_control_processor, thread_state),
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
            flow_control.cancel()
            await asyncio.shield(asyncio.to_thread(generation_thread.join))
            if self.active_generation is active_generation:
                self.active_generation = None

    def _generate(
        self,
        command: QwenGenerationCommand,
        model_inputs: BatchEncoding,
        streamer: TextIteratorStreamer,
        flow_control_processor: GenerationFlowControlLogitsProcessor,
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
                            logits_processor=LogitsProcessorList([flow_control_processor]),
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=self.sampling.temperature,
                            top_p=self.sampling.top_p,
                            top_k=self.sampling.top_k,
                        )
                    case GenerateTextLlmCommand(max_new_tokens=max_new_tokens):
                        self.model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=model_inputs.attention_mask,
                            streamer=streamer,
                            logits_processor=LogitsProcessorList([flow_control_processor]),
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
        except Exception as error:
            thread_state.error = error
            streamer.on_finalized_text("", stream_end=True)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    def pause(self, invocation_id: int) -> None:
        self._active_flow_control(invocation_id).pause()

    def resume(self, invocation_id: int) -> None:
        self._active_flow_control(invocation_id).resume()

    def _active_flow_control(self, invocation_id: int) -> GenerationFlowControl:
        active_generation = self.active_generation
        if active_generation is None or active_generation.invocation_id != invocation_id:
            raise ValueError(f"No active Transformers generation for invocation {invocation_id}.")
        return active_generation.flow_control

    async def sleep(self) -> None:
        raise RuntimeError("The Transformers Qwen backend does not support memory snapshots.")

    async def wake(self) -> None:
        raise RuntimeError("The Transformers Qwen backend does not support memory snapshots.")


def _next_text(streamer: TextIteratorStreamer) -> str | None:
    try:
        return next(streamer)
    except StopIteration:
        return None
