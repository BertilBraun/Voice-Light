from __future__ import annotations

import logging
from asyncio import Event
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast

from huggingface_hub import snapshot_download
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from app.compute.voice.llm_worker_protocol import (
    GenerateTextLlmCommand,
    StartLlmCommand,
)
from app.compute.voice.qwen_config import (
    QwenAdapterConfiguration,
    QwenModelConfiguration,
    QwenSamplingConfiguration,
)
from app.compute.voice.qwen_worker import (
    GeneratedTextDelta,
    QwenChatTemplateTokenizer,
    QwenGenerationCommand,
    render_qwen_prompt,
)

logger = logging.getLogger(__name__)
MAXIMUM_LORA_RANK = 16


@dataclass(frozen=True)
class ActiveVllmGeneration:
    invocation_id: int
    output_consumption_allowed: Event


class QwenVllmRuntime:
    def __init__(self, configuration: QwenModelConfiguration) -> None:
        adapter = configuration.adapter
        engine_arguments = AsyncEngineArgs(
            model=configuration.model_name,
            tokenizer=configuration.model_name,
            revision=configuration.model_revision,
            tokenizer_revision=configuration.model_revision,
            dtype="bfloat16",
            max_model_len=configuration.maximum_model_length,
            gpu_memory_utilization=configuration.gpu_memory_utilization,
            max_num_seqs=1,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            generation_config="vllm",
            disable_log_stats=True,
            enable_log_requests=False,
            enable_lora=adapter is not None,
            max_lora_rank=MAXIMUM_LORA_RANK,
            max_loras=1,
            max_cpu_loras=1,
            enforce_eager=configuration.enforce_eager,
            enable_sleep_mode=True,
        )
        self.engine = AsyncLLM.from_engine_args(engine_arguments)
        self.tokenizer = cast(QwenChatTemplateTokenizer, self.engine.get_tokenizer())
        self.lora_request = create_lora_request(adapter)
        self.sampling = configuration.sampling
        self.active_generation: ActiveVllmGeneration | None = None

    async def stream_text(
        self,
        command: QwenGenerationCommand,
    ) -> AsyncIterator[GeneratedTextDelta]:
        if self.active_generation is not None:
            raise RuntimeError("The vLLM Qwen runtime already has an active generation.")
        prompt = render_qwen_prompt(self.tokenizer, command)
        sampling_parameters = sampling_parameters_for_command(command, self.sampling)
        request_id = f"qwen-{command.invocation_id}"
        cumulative_token_count = 0
        output_consumption_allowed = Event()
        output_consumption_allowed.set()
        active_generation = ActiveVllmGeneration(
            invocation_id=command.invocation_id,
            output_consumption_allowed=output_consumption_allowed,
        )
        self.active_generation = active_generation
        try:
            async for request_output in self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_parameters,
                request_id=request_id,
                lora_request=self.lora_request,
            ):
                # vLLM may continue inference while paused; only worker output consumption is gated.
                await output_consumption_allowed.wait()
                if len(request_output.outputs) != 1:
                    raise AssertionError(
                        "Qwen vLLM generation must produce exactly one completion."
                    )
                completion = request_output.outputs[0]
                cumulative_token_count += len(completion.token_ids)
                yield GeneratedTextDelta(
                    text=completion.text,
                    cumulative_token_count=cumulative_token_count,
                )
        finally:
            output_consumption_allowed.set()
            if self.active_generation is active_generation:
                self.active_generation = None

    def close(self) -> None:
        self.engine.shutdown()

    def pause(self, invocation_id: int) -> None:
        self._active_output_gate(invocation_id).clear()

    def resume(self, invocation_id: int) -> None:
        self._active_output_gate(invocation_id).set()

    def _active_output_gate(self, invocation_id: int) -> Event:
        active_generation = self.active_generation
        if active_generation is None or active_generation.invocation_id != invocation_id:
            raise ValueError(f"No active vLLM generation for invocation {invocation_id}.")
        return active_generation.output_consumption_allowed

    async def sleep(self) -> None:
        await self.engine.sleep(level=1)

    async def wake(self) -> None:
        await self.engine.wake_up()


def create_lora_request(
    adapter: QwenAdapterConfiguration | None,
) -> LoRARequest | None:
    match adapter:
        case None:
            return None
        case QwenAdapterConfiguration(repository_id=repository_id, revision=revision):
            adapter_path = snapshot_download(repository_id, revision=revision)
            logger.info(
                "activated Qwen adapter %s at revision %s",
                repository_id,
                revision,
            )
            return LoRARequest("voice-light-tool-use", 1, adapter_path)


def sampling_parameters_for_command(
    command: QwenGenerationCommand,
    sampling: QwenSamplingConfiguration,
) -> SamplingParams:
    match command:
        case StartLlmCommand():
            return SamplingParams(
                max_tokens=256,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                output_kind=RequestOutputKind.DELTA,
            )
        case GenerateTextLlmCommand(max_new_tokens=max_new_tokens):
            return SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
