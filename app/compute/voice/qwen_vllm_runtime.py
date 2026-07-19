from __future__ import annotations

import logging
from collections.abc import AsyncIterator
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
)
from app.compute.voice.qwen_worker import (
    GeneratedTextDelta,
    QwenChatTemplateTokenizer,
    QwenGenerationCommand,
    render_qwen_prompt,
)

logger = logging.getLogger(__name__)
MAXIMUM_LORA_RANK = 16


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
            enable_chunked_prefill=False,
            enable_prefix_caching=True,
            generation_config="vllm",
            disable_log_stats=True,
            enable_log_requests=False,
            enable_lora=adapter is not None,
            max_lora_rank=MAXIMUM_LORA_RANK,
            max_loras=1,
            max_cpu_loras=1,
        )
        self.engine = AsyncLLM.from_engine_args(engine_arguments)
        self.tokenizer = cast(QwenChatTemplateTokenizer, self.engine.get_tokenizer())
        self.lora_request = create_lora_request(adapter)

    async def stream_text(
        self,
        command: QwenGenerationCommand,
    ) -> AsyncIterator[GeneratedTextDelta]:
        prompt = render_qwen_prompt(self.tokenizer, command)
        sampling_parameters = sampling_parameters_for_command(command)
        request_id = f"qwen-{command.invocation_id}"
        cumulative_token_count = 0
        async for request_output in self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_parameters,
            request_id=request_id,
            lora_request=self.lora_request,
        ):
            if len(request_output.outputs) != 1:
                raise AssertionError("Qwen vLLM generation must produce exactly one completion.")
            completion = request_output.outputs[0]
            cumulative_token_count += len(completion.token_ids)
            yield GeneratedTextDelta(
                text=completion.text,
                cumulative_token_count=cumulative_token_count,
            )

    def close(self) -> None:
        self.engine.shutdown()


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
) -> SamplingParams:
    match command:
        case StartLlmCommand():
            return SamplingParams(
                max_tokens=256,
                temperature=0.6,
                top_p=0.9,
                output_kind=RequestOutputKind.DELTA,
            )
        case GenerateTextLlmCommand(max_new_tokens=max_new_tokens):
            return SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
