from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from typing import TextIO

from app.compute.voice.qwen_config import (
    QwenAdapterConfiguration,
    QwenModelConfiguration,
)
from app.compute.voice.qwen_vllm_runtime import QwenVllmRuntime
from app.compute.voice.qwen_worker import QwenWorkerController


def parse_configuration(
    arguments: Sequence[str] | None = None,
) -> QwenModelConfiguration:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--gpu-memory-utilization", required=True, type=float)
    parser.add_argument("--maximum-model-length", required=True, type=int)
    options = parser.parse_args(arguments)
    if (options.adapter is None) != (options.adapter_revision is None):
        parser.error("--adapter and --adapter-revision must be provided together")
    adapter = (
        None
        if options.adapter is None
        else QwenAdapterConfiguration(
            repository_id=options.adapter,
            revision=options.adapter_revision,
        )
    )
    return QwenModelConfiguration(
        model_name=options.model,
        model_revision=options.revision,
        adapter=adapter,
        gpu_memory_utilization=options.gpu_memory_utilization,
        maximum_model_length=options.maximum_model_length,
    )


def redirect_inference_stdout() -> TextIO:
    protocol_file_descriptor = os.dup(sys.stdout.fileno())
    protocol_output = os.fdopen(
        protocol_file_descriptor,
        mode="w",
        buffering=1,
        encoding="utf-8",
    )
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return protocol_output


async def run_worker(configuration: QwenModelConfiguration) -> None:
    protocol_output = redirect_inference_stdout()
    try:
        runtime = QwenVllmRuntime(configuration)
        await QwenWorkerController(runtime, output_stream=protocol_output).run()
    finally:
        protocol_output.close()


def main(arguments: Sequence[str] | None = None) -> None:
    configuration = parse_configuration(arguments)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(run_worker(configuration))


if __name__ == "__main__":
    main()
