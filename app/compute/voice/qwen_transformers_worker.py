from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from typing import TextIO

from app.compute.voice.qwen_config import QwenBackend, QwenModelConfiguration
from app.compute.voice.qwen_transformers_runtime import QwenTransformersRuntime
from app.compute.voice.qwen_worker import QwenWorkerController
from app.compute.voice.qwen_worker_cli import parse_qwen_worker_configuration


def parse_configuration(
    arguments: Sequence[str] | None = None,
) -> QwenModelConfiguration:
    return parse_qwen_worker_configuration(QwenBackend.TRANSFORMERS, arguments)


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
        runtime = QwenTransformersRuntime(configuration)
        await QwenWorkerController(runtime, output_stream=protocol_output).run()
    finally:
        protocol_output.close()


def main(arguments: Sequence[str] | None = None) -> None:
    configuration = parse_configuration(arguments)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(run_worker(configuration))


if __name__ == "__main__":
    main()
