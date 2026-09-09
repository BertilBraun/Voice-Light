from __future__ import annotations

import argparse
from collections.abc import Sequence

from app.compute.voice.qwen_config import (
    QwenAdapterConfiguration,
    QwenBackend,
    QwenModelConfiguration,
)


def parse_qwen_worker_configuration(
    backend: QwenBackend,
    arguments: Sequence[str] | None = None,
) -> QwenModelConfiguration:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--gpu-memory-utilization", required=True, type=float)
    parser.add_argument("--maximum-model-length", required=True, type=int)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
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
        enforce_eager=options.enforce_eager,
        backend=backend,
    )
