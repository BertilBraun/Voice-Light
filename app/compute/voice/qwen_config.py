from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_ADAPTER_NAME,
    LANGUAGE_MODEL_ADAPTER_REVISION,
    LANGUAGE_MODEL_GPU_MEMORY_UTILIZATION,
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
    QWEN_MAXIMUM_MODEL_LENGTH,
)

MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE = "VOICE_LIGHT_MERGED_LANGUAGE_MODEL_NAME"
MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE = "VOICE_LIGHT_MERGED_LANGUAGE_MODEL_REVISION"
HUGGING_FACE_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)
HUGGING_FACE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class QwenAdapterConfiguration:
    repository_id: str
    revision: str


@dataclass(frozen=True)
class QwenModelConfiguration:
    model_name: str
    model_revision: str
    adapter: QwenAdapterConfiguration | None
    gpu_memory_utilization: float
    maximum_model_length: int

    def __post_init__(self) -> None:
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                "Qwen GPU memory utilization must be greater than zero and at most one."
            )
        if self.maximum_model_length <= 0:
            raise ValueError("Qwen maximum model length must be positive.")


def language_model_configuration_from_environment(
    environment: Mapping[str, str],
) -> QwenModelConfiguration:
    merged_model_name = environment.get(MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE)
    merged_model_revision = environment.get(MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE)
    if (merged_model_name is None) != (merged_model_revision is None):
        raise ValueError(
            f"{MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE} and "
            f"{MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE} must be configured together."
        )
    if merged_model_name is not None and merged_model_revision is not None:
        if not HUGGING_FACE_REPOSITORY_PATTERN.fullmatch(merged_model_name):
            raise ValueError(
                f"{MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE} must have the form owner/model."
            )
        if not HUGGING_FACE_COMMIT_PATTERN.fullmatch(merged_model_revision):
            raise ValueError(
                f"{MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE} must be a 40-character "
                "lowercase Hugging Face commit hash."
            )
        return QwenModelConfiguration(
            model_name=merged_model_name,
            model_revision=merged_model_revision,
            adapter=None,
            gpu_memory_utilization=LANGUAGE_MODEL_GPU_MEMORY_UTILIZATION,
            maximum_model_length=QWEN_MAXIMUM_MODEL_LENGTH,
        )
    return QwenModelConfiguration(
        model_name=LANGUAGE_MODEL_NAME,
        model_revision=LANGUAGE_MODEL_REVISION,
        adapter=QwenAdapterConfiguration(
            repository_id=LANGUAGE_MODEL_ADAPTER_NAME,
            revision=LANGUAGE_MODEL_ADAPTER_REVISION,
        ),
        gpu_memory_utilization=LANGUAGE_MODEL_GPU_MEMORY_UTILIZATION,
        maximum_model_length=QWEN_MAXIMUM_MODEL_LENGTH,
    )
