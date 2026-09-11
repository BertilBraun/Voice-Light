from __future__ import annotations

import pytest

from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_ADAPTER_NAME,
    LANGUAGE_MODEL_ADAPTER_REVISION,
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
)
from app.compute.voice.qwen_config import (
    MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE,
    MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE,
    QWEN_BACKEND_ENVIRONMENT_VARIABLE,
    QWEN_ENFORCE_EAGER_ENVIRONMENT_VARIABLE,
    QwenAdapterConfiguration,
    QwenBackend,
    QwenSamplingConfiguration,
    language_model_configuration_from_environment,
)


def test_language_model_configuration_defaults_to_pinned_dynamic_adapter() -> None:
    configuration = language_model_configuration_from_environment({})

    assert configuration.model_name == LANGUAGE_MODEL_NAME
    assert configuration.model_revision == LANGUAGE_MODEL_REVISION
    assert configuration.adapter == QwenAdapterConfiguration(
        repository_id=LANGUAGE_MODEL_ADAPTER_NAME,
        revision=LANGUAGE_MODEL_ADAPTER_REVISION,
    )
    assert configuration.sampling == QwenSamplingConfiguration(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"temperature": 0.0},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"top_k": 0},
    ],
)
def test_qwen_sampling_configuration_rejects_invalid_values(
    arguments: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        QwenSamplingConfiguration(**arguments)


def test_language_model_configuration_selects_pinned_merged_checkpoint() -> None:
    configuration = language_model_configuration_from_environment(
        {
            MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE: (
                "BertilBraun/qwen3-1.7b-voice-light-tool-use-merged"
            ),
            MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE: "0" * 40,
        }
    )

    assert configuration.model_name == "BertilBraun/qwen3-1.7b-voice-light-tool-use-merged"
    assert configuration.model_revision == "0" * 40
    assert configuration.adapter is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("TRUE", True), ("false", False), ("FALSE", False)],
)
def test_language_model_configuration_parses_enforce_eager(
    value: str,
    expected: bool,
) -> None:
    configuration = language_model_configuration_from_environment(
        {QWEN_ENFORCE_EAGER_ENVIRONMENT_VARIABLE: value}
    )

    assert configuration.enforce_eager is expected


def test_language_model_configuration_rejects_invalid_enforce_eager() -> None:
    with pytest.raises(ValueError, match=QWEN_ENFORCE_EAGER_ENVIRONMENT_VARIABLE):
        language_model_configuration_from_environment(
            {QWEN_ENFORCE_EAGER_ENVIRONMENT_VARIABLE: "yes"}
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("vllm", QwenBackend.VLLM), ("TRANSFORMERS", QwenBackend.TRANSFORMERS)],
)
def test_language_model_configuration_parses_backend(
    value: str,
    expected: QwenBackend,
) -> None:
    configuration = language_model_configuration_from_environment(
        {QWEN_BACKEND_ENVIRONMENT_VARIABLE: value}
    )

    assert configuration.backend is expected


def test_language_model_configuration_rejects_invalid_backend() -> None:
    with pytest.raises(ValueError, match=QWEN_BACKEND_ENVIRONMENT_VARIABLE):
        language_model_configuration_from_environment(
            {QWEN_BACKEND_ENVIRONMENT_VARIABLE: "unknown"}
        )


@pytest.mark.parametrize(
    "environment",
    [
        {MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE: "owner/model"},
        {MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE: "revision"},
        {
            MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE: "",
            MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE: "revision",
        },
        {
            MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE: "owner/model",
            MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE: "",
        },
        {
            MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE: "owner/model",
            MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE: "main",
        },
        {
            MERGED_LANGUAGE_MODEL_NAME_ENVIRONMENT_VARIABLE: "local-path",
            MERGED_LANGUAGE_MODEL_REVISION_ENVIRONMENT_VARIABLE: "0" * 40,
        },
    ],
)
def test_language_model_configuration_rejects_incomplete_merged_checkpoint(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        language_model_configuration_from_environment(environment)
