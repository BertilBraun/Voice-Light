from __future__ import annotations

import pytest

from app.compute.voice.model_constants import (
    LANGUAGE_MODEL_ADAPTER_NAME,
    LANGUAGE_MODEL_ADAPTER_REVISION,
    LANGUAGE_MODEL_NAME,
    LANGUAGE_MODEL_REVISION,
)
from deployment.compute import validate_environment


def test_compute_environment_validator_selects_default_voice_models() -> None:
    language_model = validate_environment.language_model_configuration_from_environment({})

    assert language_model.model_name == LANGUAGE_MODEL_NAME
    assert language_model.model_revision == LANGUAGE_MODEL_REVISION
    assert language_model.adapter is not None
    assert (
        language_model.adapter.repository_id
        == LANGUAGE_MODEL_ADAPTER_NAME
        == "BertilBraun/qwen3-1.7b-voice-light-tool-use-lora"
    )
    assert language_model.adapter.revision == LANGUAGE_MODEL_ADAPTER_REVISION
    assert validate_environment.SEARCH_SUMMARIZER_MODEL_NAME == "Qwen/Qwen3-0.6B"
    assert validate_environment.SEARCH_SUMMARIZER_MODEL_REVISION
    assert validate_environment.KYUTAI_TTS_MODEL_NAME
    assert validate_environment.KYUTAI_TTS_MODEL_REVISION
    assert validate_environment.NEMOTRON_ASR_MODEL_NAME
    assert validate_environment.NEMOTRON_ASR_MODEL_REVISION


def test_compute_environment_accepts_capable_gpu() -> None:
    validate_environment.validate_gpu_properties(
        device_name="NVIDIA GeForce RTX 3060",
        memory_bytes=12 * 1024**3,
        cuda_version="12.8",
    )


def test_asr_compute_environment_accepts_cuda_126_runtime() -> None:
    validate_environment.validate_gpu_properties(
        device_name="NVIDIA GeForce RTX 3060",
        memory_bytes=12 * 1024**3,
        cuda_version="12.6",
        mode="asr",
    )


def test_asr_compute_environment_rejects_incompatible_cuda_runtime() -> None:
    with pytest.raises(RuntimeError, match="require the CUDA 12.6"):
        validate_environment.validate_gpu_properties(
            device_name="NVIDIA GeForce RTX 3060",
            memory_bytes=12 * 1024**3,
            cuda_version="12.8",
            mode="asr",
        )


@pytest.mark.parametrize(
    ("memory_bytes", "cuda_version", "message"),
    [
        (8 * 1024**3, "12.8", "at least 10 GiB"),
        (12 * 1024**3, None, "no CUDA runtime"),
    ],
)
def test_compute_environment_rejects_incapable_gpu(
    memory_bytes: int,
    cuda_version: str | None,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_environment.validate_gpu_properties(
            device_name="Test GPU",
            memory_bytes=memory_bytes,
            cuda_version=cuda_version,
        )


def test_asr_only_disk_preflight_accepts_sufficient_space(monkeypatch: pytest.MonkeyPatch) -> None:
    class DiskUsage:
        free = 13 * 1024**3

    monkeypatch.setattr(validate_environment.shutil, "disk_usage", lambda _path: DiskUsage())

    validate_environment.validate_asr_only_disk_space(validate_environment.REPOSITORY_ROOT)


def test_asr_only_disk_preflight_rejects_insufficient_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DiskUsage:
        free = 11 * 1024**3

    monkeypatch.setattr(validate_environment.shutil, "disk_usage", lambda _path: DiskUsage())

    with pytest.raises(RuntimeError, match="at least 12 GiB"):
        validate_environment.validate_asr_only_disk_space(validate_environment.REPOSITORY_ROOT)
