from __future__ import annotations

import pytest

from deployment.compute import validate_environment


def test_compute_environment_validator_imports_voice_model_constants() -> None:
    assert validate_environment.LANGUAGE_MODEL_NAME == "Qwen/Qwen3-1.7B"
    assert validate_environment.LANGUAGE_MODEL_REVISION
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
