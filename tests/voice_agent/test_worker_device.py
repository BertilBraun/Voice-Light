from __future__ import annotations

import pytest

from app.compute.voice.worker_device import (
    CudaWorkerDevice,
    select_cuda_worker_environment,
)


def test_cuda_worker_device_selects_one_physical_device() -> None:
    environment = select_cuda_worker_environment(
        {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0,1"},
        CudaWorkerDevice(index=1),
    )

    assert environment == {
        "PATH": "/usr/bin",
        "CUDA_VISIBLE_DEVICES": "1",
    }


def test_cuda_worker_device_reads_optional_environment_selection() -> None:
    assert CudaWorkerDevice.from_environment({}, "VOICE_LIGHT_TTS_CUDA_DEVICE") is None
    assert CudaWorkerDevice.from_environment(
        {"VOICE_LIGHT_TTS_CUDA_DEVICE": "1"},
        "VOICE_LIGHT_TTS_CUDA_DEVICE",
    ) == CudaWorkerDevice(index=1)


@pytest.mark.parametrize("value", ("-1", "gpu-one"))
def test_cuda_worker_device_rejects_invalid_environment_selection(value: str) -> None:
    with pytest.raises(ValueError, match="VOICE_LIGHT_TTS_CUDA_DEVICE"):
        CudaWorkerDevice.from_environment(
            {"VOICE_LIGHT_TTS_CUDA_DEVICE": value},
            "VOICE_LIGHT_TTS_CUDA_DEVICE",
        )
