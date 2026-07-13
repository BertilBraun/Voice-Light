from __future__ import annotations

import pytest
import torch

from app.compute.asr.models.base import cuda_device


def test_cuda_device_fails_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available for ASR inference."):
        cuda_device()


def test_cuda_device_returns_cuda_when_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert cuda_device() == "cuda"
