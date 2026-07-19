from __future__ import annotations

from dataclasses import dataclass


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
