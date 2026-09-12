from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class CudaWorkerDevice:
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("A CUDA worker device index cannot be negative.")

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        variable_name: str,
    ) -> Self | None:
        value = environment.get(variable_name)
        if value is None:
            return None
        try:
            return cls(index=int(value))
        except ValueError as error:
            raise ValueError(f"{variable_name} must be a non-negative integer.") from error

    def select(self, environment: Mapping[str, str]) -> dict[str, str]:
        selected_environment = dict(environment)
        selected_environment["CUDA_VISIBLE_DEVICES"] = str(self.index)
        return selected_environment


def select_cuda_worker_environment(
    environment: Mapping[str, str],
    device: CudaWorkerDevice | None,
) -> dict[str, str]:
    if device is None:
        return dict(environment)
    return device.select(environment)
