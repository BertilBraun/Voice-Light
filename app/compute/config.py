from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComputeSettings:
    token: str
    log_directory: Path

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> ComputeSettings:
        token = environment.get("VOICE_LIGHT_COMPUTE_TOKEN", "")
        if not token:
            raise ValueError("VOICE_LIGHT_COMPUTE_TOKEN is required.")
        log_directory = Path(environment.get("VOICE_LIGHT_COMPUTE_LOG_DIR", "logs/compute"))
        return cls(token=token, log_directory=log_directory)
