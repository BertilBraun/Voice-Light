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
