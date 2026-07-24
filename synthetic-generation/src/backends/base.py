from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PreparedTTSInput:
    text: str
    instruction: str | None
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSGenerationRequest:
    text: str
    instruction: str | None
    variants: int
    output_dir: Path
    language: str | None = None
    speaker_id: str | None = None
    seed: int | None = None
    filename_prefix: str = "variant"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSVariantResult:
    variant_id: str
    audio_path: Path | None
    status: str
    prepared_input: PreparedTTSInput
    duration_seconds: float | None = None
    generation_seconds: float | None = None
    seed: int | None = None
    sample_rate: int | None = None
    error: str | None = None
    provider_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSGenerationResult:
    backend_id: str
    model_id: str
    variants: list[TTSVariantResult]


class TTSBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def prepare_input(self, text: str, instruction: str | None) -> PreparedTTSInput: ...

    def load(self) -> None: ...

    def generate(self, request: TTSGenerationRequest) -> TTSGenerationResult: ...

    def close(self) -> None: ...
