from __future__ import annotations

import time

from src.audio_utils import audio_duration_seconds, audio_sample_rate, write_test_tone
from src.backends.base import (
    PreparedTTSInput,
    TTSGenerationRequest,
    TTSGenerationResult,
    TTSVariantResult,
)


class FakeBackend:
    def __init__(self, model: str, options: dict[str, object]) -> None:
        self._model_id = model
        self._options = options
        self._loaded = False

    @property
    def backend_id(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return self._model_id

    def prepare_input(self, text: str, instruction: str | None) -> PreparedTTSInput:
        return PreparedTTSInput(
            text=text,
            instruction=instruction,
            parameters={"instruction_supported": True, "options": self._options},
        )

    def load(self) -> None:
        self._loaded = True

    def generate(self, request: TTSGenerationRequest) -> TTSGenerationResult:
        if not self._loaded:
            raise ValueError("FakeBackend.load() must be called before generate().")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        variants: list[TTSVariantResult] = []
        fail_variant = self._options.get("fail_variant")
        prepared_input = self.prepare_input(request.text, request.instruction)
        for variant_number in range(1, request.variants + 1):
            variant_id = f"{request.filename_prefix}_{variant_number:03d}"
            variant_seed = None if request.seed is None else request.seed + variant_number - 1
            started_at = time.perf_counter()
            if fail_variant == variant_number or fail_variant == str(variant_number):
                variants.append(
                    TTSVariantResult(
                        variant_id=variant_id,
                        audio_path=None,
                        status="failed",
                        prepared_input=prepared_input,
                        generation_seconds=time.perf_counter() - started_at,
                        seed=variant_seed,
                        error="Simulated fake backend failure.",
                        provider_metadata={"simulated": True},
                    )
                )
                continue
            audio_path = request.output_dir / f"{variant_id}.wav"
            frequency_hz = 220.0 + 30.0 * variant_number
            write_test_tone(audio_path, frequency_hz, 0.35, 16_000)
            generation_seconds = time.perf_counter() - started_at
            variants.append(
                TTSVariantResult(
                    variant_id=variant_id,
                    audio_path=audio_path,
                    status="success",
                    prepared_input=prepared_input,
                    duration_seconds=audio_duration_seconds(audio_path),
                    generation_seconds=generation_seconds,
                    seed=variant_seed,
                    sample_rate=audio_sample_rate(audio_path),
                    provider_metadata={"frequency_hz": frequency_hz, "simulated": True},
                )
            )
        return TTSGenerationResult(
            backend_id=self.backend_id,
            model_id=self.model_id,
            variants=variants,
        )

    def close(self) -> None:
        self._loaded = False
