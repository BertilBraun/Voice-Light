from __future__ import annotations

import gc
import time

import numpy as np

from src.audio_utils import audio_duration_seconds, audio_sample_rate, write_mono_wav
from src.backends.base import (
    PreparedTTSInput,
    TTSGenerationRequest,
    TTSGenerationResult,
    TTSVariantResult,
)


class QwenVoiceDesignBackend:
    def __init__(self, model: str, options: dict[str, object]) -> None:
        self._model_id = model
        self._options = options
        self._model: object | None = None
        self._torch_module: object | None = None
        self._resolved_dtype = "auto"
        self._resolved_device = "auto"
        self._flash_attention_enabled = False

    @property
    def backend_id(self) -> str:
        return "qwen-voice-design"

    @property
    def model_id(self) -> str:
        return self._model_id

    def prepare_input(self, text: str, instruction: str | None) -> PreparedTTSInput:
        return PreparedTTSInput(
            text=text,
            instruction=instruction,
            parameters={
                "language": self._language(),
                "instruct": instruction,
                "instruction_supported": True,
            },
        )

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self._torch_module = torch
        if not torch.cuda.is_available():
            raise ValueError("Qwen VoiceDesign backend requires CUDA, but CUDA is unavailable.")

        self._resolved_device = str(self._options.get("device", "cuda:0"))
        requested_dtype = str(self._options.get("dtype", self._default_dtype(torch)))
        dtype = self._resolve_dtype(requested_dtype, torch)
        self._resolved_dtype = requested_dtype
        use_flash_attention = self._option_bool("flash_attention", True)

        model_kwargs: dict[str, object] = {
            "device_map": self._resolved_device,
            "dtype": dtype,
        }
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        try:
            self._model = Qwen3TTSModel.from_pretrained(self._model_id, **model_kwargs)
            self._flash_attention_enabled = use_flash_attention
        except Exception:
            if not use_flash_attention:
                raise
            model_kwargs.pop("attn_implementation")
            self._model = Qwen3TTSModel.from_pretrained(self._model_id, **model_kwargs)
            self._flash_attention_enabled = False

    def generate(self, request: TTSGenerationRequest) -> TTSGenerationResult:
        if self._model is None:
            raise ValueError("QwenVoiceDesignBackend.load() must be called before generate().")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        variants: list[TTSVariantResult] = []
        for variant_number in range(1, request.variants + 1):
            variant_id = f"{request.filename_prefix}_{variant_number:03d}"
            variant_seed = None if request.seed is None else request.seed + variant_number - 1
            prepared_input = self.prepare_input(request.text, request.instruction)
            started_at = time.perf_counter()
            try:
                self._set_seed(variant_seed)
                audio_path = request.output_dir / f"{variant_id}.wav"
                audio_samples, sample_rate = self._generate_one(prepared_input)
                write_mono_wav(audio_path, audio_samples, sample_rate)
                duration_seconds = audio_duration_seconds(audio_path)
                generation_seconds = time.perf_counter() - started_at
                variants.append(
                    TTSVariantResult(
                        variant_id=variant_id,
                        audio_path=audio_path,
                        status="success",
                        prepared_input=prepared_input,
                        duration_seconds=duration_seconds,
                        generation_seconds=generation_seconds,
                        seed=variant_seed,
                        sample_rate=audio_sample_rate(audio_path),
                        provider_metadata=self._provider_metadata(),
                    )
                )
            except Exception as error:
                variants.append(
                    TTSVariantResult(
                        variant_id=variant_id,
                        audio_path=None,
                        status="failed",
                        prepared_input=prepared_input,
                        generation_seconds=time.perf_counter() - started_at,
                        seed=variant_seed,
                        error=str(error),
                        provider_metadata=self._provider_metadata(),
                    )
                )
        return TTSGenerationResult(
            backend_id=self.backend_id,
            model_id=self.model_id,
            variants=variants,
        )

    def close(self) -> None:
        self._model = None
        gc.collect()
        torch_module = self._torch_module
        if torch_module is not None:
            cuda_module = torch_module.cuda
            if cuda_module.is_available():
                cuda_module.empty_cache()

    def _generate_one(self, prepared_input: PreparedTTSInput) -> tuple[np.ndarray, int]:
        assert self._model is not None
        wavs, sample_rate = self._model.generate_voice_design(
            text=prepared_input.text,
            language=self._language(),
            instruct=prepared_input.instruction,
        )
        if len(wavs) == 0:
            raise ValueError("Qwen returned no waveforms.")
        return np.asarray(wavs[0], dtype=np.float32), int(sample_rate)

    def _resolve_dtype(self, requested_dtype: str, torch_module: object) -> object:
        match requested_dtype:
            case "bfloat16":
                return torch_module.bfloat16
            case "float16":
                return torch_module.float16
            case "float32":
                return torch_module.float32
            case _:
                raise ValueError("Unsupported Qwen dtype. Expected bfloat16, float16, or float32.")

    def _default_dtype(self, torch_module: object) -> str:
        if torch_module.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        torch_module = self._torch_module
        assert torch_module is not None
        torch_module.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)

    def _language(self) -> str:
        return str(self._options.get("language", "English"))

    def _option_bool(self, name: str, default: bool) -> bool:
        option_value = self._options.get(name, default)
        if isinstance(option_value, bool):
            return option_value
        if isinstance(option_value, str):
            return option_value.lower() in {"1", "true", "yes", "on"}
        raise ValueError(f"Option {name} must be a boolean.")

    def _provider_metadata(self) -> dict[str, object]:
        return {
            "device": self._resolved_device,
            "dtype": self._resolved_dtype,
            "flash_attention": self._flash_attention_enabled,
            "language": self._language(),
        }
