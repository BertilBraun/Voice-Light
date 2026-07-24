from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.audio_utils import write_audio_bytes_as_mono_wav
from src.backends.base import (
    PreparedTTSInput,
    TTSGenerationRequest,
    TTSGenerationResult,
    TTSVariantResult,
)


@dataclass(frozen=True)
class ElevenLabsVoiceSettings:
    stability: float
    similarity_boost: float
    style: float
    use_speaker_boost: bool
    speed: float | None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }
        if self.speed is not None:
            payload["speed"] = self.speed
        return payload


class ElevenLabsBackend:
    def __init__(self, model: str, options: dict[str, object]) -> None:
        self._model_id = model
        self._options = options
        self._api_key: str | None = None

    @property
    def backend_id(self) -> str:
        return "elevenlabs"

    @property
    def model_id(self) -> str:
        return self._model_id

    def prepare_input(self, text: str, instruction: str | None) -> PreparedTTSInput:
        return PreparedTTSInput(
            text=text,
            instruction=instruction,
            parameters={
                "instruction_supported": False,
                "instruction_strategy": "none",
                "instruction_note": (
                    "ElevenLabs receives speaker.text exactly as written. Put v3 audio tags "
                    "directly in speaker.text when desired."
                ),
                "output_format": self._output_format(),
                "voice_settings": self._voice_settings().to_payload(),
            },
        )

    def load(self) -> None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if api_key is None or api_key == "":
            raise ValueError("ELEVENLABS_API_KEY must be set for the ElevenLabs backend.")
        self._api_key = api_key

    def generate(self, request: TTSGenerationRequest) -> TTSGenerationResult:
        if self._api_key is None:
            raise ValueError("ElevenLabsBackend.load() must be called before generate().")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        variants: list[TTSVariantResult] = []
        for variant_number in range(1, request.variants + 1):
            variant_id = f"{request.filename_prefix}_{variant_number:03d}"
            variant_seed = None if request.seed is None else request.seed + variant_number - 1
            prepared_input = self.prepare_input(request.text, request.instruction)
            started_at = time.perf_counter()
            try:
                audio_path = request.output_dir / f"{variant_id}.wav"
                audio_bytes, response_metadata = self._create_speech(
                    request=request,
                    prepared_input=prepared_input,
                    seed=variant_seed,
                )
                duration_seconds, sample_rate = write_audio_bytes_as_mono_wav(
                    audio_path,
                    audio_bytes,
                )
                variants.append(
                    TTSVariantResult(
                        variant_id=variant_id,
                        audio_path=audio_path,
                        status="success",
                        prepared_input=prepared_input,
                        duration_seconds=duration_seconds,
                        generation_seconds=time.perf_counter() - started_at,
                        seed=variant_seed,
                        sample_rate=sample_rate,
                        provider_metadata=response_metadata,
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
                        provider_metadata=self._base_metadata(request),
                    )
                )
        return TTSGenerationResult(
            backend_id=self.backend_id,
            model_id=self.model_id,
            variants=variants,
        )

    def close(self) -> None:
        return

    def _create_speech(
        self,
        request: TTSGenerationRequest,
        prepared_input: PreparedTTSInput,
        seed: int | None,
    ) -> tuple[bytes, dict[str, object]]:
        voice_id = self._voice_id(request)
        payload = self._payload(prepared_input, seed)
        endpoint = (
            f"{self._api_base_url()}/v1/text-to-speech/{voice_id}"
            f"?output_format={self._output_format()}"
        )
        http_request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "xi-api-key": self._required_api_key(),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout_seconds()) as response:
                audio_bytes = response.read()
                request_id = response.headers.get("request-id")
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise ValueError(f"ElevenLabs API error {error.code}: {error_body}") from error
        metadata = self._base_metadata(request)
        metadata.update(
            {
                "voice_id": voice_id,
                "request_id": request_id,
                "output_format": self._output_format(),
                "payload": payload,
            }
        )
        return audio_bytes, metadata

    def _payload(self, prepared_input: PreparedTTSInput, seed: int | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "text": prepared_input.text,
            "model_id": self._model_id,
            "voice_settings": self._voice_settings().to_payload(),
        }
        if seed is not None:
            payload["seed"] = seed
        previous_text = self._option_string("previous_text")
        next_text = self._option_string("next_text")
        language_code = self._option_string("language_code")
        if previous_text is not None:
            payload["previous_text"] = previous_text
        if next_text is not None:
            payload["next_text"] = next_text
        if language_code is not None:
            payload["language_code"] = language_code
        return payload

    def _voice_id(self, request: TTSGenerationRequest) -> str:
        if request.speaker_id is not None:
            return request.speaker_id
        speaker = request.metadata.get("speaker")
        if speaker == "speaker_a":
            voice_id = self._option_string("speaker_a_voice_id")
            if voice_id is not None:
                return voice_id
        if speaker == "speaker_b":
            voice_id = self._option_string("speaker_b_voice_id")
            if voice_id is not None:
                return voice_id
        voice_id = self._option_string("voice_id")
        if voice_id is None:
            raise ValueError(
                "ElevenLabs requires voice_id, speaker_a_voice_id/speaker_b_voice_id, "
                "or request.speaker_id."
            )
        return voice_id

    def _voice_settings(self) -> ElevenLabsVoiceSettings:
        return ElevenLabsVoiceSettings(
            stability=self._option_float("stability", 0.4),
            similarity_boost=self._option_float("similarity_boost", 0.75),
            style=self._option_float("style", 0.3),
            use_speaker_boost=self._option_bool("use_speaker_boost", True),
            speed=self._option_optional_float("speed"),
        )

    def _base_metadata(self, request: TTSGenerationRequest) -> dict[str, object]:
        return {
            "model_id": self._model_id,
            "speaker": request.metadata.get("speaker", ""),
            "api_base_url": self._api_base_url(),
        }

    def _required_api_key(self) -> str:
        assert self._api_key is not None
        return self._api_key

    def _api_base_url(self) -> str:
        return str(self._options.get("api_base_url", "https://api.elevenlabs.io"))

    def _output_format(self) -> str:
        return str(self._options.get("output_format", "mp3_44100_128"))

    def _timeout_seconds(self) -> float:
        return self._option_float("timeout_seconds", 120.0)

    def _option_string(self, name: str) -> str | None:
        value = self._options.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"Option {name} must be a string.")
        if value == "":
            return None
        return value

    def _option_float(self, name: str, default: float) -> float:
        value = self._options.get(name, default)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            return float(value)
        raise ValueError(f"Option {name} must be a number.")

    def _option_optional_float(self, name: str) -> float | None:
        value = self._options.get(name)
        if value is None:
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            return float(value)
        raise ValueError(f"Option {name} must be a number.")

    def _option_bool(self, name: str, default: bool) -> bool:
        value = self._options.get(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        raise ValueError(f"Option {name} must be a boolean.")
