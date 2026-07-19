from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.compute.voice.interfaces import SpeechSynthesizer
from app.compute.voice.kyutai_tts import KyutaiSpeechSynthesizer
from app.compute.voice.voxtream_speaking_rate import FinalPhraseSlowdown
from app.compute.voice.voxtream_tts import VoxtreamSpeechSynthesizer


class SpeechSynthesisBackend(StrEnum):
    KYUTAI = "kyutai"
    VOXTREAM = "voxtream"


@dataclass(frozen=True)
class SpeechSynthesisSettings:
    backend: SpeechSynthesisBackend
    voxtream_python_path: Path
    voxtream_config_path: Path
    voxtream_speaking_rate_config_path: Path
    voxtream_prompt_audio_path: Path
    voxtream_compile: bool
    voxtream_prompt_memory_cache: bool
    voxtream_final_phrase_slowdown: FinalPhraseSlowdown | None

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        repository_root: Path,
    ) -> SpeechSynthesisSettings:
        backend_value = environment.get(
            "VOICE_LIGHT_TTS_BACKEND",
            SpeechSynthesisBackend.KYUTAI,
        )
        try:
            backend = SpeechSynthesisBackend(backend_value)
        except ValueError as error:
            supported_backends = ", ".join(backend.value for backend in SpeechSynthesisBackend)
            raise ValueError(
                f"VOICE_LIGHT_TTS_BACKEND must be one of: {supported_backends}."
            ) from error

        voxtream_root = repository_root / ".cache" / "compute" / "voxtream"
        voxtream_source_root = voxtream_root / "source"
        return cls(
            backend=backend,
            voxtream_python_path=Path(
                environment.get(
                    "VOICE_LIGHT_VOXTREAM_PYTHON_PATH",
                    voxtream_root / ".venv" / "bin" / "python",
                )
            ),
            voxtream_config_path=Path(
                environment.get(
                    "VOICE_LIGHT_VOXTREAM_CONFIG_PATH",
                    voxtream_source_root / "configs" / "generator.json",
                )
            ),
            voxtream_speaking_rate_config_path=Path(
                environment.get(
                    "VOICE_LIGHT_VOXTREAM_SPEAKING_RATE_CONFIG_PATH",
                    voxtream_source_root / "configs" / "speaking_rate.json",
                )
            ),
            voxtream_prompt_audio_path=Path(
                environment.get(
                    "VOICE_LIGHT_VOXTREAM_PROMPT_AUDIO_PATH",
                    voxtream_source_root / "assets" / "audio" / "english_female.wav",
                )
            ),
            voxtream_compile=_read_boolean(
                environment,
                "VOICE_LIGHT_VOXTREAM_COMPILE",
                default=True,
            ),
            voxtream_prompt_memory_cache=_read_boolean(
                environment,
                "VOICE_LIGHT_VOXTREAM_PROMPT_MEMORY_CACHE",
                default=True,
            ),
            voxtream_final_phrase_slowdown=_read_final_phrase_slowdown(environment),
        )


def create_speech_synthesizer(settings: SpeechSynthesisSettings) -> SpeechSynthesizer:
    match settings.backend:
        case SpeechSynthesisBackend.KYUTAI:
            return KyutaiSpeechSynthesizer()
        case SpeechSynthesisBackend.VOXTREAM:
            _require_file(settings.voxtream_python_path, "VoXtream Python")
            _require_file(settings.voxtream_config_path, "VoXtream configuration")
            _require_file(settings.voxtream_prompt_audio_path, "VoXtream prompt audio")
            if settings.voxtream_final_phrase_slowdown is not None:
                _require_file(
                    settings.voxtream_speaking_rate_config_path,
                    "VoXtream speaking-rate configuration",
                )
            return VoxtreamSpeechSynthesizer(
                python_path=settings.voxtream_python_path,
                config_path=settings.voxtream_config_path,
                prompt_audio_path=settings.voxtream_prompt_audio_path,
                compile_model=settings.voxtream_compile,
                cache_prompt_in_memory=settings.voxtream_prompt_memory_cache,
                speaking_rate_config_path=(
                    settings.voxtream_speaking_rate_config_path
                    if settings.voxtream_final_phrase_slowdown is not None
                    else None
                ),
                final_phrase_slowdown=settings.voxtream_final_phrase_slowdown,
            )


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")


def _read_boolean(
    environment: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    match value.lower():
        case "1" | "true":
            return True
        case "0" | "false":
            return False
        case _:
            raise ValueError(f"{name} must be true or false.")


def _read_final_phrase_slowdown(
    environment: Mapping[str, str],
) -> FinalPhraseSlowdown | None:
    rate_name = "VOICE_LIGHT_VOXTREAM_FINAL_SLOWDOWN_SYLLABLES_PER_SECOND"
    rate_value = environment.get(rate_name)
    if rate_value is None or not rate_value.strip():
        return None
    try:
        syllables_per_second = float(rate_value)
    except ValueError as error:
        raise ValueError(f"{rate_name} must be a number.") from error

    word_count_name = "VOICE_LIGHT_VOXTREAM_FINAL_SLOWDOWN_WORD_COUNT"
    word_count_value = environment.get(word_count_name, "4")
    try:
        word_count = int(word_count_value)
    except ValueError as error:
        raise ValueError(f"{word_count_name} must be an integer.") from error
    return FinalPhraseSlowdown(
        syllables_per_second=syllables_per_second,
        word_count=word_count,
    )
