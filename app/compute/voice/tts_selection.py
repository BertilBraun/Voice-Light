from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.compute.voice.interfaces import SpeechSynthesizer
from app.compute.voice.kyutai_tts import KyutaiSpeechSynthesizer
from app.compute.voice.voxtream_tts import VoxtreamSpeechSynthesizer


class SpeechSynthesisBackend(StrEnum):
    KYUTAI = "kyutai"
    VOXTREAM = "voxtream"


@dataclass(frozen=True)
class SpeechSynthesisSettings:
    backend: SpeechSynthesisBackend
    voxtream_python_path: Path
    voxtream_config_path: Path
    voxtream_prompt_audio_path: Path

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
            voxtream_prompt_audio_path=Path(
                environment.get(
                    "VOICE_LIGHT_VOXTREAM_PROMPT_AUDIO_PATH",
                    voxtream_source_root / "assets" / "audio" / "english_female.wav",
                )
            ),
        )


def create_speech_synthesizer(settings: SpeechSynthesisSettings) -> SpeechSynthesizer:
    match settings.backend:
        case SpeechSynthesisBackend.KYUTAI:
            return KyutaiSpeechSynthesizer()
        case SpeechSynthesisBackend.VOXTREAM:
            _require_file(settings.voxtream_python_path, "VoXtream Python")
            _require_file(settings.voxtream_config_path, "VoXtream configuration")
            _require_file(settings.voxtream_prompt_audio_path, "VoXtream prompt audio")
            return VoxtreamSpeechSynthesizer(
                python_path=settings.voxtream_python_path,
                config_path=settings.voxtream_config_path,
                prompt_audio_path=settings.voxtream_prompt_audio_path,
            )


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")
