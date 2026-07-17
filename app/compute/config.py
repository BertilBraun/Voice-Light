from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.compute.voice.tts_selection import SpeechSynthesisSettings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VoiceStackSettings:
    speech_synthesis: SpeechSynthesisSettings


@dataclass(frozen=True)
class ComputeSettings:
    token: str
    log_directory: Path
    voice_stack: VoiceStackSettings | None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> ComputeSettings:
        token = environment.get("VOICE_LIGHT_COMPUTE_TOKEN", "")
        if not token:
            raise ValueError("VOICE_LIGHT_COMPUTE_TOKEN is required.")
        log_directory = Path(environment.get("VOICE_LIGHT_COMPUTE_LOG_DIR", "logs/compute"))
        voice_stack_enabled = _parse_boolean_setting(
            environment=environment,
            name="VOICE_LIGHT_VOICE_STACK_ENABLED",
            default=True,
        )
        return cls(
            token=token,
            log_directory=log_directory,
            voice_stack=(
                VoiceStackSettings(
                    speech_synthesis=SpeechSynthesisSettings.from_environment(
                        environment,
                        REPOSITORY_ROOT,
                    )
                )
                if voice_stack_enabled
                else None
            ),
        )


def _parse_boolean_setting(
    environment: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    match value.strip().lower():
        case "true":
            return True
        case "false":
            return False
        case _:
            raise ValueError(f"{name} must be either 'true' or 'false'.")
