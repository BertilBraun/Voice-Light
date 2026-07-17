from __future__ import annotations

import pytest

from app.compute.config import ComputeSettings


def test_compute_settings_disable_voice_stack_without_parsing_tts_settings() -> None:
    settings = ComputeSettings.from_environment(
        {
            "VOICE_LIGHT_COMPUTE_TOKEN": "secret-token",
            "VOICE_LIGHT_VOICE_STACK_ENABLED": "false",
            "VOICE_LIGHT_TTS_BACKEND": "invalid-and-irrelevant",
        }
    )

    assert settings.voice_stack is None


def test_compute_settings_enable_voice_stack_by_default() -> None:
    settings = ComputeSettings.from_environment({"VOICE_LIGHT_COMPUTE_TOKEN": "secret-token"})

    assert settings.voice_stack is not None


def test_compute_settings_reject_invalid_voice_stack_setting() -> None:
    with pytest.raises(ValueError, match="VOICE_LIGHT_VOICE_STACK_ENABLED"):
        ComputeSettings.from_environment(
            {
                "VOICE_LIGHT_COMPUTE_TOKEN": "secret-token",
                "VOICE_LIGHT_VOICE_STACK_ENABLED": "sometimes",
            }
        )
