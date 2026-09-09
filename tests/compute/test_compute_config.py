from __future__ import annotations

import pytest

from app.compute.config import ComputeSettings
from app.compute.voice.search import (
    TAVILY_SEARCH_API_KEY_ENVIRONMENT_VARIABLE,
    ConfiguredTavilySearchSettings,
    UnconfiguredSearchSettings,
)


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
    assert settings.voice_stack.search == UnconfiguredSearchSettings()
    assert settings.eager_model_loading is False


def test_compute_settings_can_eagerly_load_models() -> None:
    settings = ComputeSettings.from_environment(
        {
            "VOICE_LIGHT_COMPUTE_TOKEN": "secret-token",
            "VOICE_LIGHT_EAGER_MODEL_LOADING": "true",
        }
    )

    assert settings.eager_model_loading is True


def test_compute_settings_load_optional_tavily_search_key() -> None:
    settings = ComputeSettings.from_environment(
        {
            "VOICE_LIGHT_COMPUTE_TOKEN": "secret-token",
            TAVILY_SEARCH_API_KEY_ENVIRONMENT_VARIABLE: "tavily-key",
        }
    )

    assert settings.voice_stack is not None
    assert settings.voice_stack.search == ConfiguredTavilySearchSettings(api_key="tavily-key")


def test_compute_settings_reject_invalid_voice_stack_setting() -> None:
    with pytest.raises(ValueError, match="VOICE_LIGHT_VOICE_STACK_ENABLED"):
        ComputeSettings.from_environment(
            {
                "VOICE_LIGHT_COMPUTE_TOKEN": "secret-token",
                "VOICE_LIGHT_VOICE_STACK_ENABLED": "sometimes",
            }
        )


def test_compute_settings_reject_invalid_eager_loading_setting() -> None:
    with pytest.raises(ValueError, match="VOICE_LIGHT_EAGER_MODEL_LOADING"):
        ComputeSettings.from_environment(
            {
                "VOICE_LIGHT_COMPUTE_TOKEN": "secret-token",
                "VOICE_LIGHT_EAGER_MODEL_LOADING": "sometimes",
            }
        )
