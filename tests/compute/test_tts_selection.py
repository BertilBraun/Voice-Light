from __future__ import annotations

from pathlib import Path
from typing import Self

import pytest

from app.compute.voice.interfaces import SpeechSynthesisSession
from app.compute.voice.kyutai_tts import KyutaiSpeechSynthesizer
from app.compute.voice.tts_selection import (
    SpeechSynthesisBackend,
    SpeechSynthesisSettings,
    create_speech_synthesizer,
)
from app.compute.voice.voxtream_speaking_rate import FinalPhraseSlowdown


class FakeSpeechSynthesizer:
    @property
    def sample_rate(self) -> int:
        return 24_000

    def start_session(self) -> SpeechSynthesisSession:
        raise AssertionError("The selection test must not start synthesis.")

    def close(self) -> None:
        return

    @classmethod
    def create(cls) -> Self:
        return cls()


def test_speech_synthesis_settings_default_to_kyutai(tmp_path: Path) -> None:
    settings = SpeechSynthesisSettings.from_environment({}, tmp_path)

    assert settings.backend is SpeechSynthesisBackend.KYUTAI
    assert settings.voxtream_python_path == (
        tmp_path / ".cache" / "compute" / "voxtream" / ".venv" / "bin" / "python"
    )
    assert settings.voxtream_compile
    assert settings.voxtream_prompt_memory_cache
    assert settings.voxtream_final_phrase_slowdown is None
    assert settings.voxtream_speaking_rate_config_path == (
        tmp_path / ".cache" / "compute" / "voxtream" / "source" / "configs" / "speaking_rate.json"
    )


def test_speech_synthesis_settings_allow_disabling_voxtream_optimizations(
    tmp_path: Path,
) -> None:
    settings = SpeechSynthesisSettings.from_environment(
        {
            "VOICE_LIGHT_VOXTREAM_COMPILE": "false",
            "VOICE_LIGHT_VOXTREAM_PROMPT_MEMORY_CACHE": "0",
        },
        tmp_path,
    )

    assert not settings.voxtream_compile
    assert not settings.voxtream_prompt_memory_cache


@pytest.mark.parametrize(
    "environment_name",
    (
        "VOICE_LIGHT_VOXTREAM_COMPILE",
        "VOICE_LIGHT_VOXTREAM_PROMPT_MEMORY_CACHE",
    ),
)
def test_speech_synthesis_settings_reject_invalid_voxtream_booleans(
    environment_name: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=environment_name):
        SpeechSynthesisSettings.from_environment(
            {environment_name: "sometimes"},
            tmp_path,
        )


def test_speech_synthesis_settings_reject_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="VOICE_LIGHT_TTS_BACKEND"):
        SpeechSynthesisSettings.from_environment(
            {"VOICE_LIGHT_TTS_BACKEND": "unknown"},
            tmp_path,
        )


def test_speech_synthesis_settings_enable_final_phrase_slowdown(tmp_path: Path) -> None:
    settings = SpeechSynthesisSettings.from_environment(
        {
            "VOICE_LIGHT_VOXTREAM_FINAL_SLOWDOWN_SYLLABLES_PER_SECOND": "3.0",
            "VOICE_LIGHT_VOXTREAM_FINAL_SLOWDOWN_WORD_COUNT": "5",
        },
        tmp_path,
    )

    assert settings.voxtream_final_phrase_slowdown == FinalPhraseSlowdown(
        syllables_per_second=3.0,
        word_count=5,
    )


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        (
            {"VOICE_LIGHT_VOXTREAM_FINAL_SLOWDOWN_SYLLABLES_PER_SECOND": "slow"},
            "must be a number",
        ),
        (
            {
                "VOICE_LIGHT_VOXTREAM_FINAL_SLOWDOWN_SYLLABLES_PER_SECOND": "3",
                "VOICE_LIGHT_VOXTREAM_FINAL_SLOWDOWN_WORD_COUNT": "several",
            },
            "must be an integer",
        ),
        (
            {"VOICE_LIGHT_VOXTREAM_FINAL_SLOWDOWN_SYLLABLES_PER_SECOND": "0"},
            "speaking rate must be positive",
        ),
    ),
)
def test_speech_synthesis_settings_reject_invalid_final_phrase_slowdown(
    environment: dict[str, str],
    message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpeechSynthesisSettings.from_environment(environment, tmp_path)


def test_kyutai_selection_uses_existing_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_synthesizer = FakeSpeechSynthesizer.create()
    monkeypatch.setattr(
        "app.compute.voice.tts_selection.KyutaiSpeechSynthesizer",
        lambda: expected_synthesizer,
    )
    settings = SpeechSynthesisSettings.from_environment(
        {"VOICE_LIGHT_TTS_BACKEND": "kyutai"},
        tmp_path,
    )

    assert create_speech_synthesizer(settings) is expected_synthesizer


def test_voxtream_selection_requires_installed_files(tmp_path: Path) -> None:
    settings = SpeechSynthesisSettings.from_environment(
        {"VOICE_LIGHT_TTS_BACKEND": "voxtream"},
        tmp_path,
    )

    with pytest.raises(ValueError, match="VoXtream Python"):
        create_speech_synthesizer(settings)


def test_voxtream_selection_passes_isolated_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python_path = tmp_path / "python"
    config_path = tmp_path / "generator.json"
    prompt_audio_path = tmp_path / "voice.wav"
    for path in (python_path, config_path, prompt_audio_path):
        path.touch()
    expected_synthesizer = FakeSpeechSynthesizer.create()
    received_configuration: list[
        tuple[
            Path,
            Path,
            Path,
            bool,
            bool,
            Path | None,
            FinalPhraseSlowdown | None,
        ]
    ] = []

    def create_voxtream(
        python_path: Path,
        config_path: Path,
        prompt_audio_path: Path,
        compile_model: bool,
        cache_prompt_in_memory: bool,
        speaking_rate_config_path: Path | None,
        final_phrase_slowdown: FinalPhraseSlowdown | None,
    ) -> FakeSpeechSynthesizer:
        received_configuration.append(
            (
                python_path,
                config_path,
                prompt_audio_path,
                compile_model,
                cache_prompt_in_memory,
                speaking_rate_config_path,
                final_phrase_slowdown,
            )
        )
        return expected_synthesizer

    monkeypatch.setattr(
        "app.compute.voice.tts_selection.VoxtreamSpeechSynthesizer",
        create_voxtream,
    )
    settings = SpeechSynthesisSettings(
        backend=SpeechSynthesisBackend.VOXTREAM,
        voxtream_python_path=python_path,
        voxtream_config_path=config_path,
        voxtream_speaking_rate_config_path=tmp_path / "speaking_rate.json",
        voxtream_prompt_audio_path=prompt_audio_path,
        voxtream_compile=False,
        voxtream_prompt_memory_cache=False,
        voxtream_final_phrase_slowdown=None,
    )

    assert create_speech_synthesizer(settings) is expected_synthesizer
    assert received_configuration == [
        (
            python_path,
            config_path,
            prompt_audio_path,
            False,
            False,
            None,
            None,
        )
    ]


def test_kyutai_synthesizer_implements_close() -> None:
    assert callable(KyutaiSpeechSynthesizer.close)
