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


def test_speech_synthesis_settings_reject_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="VOICE_LIGHT_TTS_BACKEND"):
        SpeechSynthesisSettings.from_environment(
            {"VOICE_LIGHT_TTS_BACKEND": "unknown"},
            tmp_path,
        )


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
    received_paths: list[tuple[Path, Path, Path]] = []

    def create_voxtream(
        python_path: Path,
        config_path: Path,
        prompt_audio_path: Path,
    ) -> FakeSpeechSynthesizer:
        received_paths.append((python_path, config_path, prompt_audio_path))
        return expected_synthesizer

    monkeypatch.setattr(
        "app.compute.voice.tts_selection.VoxtreamSpeechSynthesizer",
        create_voxtream,
    )
    settings = SpeechSynthesisSettings(
        backend=SpeechSynthesisBackend.VOXTREAM,
        voxtream_python_path=python_path,
        voxtream_config_path=config_path,
        voxtream_prompt_audio_path=prompt_audio_path,
    )

    assert create_speech_synthesizer(settings) is expected_synthesizer
    assert received_paths == [(python_path, config_path, prompt_audio_path)]


def test_kyutai_synthesizer_implements_close() -> None:
    assert callable(KyutaiSpeechSynthesizer.close)
