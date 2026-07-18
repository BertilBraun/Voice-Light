from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("moshi", reason="Compute health tests require the compute extra.")

from app.compute.config import ComputeSettings, VoiceStackSettings
from app.compute.main import create_compute_app
from app.compute.voice.search import UnconfiguredSearchSettings
from app.compute.voice.tts_selection import (
    SpeechSynthesisBackend,
    SpeechSynthesisSettings,
)


def test_liveness_is_public_and_readiness_requires_authentication(tmp_path: Path) -> None:
    application = create_compute_app(
        ComputeSettings(
            token="secret-token",
            log_directory=tmp_path / "logs",
            voice_stack=VoiceStackSettings(
                search=UnconfiguredSearchSettings(),
                speech_synthesis=SpeechSynthesisSettings(
                    backend=SpeechSynthesisBackend.KYUTAI,
                    voxtream_python_path=tmp_path / "voxtream-python",
                    voxtream_config_path=tmp_path / "voxtream-config.json",
                    voxtream_prompt_audio_path=tmp_path / "voice.wav",
                ),
            ),
        )
    )
    client = TestClient(application)

    assert client.get("/health/live").json() == {"status": "live"}
    assert client.get("/health/ready").status_code == 401

    response = client.get(
        "/health/ready",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert [stage["status"] for stage in response.json()["stages"]] == [
        "pending",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert response.headers["X-Request-ID"]
