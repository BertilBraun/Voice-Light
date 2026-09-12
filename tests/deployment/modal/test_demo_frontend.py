from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from deployment.modal.demo_frontend import create_demo_frontend_app

VOICE_WEBSOCKET_URL = "wss://voice.example/v1/voice"


def voice_page_directory() -> Path:
    return Path(__file__).parents[3] / "app" / "local" / "web" / "pages" / "voice-agent"


def test_demo_frontend_redirects_to_voice_page_with_public_compute_endpoint() -> None:
    application = create_demo_frontend_app(
        voice_page_directory=voice_page_directory(),
        voice_websocket_url=VOICE_WEBSOCKET_URL,
    )

    with TestClient(application, follow_redirects=False) as client:
        overview_response = client.get("/")
        voice_response = client.get("/voice-agent")

    assert overview_response.status_code == 307
    assert overview_response.headers["location"] == "/voice-agent"
    assert voice_response.status_code == 307
    redirect = urlparse(voice_response.headers["location"])
    assert redirect.path == "/pages/voice-agent/index.html"
    assert parse_qs(redirect.query) == {"compute": [VOICE_WEBSOCKET_URL]}


def test_demo_frontend_serves_voice_page_and_assets() -> None:
    application = create_demo_frontend_app(
        voice_page_directory=voice_page_directory(),
        voice_websocket_url=VOICE_WEBSOCKET_URL,
    )

    with TestClient(application) as client:
        page_response = client.get("/")
        script_response = client.get("/pages/voice-agent/app.js")

    assert page_response.status_code == 200
    assert "Streaming Voice Agent" in page_response.text
    assert script_response.status_code == 200
    assert "voice-light-compute-voice-endpoint" in script_response.text


def test_demo_frontend_rejects_insecure_public_websocket() -> None:
    with pytest.raises(ValueError, match="must use wss"):
        create_demo_frontend_app(
            voice_page_directory=voice_page_directory(),
            voice_websocket_url="ws://voice.example/v1/voice",
        )
