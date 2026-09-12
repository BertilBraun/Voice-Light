from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


def create_demo_frontend_app(
    voice_page_directory: Path,
    voice_websocket_url: str,
) -> FastAPI:
    if not voice_websocket_url.startswith("wss://"):
        raise ValueError("The public voice WebSocket URL must use wss://.")

    application = FastAPI(title="Voice Light Demo")
    application.mount(
        "/pages/voice-agent",
        StaticFiles(directory=voice_page_directory),
        name="voice-agent-assets",
    )

    @application.get("/", include_in_schema=False)
    def overview() -> RedirectResponse:
        return RedirectResponse("/voice-agent")

    @application.get("/voice-agent", include_in_schema=False)
    def voice_agent() -> RedirectResponse:
        query = urlencode({"compute": voice_websocket_url})
        return RedirectResponse(f"/pages/voice-agent/index.html?{query}")

    return application
