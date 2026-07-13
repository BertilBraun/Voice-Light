from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("VOICE_LIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("VOICE_LIGHT_PORT", "8000"))
    reload_enabled = os.environ.get("VOICE_LIGHT_RELOAD", "true").lower() == "true"
    uvicorn.run("app.local.main:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
