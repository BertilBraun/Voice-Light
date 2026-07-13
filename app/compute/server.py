from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("VOICE_LIGHT_COMPUTE_HOST", "0.0.0.0")
    port = int(os.environ.get("VOICE_LIGHT_COMPUTE_PORT", "8000"))
    uvicorn.run(
        "app.compute.main:create_app_from_environment",
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        factory=True,
    )


if __name__ == "__main__":
    main()
