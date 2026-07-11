from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("VOICE_LIGHT_PORT", "8000"))
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=True)


if __name__ == "__main__":
    main()
