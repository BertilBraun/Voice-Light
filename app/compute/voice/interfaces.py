from __future__ import annotations

from typing import Protocol


class TranscriptionSession(Protocol):
    async def add_audio(self, pcm_bytes: bytes) -> str | None: ...

    async def finish(self) -> str: ...

    async def close(self) -> None: ...
