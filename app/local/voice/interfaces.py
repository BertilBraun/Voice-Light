from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class SpeechDetector(Protocol):
    def process_audio(self, pcm_bytes: bytes) -> bool: ...


class TranscriptionSession(Protocol):
    async def add_audio(self, pcm_bytes: bytes) -> str | None: ...

    async def finish(self) -> str: ...

    async def close(self) -> None: ...


class Transcriber(Protocol):
    def start_session(self) -> TranscriptionSession: ...


class LanguageModel(Protocol):
    def stream_response(
        self,
        conversation: tuple[tuple[str, str], ...],
    ) -> AsyncIterator[str]: ...


class SpeechSynthesizer(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def stream_audio(self, text: str) -> AsyncIterator[bytes]: ...
