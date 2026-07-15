from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from app.compute.voice.conversation import ConversationMessage


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
        conversation: tuple[ConversationMessage, ...],
    ) -> AsyncIterator[str]: ...


@dataclass(frozen=True)
class SynthesisWord:
    text: str
    text_start: int
    text_end: int


@dataclass(frozen=True)
class SynthesizedAudioChunk:
    pcm_bytes: bytes
    start_sample: int


@dataclass(frozen=True)
class SynthesizedWordBoundary:
    text_offset: int
    start_sample: int


SynthesisEvent = SynthesizedAudioChunk | SynthesizedWordBoundary


class SpeechSynthesisSession(Protocol):
    async def add_word(self, word: SynthesisWord) -> None: ...

    async def finish_input(self) -> None: ...

    def stream_events(self) -> AsyncIterator[SynthesisEvent]: ...

    async def cancel(self) -> None: ...


class SpeechSynthesizer(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def start_session(self) -> SpeechSynthesisSession: ...
