from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from enum import Enum, auto

from app.compute.voice.interfaces import (
    KyutaiSynthesisFirstAudioMetrics,
    SpeechSynthesisSession,
    SpeechSynthesizer,
    SynthesisEvent,
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
    VoxtreamSynthesisFirstAudioMetrics,
)


class _SequenceMarker(Enum):
    END = auto()


_SequenceItem = SpeechSynthesisSession | _SequenceMarker


class SpeechSynthesisSequence:
    def __init__(self, synthesizer: SpeechSynthesizer) -> None:
        self.synthesizer = synthesizer
        self.sessions: list[SpeechSynthesisSession] = []
        self.session_queue: asyncio.Queue[_SequenceItem] = asyncio.Queue()
        self.current_session: SpeechSynthesisSession | None = None
        self.input_finished = False

    async def add_word(self, word: SynthesisWord) -> None:
        if self.input_finished:
            raise ValueError("Cannot add a word after synthesis input has finished.")
        session = self.current_session
        if session is None:
            session = self.synthesizer.start_session()
            self.sessions.append(session)
            self.current_session = session
            await self.session_queue.put(session)
        await session.add_word(word)

    async def finish_utterance(self) -> None:
        if self.input_finished:
            raise ValueError("Cannot finish an utterance after synthesis input has finished.")
        session = self.current_session
        if session is None:
            return
        self.current_session = None
        await session.finish_input()

    async def finish_input(self) -> None:
        if self.input_finished:
            raise ValueError("Synthesis input may only be finished once.")
        await self.finish_utterance()
        self.input_finished = True
        await self.session_queue.put(_SequenceMarker.END)

    async def stream_events(self) -> AsyncIterator[SynthesisEvent]:
        sample_offset = 0
        while True:
            item = await self.session_queue.get()
            match item:
                case _SequenceMarker.END:
                    return
                case _:
                    session_sample_count = 0
                    async for event in item.stream_events():
                        match event:
                            case SynthesizedAudioChunk():
                                event_end_sample = event.start_sample + len(event.pcm_bytes) // 2
                                session_sample_count = max(session_sample_count, event_end_sample)
                                yield SynthesizedAudioChunk(
                                    pcm_bytes=event.pcm_bytes,
                                    start_sample=sample_offset + event.start_sample,
                                )
                            case SynthesizedWordBoundary():
                                yield SynthesizedWordBoundary(
                                    text_offset=event.text_offset,
                                    start_sample=sample_offset + event.start_sample,
                                )
                            case (
                                KyutaiSynthesisFirstAudioMetrics()
                                | VoxtreamSynthesisFirstAudioMetrics()
                            ):
                                yield event
                    sample_offset += session_sample_count

    async def cancel(self) -> None:
        self.input_finished = True
        self.current_session = None
        for session in self.sessions:
            await session.cancel()
        await self.session_queue.put(_SequenceMarker.END)
