from __future__ import annotations

import re

from app.compute.voice.interfaces import SynthesisWord

COMPLETED_WORD_PATTERN = re.compile(r"\S+\s+")
TRAILING_WORD_PATTERN = re.compile(r"\S+")


class CompleteWordStream:
    def __init__(self) -> None:
        self.pending_text = ""
        self.pending_start = 0

    def add_text(self, text: str) -> tuple[SynthesisWord, ...]:
        self.pending_text += text
        words: list[SynthesisWord] = []
        consumed_length = 0
        for match in COMPLETED_WORD_PATTERN.finditer(self.pending_text):
            word_text = match.group().rstrip()
            word_start = self.pending_start + match.start()
            words.append(
                SynthesisWord(
                    text=word_text,
                    text_start=word_start,
                    text_end=word_start + len(word_text),
                )
            )
            consumed_length = match.end()
        if consumed_length:
            self.pending_text = self.pending_text[consumed_length:]
            self.pending_start += consumed_length
        return tuple(words)

    def finish(self) -> tuple[SynthesisWord, ...]:
        match = TRAILING_WORD_PATTERN.search(self.pending_text)
        if match is None:
            self.pending_start += len(self.pending_text)
            self.pending_text = ""
            return ()
        word = SynthesisWord(
            text=match.group(),
            text_start=self.pending_start + match.start(),
            text_end=self.pending_start + match.end(),
        )
        self.pending_start += len(self.pending_text)
        self.pending_text = ""
        return (word,)
