from __future__ import annotations

import re
from dataclasses import dataclass

WORD_PATTERN = re.compile(r"\S+\s+")
SENTENCE_ENDINGS = (".", "!", "?", '."', '!"', '?"')


@dataclass(frozen=True)
class TextChunk:
    text: str
    word_count: int
    is_final: bool


class WordTextChunker:
    def __init__(self, minimum_words: int = 6, maximum_words: int = 8) -> None:
        if minimum_words < 1:
            raise ValueError("minimum_words must be positive.")
        if maximum_words < minimum_words:
            raise ValueError("maximum_words must be at least minimum_words.")
        self.minimum_words = minimum_words
        self.maximum_words = maximum_words
        self._incomplete_text = ""
        self._words: list[str] = []

    def add_text(self, text_delta: str) -> tuple[TextChunk, ...]:
        self._incomplete_text += text_delta
        complete_matches = tuple(WORD_PATTERN.finditer(self._incomplete_text))
        if complete_matches:
            self._words.extend(match.group().strip() for match in complete_matches)
            self._incomplete_text = self._incomplete_text[complete_matches[-1].end() :]
        return self._emit_ready_chunks()

    def finish(self) -> tuple[TextChunk, ...]:
        trailing_text = self._incomplete_text.strip()
        if trailing_text:
            self._words.extend(trailing_text.split())
        self._incomplete_text = ""

        chunks = list(self._emit_ready_chunks())
        if self._words:
            chunks.append(self._pop_chunk(len(self._words), is_final=True))
        elif chunks:
            last_chunk = chunks[-1]
            chunks[-1] = TextChunk(
                text=last_chunk.text,
                word_count=last_chunk.word_count,
                is_final=True,
            )
        return tuple(chunks)

    def _emit_ready_chunks(self) -> tuple[TextChunk, ...]:
        chunks: list[TextChunk] = []
        while len(self._words) >= self.minimum_words:
            chunk_size = self._ready_chunk_size()
            if chunk_size is None:
                break
            chunks.append(self._pop_chunk(chunk_size, is_final=False))
        return tuple(chunks)

    def _ready_chunk_size(self) -> int | None:
        upper_bound = min(len(self._words), self.maximum_words)
        for word_count in range(self.minimum_words, upper_bound + 1):
            if self._words[word_count - 1].endswith(SENTENCE_ENDINGS):
                return word_count
        if len(self._words) >= self.maximum_words:
            return self.maximum_words
        return None

    def _pop_chunk(self, word_count: int, is_final: bool) -> TextChunk:
        words = self._words[:word_count]
        del self._words[:word_count]
        return TextChunk(text=" ".join(words), word_count=word_count, is_final=is_final)
