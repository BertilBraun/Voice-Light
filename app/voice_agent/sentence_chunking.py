from __future__ import annotations

from dataclasses import dataclass

SENTENCE_ENDINGS = frozenset(".?!")


@dataclass
class SentenceTextChunker:
    buffered_text: str = ""

    def add_text(self, text: str) -> tuple[str, ...]:
        self.buffered_text += text
        sentences: list[str] = []
        sentence_start = 0
        for index, character in enumerate(self.buffered_text):
            if character not in SENTENCE_ENDINGS:
                continue
            sentence = self.buffered_text[sentence_start : index + 1].strip()
            if sentence:
                sentences.append(sentence)
            sentence_start = index + 1
        self.buffered_text = self.buffered_text[sentence_start:].lstrip()
        return tuple(sentences)

    def finish(self) -> tuple[str, ...]:
        remaining_text = self.buffered_text.strip()
        self.buffered_text = ""
        return (remaining_text,) if remaining_text else ()
