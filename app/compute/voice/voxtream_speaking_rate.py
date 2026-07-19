from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class FinalPhraseSlowdown:
    syllables_per_second: float
    word_count: int

    def __post_init__(self) -> None:
        if self.syllables_per_second <= 0:
            raise ValueError("Final phrase speaking rate must be positive.")
        if self.word_count <= 0:
            raise ValueError("Final phrase word count must be positive.")


class FinalPhraseSpeakingRate:
    def __init__(self, slowdown: FinalPhraseSlowdown) -> None:
        self.slowdown = slowdown
        self._lock = threading.Lock()
        self._received_word_count = 0
        self._yielded_word_count = 0
        self._slowdown_start_word_index: int | None = None
        self._slowdown_active = False

    def register_word(self) -> None:
        with self._lock:
            if self._slowdown_start_word_index is not None:
                raise ValueError("Cannot register a word after synthesis input has finished.")
            self._received_word_count += 1

    def finish_input(self) -> None:
        with self._lock:
            if self._slowdown_start_word_index is not None:
                raise ValueError("Synthesis input may only be finished once.")
            self._slowdown_start_word_index = max(
                0,
                self._received_word_count - self.slowdown.word_count,
            )
            self._slowdown_active = self._yielded_word_count > self._slowdown_start_word_index

    def mark_word_yielded(self) -> None:
        with self._lock:
            word_index = self._yielded_word_count
            self._yielded_word_count += 1
            if (
                self._slowdown_start_word_index is not None
                and word_index >= self._slowdown_start_word_index
            ):
                self._slowdown_active = True

    def values(self) -> Iterator[float | None]:
        while True:
            with self._lock:
                slowdown_active = self._slowdown_active
            yield self.slowdown.syllables_per_second if slowdown_active else None
