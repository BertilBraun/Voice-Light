from __future__ import annotations

import pytest

from app.local.voice.text_chunking import TextChunk, WordTextChunker


def test_emits_after_eight_streamed_words() -> None:
    chunker = WordTextChunker()

    assert chunker.add_text("One two three ") == ()
    assert chunker.add_text("four five six seven eight nine ") == (
        TextChunk(
            text="One two three four five six seven eight",
            word_count=8,
            is_final=False,
        ),
    )
    assert chunker.finish() == (TextChunk(text="nine", word_count=1, is_final=True),)


@pytest.mark.parametrize("word_count", [6, 7])
def test_emits_shorter_chunk_at_sentence_boundary(word_count: int) -> None:
    words = [f"word{index}" for index in range(word_count - 1)] + ["done."]
    chunker = WordTextChunker()

    assert chunker.add_text(" ".join(words) + " ") == (
        TextChunk(text=" ".join(words), word_count=word_count, is_final=False),
    )


def test_preserves_word_split_across_deltas() -> None:
    chunker = WordTextChunker()

    assert chunker.add_text("Hel") == ()
    assert chunker.add_text("lo there this is a streaming test now ") == (
        TextChunk(
            text="Hello there this is a streaming test now",
            word_count=8,
            is_final=False,
        ),
    )


def test_final_chunk_can_have_fewer_than_six_words() -> None:
    chunker = WordTextChunker()

    assert chunker.add_text("A short final response") == ()
    assert chunker.finish() == (
        TextChunk(text="A short final response", word_count=4, is_final=True),
    )


@pytest.mark.parametrize(
    ("minimum_words", "maximum_words"),
    [(0, 8), (8, 7)],
)
def test_rejects_invalid_limits(minimum_words: int, maximum_words: int) -> None:
    with pytest.raises(ValueError):
        WordTextChunker(minimum_words=minimum_words, maximum_words=maximum_words)
