from __future__ import annotations

import pytest

from app.compute.voice.sentence_chunking import SentenceTextChunker


@pytest.mark.parametrize(
    ("text_deltas", "expected_sentences", "expected_final"),
    [
        (("Hello", " world."), ("Hello world.",), ()),
        (("One! Two? Three.",), ("One!", "Two?", "Three."), ()),
        (("Complete sentence. Remaining words",), ("Complete sentence.",), ("Remaining words",)),
        (("No punctuation",), (), ("No punctuation",)),
    ],
)
def test_sentence_chunking(
    text_deltas: tuple[str, ...],
    expected_sentences: tuple[str, ...],
    expected_final: tuple[str, ...],
) -> None:
    chunker = SentenceTextChunker()

    sentences = tuple(
        sentence for text_delta in text_deltas for sentence in chunker.add_text(text_delta)
    )

    assert sentences == expected_sentences
    assert chunker.finish() == expected_final
