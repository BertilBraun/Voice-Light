from __future__ import annotations

from app.compute.voice.interfaces import SynthesisWord
from app.compute.voice.word_stream import CompleteWordStream


def test_complete_words_are_emitted_as_soon_as_whitespace_arrives() -> None:
    stream = CompleteWordStream()

    assert stream.add_text("  Hel") == ()
    assert stream.add_text("lo, ") == (SynthesisWord(text="Hello,", text_start=2, text_end=8),)
    assert stream.add_text("world") == ()
    assert stream.add_text("!  next ") == (
        SynthesisWord(text="world!", text_start=9, text_end=15),
        SynthesisWord(text="next", text_start=17, text_end=21),
    )


def test_finish_flushes_one_trailing_word_with_original_offsets() -> None:
    stream = CompleteWordStream()

    assert stream.add_text("first ") == (SynthesisWord(text="first", text_start=0, text_end=5),)
    assert stream.add_text("  final.") == ()
    assert stream.finish() == (SynthesisWord(text="final.", text_start=8, text_end=14),)
    assert stream.finish() == ()
