from __future__ import annotations

import pytest

from app.compute.voice.interfaces import SynthesisWord
from app.compute.voice.tts_text import normalize_synthesis_word


@pytest.mark.parametrize(
    ("source", "spoken"),
    [
        ("**25", "25"),
        ("knots**", "knots"),
        ('"quoted"', "quoted"),
        ("“quoted”", "quoted"),
        ("`code`", "code"),
        ("don't", "don't"),
        ("Leo’s", "Leo’s"),
        ("25–35", "25–35"),
        ("Great! 😄", "Great! "),
        ("weather☀️", "weather"),
        ("flag🇬🇧", "flag"),
        ("20°C", "20°C"),
    ],
)
def test_normalize_synthesis_word_removes_speech_formatting(
    source: str,
    spoken: str,
) -> None:
    word = SynthesisWord(text=source, text_start=4, text_end=4 + len(source))

    assert normalize_synthesis_word(word) == SynthesisWord(
        text=spoken,
        text_start=word.text_start,
        text_end=word.text_end,
    )


def test_normalize_synthesis_word_drops_formatting_only_token() -> None:
    word = SynthesisWord(text="***", text_start=4, text_end=7)

    assert normalize_synthesis_word(word) is None


def test_normalize_synthesis_word_drops_emoji_only_token() -> None:
    word = SynthesisWord(text="👩🏽‍💻", text_start=4, text_end=8)

    assert normalize_synthesis_word(word) is None
