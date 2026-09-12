from __future__ import annotations

import re

from app.compute.voice.interfaces import SynthesisWord

SPEECH_FORMATTING_CHARACTERS = '*_`~"“”„‟«»'
SPEECH_FORMATTING_TRANSLATION = str.maketrans("", "", SPEECH_FORMATTING_CHARACTERS)
EMOJI_CHARACTER_PATTERN = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001f6ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]"
)
EMOJI_JOINER_PATTERN = re.compile("[\u200d\ufe0f\u20e3]")


def normalize_synthesis_word(word: SynthesisWord) -> SynthesisWord | None:
    spoken_text = word.text.translate(SPEECH_FORMATTING_TRANSLATION)
    spoken_text = EMOJI_CHARACTER_PATTERN.sub("", spoken_text)
    spoken_text = EMOJI_JOINER_PATTERN.sub("", spoken_text)
    if not spoken_text:
        return None
    return SynthesisWord(
        text=spoken_text,
        text_start=word.text_start,
        text_end=word.text_end,
    )
