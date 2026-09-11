from __future__ import annotations

from app.compute.voice.interfaces import SynthesisWord

SPEECH_FORMATTING_CHARACTERS = '*_`~"“”„‟«»'
SPEECH_FORMATTING_TRANSLATION = str.maketrans("", "", SPEECH_FORMATTING_CHARACTERS)


def normalize_synthesis_word(word: SynthesisWord) -> SynthesisWord | None:
    spoken_text = word.text.translate(SPEECH_FORMATTING_TRANSLATION)
    if not spoken_text:
        return None
    return SynthesisWord(
        text=spoken_text,
        text_start=word.text_start,
        text_end=word.text_end,
    )
