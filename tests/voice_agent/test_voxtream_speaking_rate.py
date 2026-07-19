from __future__ import annotations

import pytest

from app.compute.voice.voxtream_speaking_rate import (
    FinalPhraseSlowdown,
    FinalPhraseSpeakingRate,
)


@pytest.mark.parametrize(
    ("syllables_per_second", "word_count", "message"),
    (
        (0.0, 4, "speaking rate"),
        (3.0, 0, "word count"),
    ),
)
def test_final_phrase_slowdown_rejects_invalid_values(
    syllables_per_second: float,
    word_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FinalPhraseSlowdown(
            syllables_per_second=syllables_per_second,
            word_count=word_count,
        )


def test_final_phrase_rate_stays_disabled_until_target_word_is_yielded() -> None:
    controller = FinalPhraseSpeakingRate(
        FinalPhraseSlowdown(syllables_per_second=3.0, word_count=2)
    )
    values = controller.values()
    for _ in range(5):
        controller.register_word()

    controller.mark_word_yielded()
    controller.mark_word_yielded()
    controller.finish_input()

    assert next(values) is None

    controller.mark_word_yielded()
    assert next(values) is None

    controller.mark_word_yielded()
    assert next(values) == 3.0


def test_final_phrase_rate_activates_when_finish_arrives_after_target_word() -> None:
    controller = FinalPhraseSpeakingRate(
        FinalPhraseSlowdown(syllables_per_second=3.0, word_count=2)
    )
    values = controller.values()
    for _ in range(5):
        controller.register_word()
    for _ in range(4):
        controller.mark_word_yielded()

    controller.finish_input()

    assert next(values) == 3.0


def test_short_utterance_slows_from_its_first_word() -> None:
    controller = FinalPhraseSpeakingRate(
        FinalPhraseSlowdown(syllables_per_second=3.0, word_count=4)
    )
    values = controller.values()
    controller.register_word()
    controller.register_word()
    controller.finish_input()

    assert next(values) is None

    controller.mark_word_yielded()
    assert next(values) == 3.0
