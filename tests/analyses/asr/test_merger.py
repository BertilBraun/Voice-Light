from __future__ import annotations

from app.analyses.asr.merger import merged_words, parakeet_canary_union_words
from app.asr_quality.schemas import Word


def test_merged_words_average_timestamps_for_matching_tokens() -> None:
    merged = merged_words(
        primary=(Word(text="Hello", start_seconds=0.0, end_seconds=0.4),),
        secondary=(Word(text="hello", start_seconds=0.2, end_seconds=0.6),),
    )

    assert merged == (Word(text="Hello", start_seconds=0.1, end_seconds=0.5),)


def test_merged_words_keep_disagreements_in_timeline_order() -> None:
    merged = merged_words(
        primary=(
            Word(text="hello", start_seconds=0.0, end_seconds=0.3),
            Word(text="world", start_seconds=0.9, end_seconds=1.1),
        ),
        secondary=(
            Word(text="hello", start_seconds=0.0, end_seconds=0.4),
            Word(text="there", start_seconds=0.5, end_seconds=0.8),
            Word(text="world", start_seconds=0.9, end_seconds=1.2),
        ),
    )

    assert merged == (
        Word(text="hello", start_seconds=0.0, end_seconds=0.35),
        Word(text="there", start_seconds=0.5, end_seconds=0.8),
        Word(text="world", start_seconds=0.9, end_seconds=1.15),
    )


def test_parakeet_canary_union_words_preserves_parakeet_and_adds_canary_only_coverage() -> None:
    merged = parakeet_canary_union_words(
        parakeet_words=(
            Word(text="hello", start_seconds=0.0, end_seconds=0.4),
            Word(text="outside", start_seconds=1.5, end_seconds=1.9),
        ),
        canary_words=(
            Word(text="hello", start_seconds=0.1, end_seconds=0.3),
            Word(text="world", start_seconds=0.5, end_seconds=0.9),
        ),
    )

    assert merged == (
        Word(text="hello", start_seconds=0.0, end_seconds=0.4),
        Word(text="world", start_seconds=0.5, end_seconds=0.9),
        Word(text="outside", start_seconds=1.5, end_seconds=1.9),
    )
