from __future__ import annotations

from app.analyses.asr.runners import merge_timestamp_pieces, words_from_whisperx_output
from app.asr_quality.schemas import Word


def test_merge_parakeet_timestamp_pieces_combines_word_fragments() -> None:
    words = merge_timestamp_pieces(
        [
            Word(text=" there", start_seconds=0.0, end_seconds=0.2),
            Word(text="'", start_seconds=0.2, end_seconds=0.25),
            Word(text="s", start_seconds=0.25, end_seconds=0.3),
            Word(text=" audio", start_seconds=0.4, end_seconds=0.7),
        ]
    )

    assert words == (
        [
            Word(text="there's", start_seconds=0.0, end_seconds=0.3),
            Word(text="audio", start_seconds=0.4, end_seconds=0.7),
        ]
    )


def test_words_from_whisperx_output_extracts_aligned_words() -> None:
    words = words_from_whisperx_output(
        {
            "segments": [
                {
                    "words": [
                        {"word": " hello", "start": 1.0, "end": 1.3, "score": 0.9},
                        {"word": "world", "start": 1.4, "end": 1.9, "score": 0.8},
                    ]
                }
            ]
        }
    )

    assert words == [
        Word(text="hello", start_seconds=1.0, end_seconds=1.3, confidence=0.9),
        Word(text="world", start_seconds=1.4, end_seconds=1.9, confidence=0.8),
    ]
