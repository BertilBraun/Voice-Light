from __future__ import annotations

from app.local.analyses.asr.metrics import timestamp_metrics
from app.local.asr.alignment import align_words, word_error_counts
from app.local.asr.normalization import normalize_token
from app.local.asr.transcript import Word


def test_alignment_counts_insertions_deletions_and_substitutions() -> None:
    reference_words = (
        Word(text="hello"),
        Word(text="world"),
        Word(text="again"),
    )
    prediction_words = (
        Word(text="hello"),
        Word(text="there"),
        Word(text="again"),
        Word(text="now"),
    )

    counts = word_error_counts(align_words(reference_words, prediction_words))

    assert counts.substitutions == 1
    assert counts.insertions == 1
    assert counts.deletions == 0
    assert counts.reference_words == 3
    assert counts.wer == 2 / 3


def test_timestamp_metrics_use_equal_words_only() -> None:
    alignment = align_words(
        (
            Word(text="um", start_seconds=1.0, end_seconds=1.2),
            Word(text="go", start_seconds=2.0, end_seconds=2.4),
        ),
        (
            Word(text="um", start_seconds=1.05, end_seconds=1.25),
            Word(text="gone", start_seconds=2.5, end_seconds=2.8),
        ),
    )

    metrics = timestamp_metrics(alignment)

    assert metrics.start.count == 1
    assert metrics.start.median_absolute_error == 0.050000000000000044
    assert metrics.end.count == 1


def test_normalization_preserves_disfluencies_and_partials() -> None:
    assert normalize_token("Uh-huh!") == "uh-huh"
    assert normalize_token("go-") == "go-"
