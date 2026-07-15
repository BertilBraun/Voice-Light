from __future__ import annotations

import pytest

from app.compute.voice.interfaces import SynthesizedWordBoundary
from app.compute.voice.tts_alignment import TranscriptBoundaryTracker


def test_removed_word_extends_pending_preceding_audible_word() -> None:
    tracker = TranscriptBoundaryTracker()

    assert tracker.add_source_word((True,), text_offset=5) == ()
    assert tracker.add_source_word((), text_offset=7) == ()

    assert tracker.consume_transcript_word(start_sample=1_920) == SynthesizedWordBoundary(
        text_offset=7,
        start_sample=1_920,
    )


def test_removed_word_reuses_emitted_preceding_word_start() -> None:
    tracker = TranscriptBoundaryTracker()
    tracker.add_source_word((True,), text_offset=5)
    assert tracker.consume_transcript_word(start_sample=1_920) == SynthesizedWordBoundary(
        text_offset=5,
        start_sample=1_920,
    )

    assert tracker.add_source_word((), text_offset=7) == (
        SynthesizedWordBoundary(text_offset=7, start_sample=1_920),
    )


def test_leading_removed_word_is_covered_by_next_audible_word() -> None:
    tracker = TranscriptBoundaryTracker()

    assert tracker.add_source_word((), text_offset=1) == ()
    assert tracker.add_source_word((True,), text_offset=7) == ()

    assert tracker.consume_transcript_word(start_sample=3_840) == SynthesizedWordBoundary(
        text_offset=7,
        start_sample=3_840,
    )


def test_multiple_normalized_entries_only_map_first_transcript_word() -> None:
    tracker = TranscriptBoundaryTracker()
    tracker.add_source_word((True, True), text_offset=7)

    assert tracker.consume_transcript_word(start_sample=1_920) == SynthesizedWordBoundary(
        text_offset=7,
        start_sample=1_920,
    )
    assert tracker.consume_transcript_word(start_sample=3_840) is None


def test_source_offsets_must_be_monotonic() -> None:
    tracker = TranscriptBoundaryTracker()
    tracker.add_source_word((True,), text_offset=7)

    with pytest.raises(ValueError, match="increase monotonically"):
        tracker.add_source_word((), text_offset=7)
