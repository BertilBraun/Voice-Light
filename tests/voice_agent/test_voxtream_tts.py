from __future__ import annotations

from collections import deque

from app.compute.voice.tts_worker_protocol import TtsWordBoundaryEvent
from app.compute.voice.voxtream_alignment import (
    VoxtreamPendingWordBoundary,
    release_started_word_boundaries,
)


def test_voxtream_releases_text_offset_when_word_phonemes_start() -> None:
    pending_boundaries = deque(
        (
            VoxtreamPendingWordBoundary(phone_start=3, text_offset=5),
            VoxtreamPendingWordBoundary(phone_start=8, text_offset=11),
        )
    )

    assert release_started_word_boundaries(pending_boundaries, 2, 0) == ()

    assert release_started_word_boundaries(pending_boundaries, 3, 480) == (
        TtsWordBoundaryEvent(text_offset=5, start_sample=480),
    )

    assert release_started_word_boundaries(pending_boundaries, 8, 1_440) == (
        TtsWordBoundaryEvent(text_offset=11, start_sample=1_440),
    )
