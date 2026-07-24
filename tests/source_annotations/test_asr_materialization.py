from __future__ import annotations

import pytest

from app.local.source_annotations.asr_materialization import (
    SOURCE_TRANSCRIPT_ANNOTATOR_ID,
    source_transcript_track,
    timestamped_source_events,
)
from app.local.source_annotations.models import SourceAnnotationEvent, SourceAnnotationTrack


def test_timestamped_source_events_preserve_nonempty_source_text_and_timestamps() -> None:
    track = SourceAnnotationTrack(
        annotator_id=SOURCE_TRANSCRIPT_ANNOTATOR_ID,
        events=(
            SourceAnnotationEvent(
                start_seconds=1.25,
                end_seconds=2.5,
                label="Normal Turn",
                text="  hello there  ",
            ),
            SourceAnnotationEvent(
                start_seconds=3.0,
                end_seconds=3.5,
                label="Non-Speech Noise",
                text=" ",
            ),
        ),
    )

    words = timestamped_source_events(track)

    assert len(words) == 1
    assert words[0].model_dump() == {
        "text": "hello there",
        "start_seconds": 1.25,
        "end_seconds": 2.5,
        "confidence": None,
    }


def test_source_transcript_track_requires_designated_annotator() -> None:
    track = SourceAnnotationTrack(annotator_id="b", events=())

    with pytest.raises(ValueError, match="annotator 'a'"):
        source_transcript_track((track,))
