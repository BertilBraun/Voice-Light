from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from app.compute.voice.tts_worker_protocol import TtsWordBoundaryEvent


@dataclass(frozen=True)
class VoxtreamPendingWordBoundary:
    phone_start: int
    text_offset: int


def release_started_word_boundaries(
    pending_boundaries: deque[VoxtreamPendingWordBoundary],
    phone_position: int,
    start_sample: int,
) -> tuple[TtsWordBoundaryEvent, ...]:
    released_boundaries: list[TtsWordBoundaryEvent] = []
    while pending_boundaries and pending_boundaries[0].phone_start <= phone_position:
        boundary = pending_boundaries.popleft()
        released_boundaries.append(
            TtsWordBoundaryEvent(
                text_offset=boundary.text_offset,
                start_sample=start_sample,
            )
        )
    return tuple(released_boundaries)
