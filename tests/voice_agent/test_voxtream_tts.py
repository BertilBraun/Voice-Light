from __future__ import annotations

from collections import deque
from pathlib import Path

from app.compute.voice.tts_worker_protocol import TtsWordBoundaryEvent
from app.compute.voice.voxtream_alignment import (
    VoxtreamPendingWordBoundary,
    release_started_word_boundaries,
)
from app.compute.voice.voxtream_speaking_rate import FinalPhraseSlowdown
from app.compute.voice.voxtream_tts import _voxtream_worker_arguments


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


def test_voxtream_final_phrase_slowdown_is_forwarded_to_worker() -> None:
    module_arguments = _voxtream_worker_arguments(
        config_path=Path("/voxtream/generator.json"),
        prompt_audio_path=Path("/voxtream/prompt.wav"),
        compile_model=True,
        cache_prompt_in_memory=True,
        speaking_rate_config_path=Path("/voxtream/speaking_rate.json"),
        final_phrase_slowdown=FinalPhraseSlowdown(
            syllables_per_second=3.0,
            word_count=4,
        ),
    )

    assert module_arguments[-6:] == (
        "--speaking-rate-config",
        str(Path("/voxtream/speaking_rate.json")),
        "--final-slowdown-syllables-per-second",
        "3.0",
        "--final-slowdown-word-count",
        "4",
    )
