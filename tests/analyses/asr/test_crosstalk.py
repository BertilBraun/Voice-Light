from __future__ import annotations

import numpy as np

from app.analyses.asr.crosstalk import (
    CrosstalkFilterConfig,
    CrosstalkFramePower,
    filter_crosstalk_words,
)
from app.asr.transcript import Word


def test_filter_crosstalk_words_removes_quiet_and_other_dominant_words() -> None:
    words = (
        Word(text="target", start_seconds=0.0, end_seconds=1.0),
        Word(text="bleed", start_seconds=1.0, end_seconds=2.0),
        Word(text="quiet", start_seconds=2.0, end_seconds=3.0),
    )
    frame_power_pair = CrosstalkFramePower(
        target=np.array([1.0, 0.1, 0.001], dtype=np.float32),
        other=np.array([0.01, 1.0, 0.001], dtype=np.float32),
        frame_duration_seconds=1.0,
    )

    filtered = filter_crosstalk_words(
        words=words,
        frame_power_pair=frame_power_pair,
        config=CrosstalkFilterConfig(),
    )

    assert filtered == (words[0],)


def test_filter_crosstalk_words_keeps_normal_target_speech_during_overlap() -> None:
    word = Word(text="backchannel", start_seconds=0.0, end_seconds=1.0)
    frame_power_pair = CrosstalkFramePower(
        target=np.array([0.1], dtype=np.float32),
        other=np.array([1.0], dtype=np.float32),
        frame_duration_seconds=1.0,
    )

    filtered = filter_crosstalk_words(
        words=(word,),
        frame_power_pair=frame_power_pair,
        config=CrosstalkFilterConfig(),
    )

    assert filtered == (word,)
