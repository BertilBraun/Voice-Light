from __future__ import annotations

import numpy as np
import pytest

from app.local.asr.transcript import Word
from app.local.ingestion.alignment import align_audio_tracks, shift_words
from app.shared.audio import AudioTrack
from app.shared.quality import AudioMetadata

SAMPLE_RATE = 16_000


def test_negative_speaker2_shift_trims_start_and_preserves_shared_timeline() -> None:
    speaker1 = audio_track(sample_count=32_000, value=1.0)
    speaker2 = audio_track(sample_count=40_000, value=2.0)

    aligned = align_audio_tracks(
        speaker1=speaker1,
        speaker2=speaker2,
        speaker2_shift_seconds=-0.5,
    )

    assert len(aligned.speaker1.samples) == 32_000
    assert len(aligned.speaker2.samples) == 32_000
    assert aligned.duration_seconds == pytest.approx(2.0)
    assert np.all(aligned.speaker2.samples == 2.0)


def test_positive_speaker2_shift_left_pads_silence_and_end_pads_speaker1() -> None:
    speaker1 = audio_track(sample_count=16_000, value=1.0)
    speaker2 = audio_track(sample_count=16_000, value=2.0)

    aligned = align_audio_tracks(
        speaker1=speaker1,
        speaker2=speaker2,
        speaker2_shift_seconds=0.5,
    )

    assert len(aligned.speaker1.samples) == 24_000
    assert len(aligned.speaker2.samples) == 24_000
    assert np.all(aligned.speaker2.samples[:8_000] == 0.0)
    assert np.all(aligned.speaker2.samples[8_000:] == 2.0)
    assert np.all(aligned.speaker1.samples[16_000:] == 0.0)


def test_tail_length_difference_is_end_padded_instead_of_invalidated() -> None:
    aligned = align_audio_tracks(
        speaker1=audio_track(sample_count=16_000, value=1.0),
        speaker2=audio_track(sample_count=48_000, value=2.0),
        speaker2_shift_seconds=0.0,
    )

    assert aligned.duration_seconds == pytest.approx(3.0)
    assert len(aligned.speaker1.samples) == len(aligned.speaker2.samples) == 48_000


def test_transcript_shift_uses_audio_sign_and_clips_to_shared_timeline() -> None:
    words = (
        Word(text="trimmed", start_seconds=0.1, end_seconds=0.4),
        Word(text="clipped", start_seconds=0.4, end_seconds=0.8),
        Word(text="kept", start_seconds=1.0, end_seconds=1.4),
    )

    shifted = shift_words(
        words=words,
        shift_seconds=-0.5,
        timeline_duration_seconds=1.0,
    )

    assert tuple(word.text for word in shifted) == ("clipped", "kept")
    assert shifted[0].start_seconds == pytest.approx(0.0)
    assert shifted[0].end_seconds == pytest.approx(0.3)
    assert shifted[1].start_seconds == pytest.approx(0.5)
    assert shifted[1].end_seconds == pytest.approx(0.9)


def audio_track(sample_count: int, value: float) -> AudioTrack:
    samples = np.full((sample_count, 1), value, dtype=np.float32)
    return AudioTrack(
        samples=samples,
        metadata=AudioMetadata(
            duration_seconds=sample_count / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            channels=1,
            sample_count=sample_count,
        ),
    )
