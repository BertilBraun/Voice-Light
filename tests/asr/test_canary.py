from __future__ import annotations

import numpy as np
import pytest

from app.compute.asr.chunking import (
    CANARY_CHUNK_DURATION_SECONDS,
    CANARY_CHUNK_OVERLAP_SECONDS,
    CANARY_INFERENCE_BATCH_SIZE,
    CanaryAudioChunk,
    canary_audio_chunks,
    canary_chunk_samples,
    global_canary_chunk_words,
)
from app.compute.asr.models.base import inference_batches
from app.shared.asr import TimestampedWord


def test_canary_audio_chunks_overlap_and_bound_inference_windows() -> None:
    chunks = canary_audio_chunks(audio_duration_seconds=65.25)

    assert chunks == (
        CanaryAudioChunk(
            start_seconds=0.0,
            duration_seconds=30.0,
            keep_start_seconds=0.0,
            keep_end_seconds=29.5,
            is_final=False,
        ),
        CanaryAudioChunk(
            start_seconds=29.0,
            duration_seconds=30.0,
            keep_start_seconds=0.5,
            keep_end_seconds=29.5,
            is_final=False,
        ),
        CanaryAudioChunk(
            start_seconds=58.0,
            duration_seconds=7.25,
            keep_start_seconds=0.5,
            keep_end_seconds=7.25,
            is_final=True,
        ),
    )
    assert all(chunk.duration_seconds <= CANARY_CHUNK_DURATION_SECONDS for chunk in chunks)
    assert all(
        previous.start_seconds + previous.duration_seconds - CANARY_CHUNK_OVERLAP_SECONDS
        == current.start_seconds
        for previous, current in zip(chunks, chunks[1:], strict=False)
    )


def test_canary_audio_chunks_do_not_add_empty_exact_boundary_chunk() -> None:
    chunks = canary_audio_chunks(audio_duration_seconds=60.0)

    assert chunks == (
        CanaryAudioChunk(
            start_seconds=0.0,
            duration_seconds=30.0,
            keep_start_seconds=0.0,
            keep_end_seconds=29.5,
            is_final=False,
        ),
        CanaryAudioChunk(
            start_seconds=29.0,
            duration_seconds=30.0,
            keep_start_seconds=0.5,
            keep_end_seconds=29.5,
            is_final=False,
        ),
        CanaryAudioChunk(
            start_seconds=58.0,
            duration_seconds=2.0,
            keep_start_seconds=0.5,
            keep_end_seconds=2.0,
            is_final=True,
        ),
    )


def test_canary_chunks_are_grouped_into_bounded_inference_batches() -> None:
    chunks = canary_audio_chunks(audio_duration_seconds=300.0)

    batches = inference_batches(chunks, CANARY_INFERENCE_BATCH_SIZE)

    assert len(batches) == 2
    assert len(batches[0]) == CANARY_INFERENCE_BATCH_SIZE
    assert len(batches[1]) == len(chunks) - CANARY_INFERENCE_BATCH_SIZE
    assert tuple(chunk for batch in batches for chunk in batch) == chunks


def test_canary_chunk_samples_slices_the_prepared_waveform() -> None:
    audio = np.arange(80, dtype=np.float32)
    chunk = CanaryAudioChunk(
        start_seconds=2.0,
        duration_seconds=3.0,
        keep_start_seconds=0.5,
        keep_end_seconds=3.0,
        is_final=True,
    )

    samples = canary_chunk_samples(audio=audio, sample_rate=10, chunk=chunk)

    np.testing.assert_array_equal(samples, np.arange(20, 50, dtype=np.float32))


@pytest.mark.parametrize(
    ("audio_duration_seconds", "chunk_duration_seconds"),
    (
        (0.0, 30.0),
        (-1.0, 30.0),
        (float("inf"), 30.0),
        (30.0, 0.0),
        (30.0, -1.0),
        (30.0, 0.5),
        (30.0, float("nan")),
    ),
)
def test_canary_audio_chunks_reject_invalid_durations(
    audio_duration_seconds: float,
    chunk_duration_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        canary_audio_chunks(
            audio_duration_seconds=audio_duration_seconds,
            chunk_duration_seconds=chunk_duration_seconds,
        )


def test_global_canary_chunk_words_rebases_timestamps() -> None:
    chunk = CanaryAudioChunk(
        start_seconds=29.0,
        duration_seconds=30.0,
        keep_start_seconds=0.5,
        keep_end_seconds=29.5,
        is_final=False,
    )

    words = global_canary_chunk_words(
        chunk=chunk,
        words=(
            TimestampedWord(
                text="left duplicate",
                start_seconds=0.1,
                end_seconds=0.3,
            ),
            TimestampedWord(
                text="hello",
                start_seconds=0.5,
                end_seconds=0.9,
                confidence=0.9,
            ),
            TimestampedWord(
                text="world",
                start_seconds=1.0,
                end_seconds=1.4,
            ),
            TimestampedWord(
                text="right duplicate",
                start_seconds=29.6,
                end_seconds=29.8,
            ),
        ),
    )

    assert words == (
        TimestampedWord(
            text="hello",
            start_seconds=29.5,
            end_seconds=29.9,
            confidence=0.9,
        ),
        TimestampedWord(
            text="world",
            start_seconds=30.0,
            end_seconds=30.4,
        ),
    )
