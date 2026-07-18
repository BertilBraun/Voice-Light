from __future__ import annotations

import numpy as np

from app.compute.asr.chunking import (
    PARAKEET_CHUNK_DURATION_SECONDS,
    PARAKEET_INFERENCE_BATCH_SIZE,
    ParakeetAudioChunk,
    global_chunk_words,
    parakeet_audio_chunks,
)
from app.compute.asr.models.base import inference_batches
from app.shared.asr import TimestampedWord


def test_parakeet_audio_chunks_overlap_and_bound_inference_windows() -> None:
    sample_rate = 10
    audio = np.arange(610, dtype=np.float32)

    chunks = parakeet_audio_chunks(audio=audio, sample_rate=sample_rate)

    assert [chunk.start_seconds for chunk in chunks] == [0.0, 29.0, 58.0]
    assert [len(chunk.samples) for chunk in chunks] == [300, 300, 30]
    assert all(
        len(chunk.samples) <= PARAKEET_CHUNK_DURATION_SECONDS * sample_rate for chunk in chunks
    )
    assert chunks[-1].is_final


def test_global_chunk_words_offsets_and_excludes_overlap_duplicates() -> None:
    chunk = ParakeetAudioChunk(
        samples=np.zeros(300, dtype=np.float32),
        start_seconds=29.0,
        keep_start_seconds=0.5,
        keep_end_seconds=29.5,
        is_final=False,
    )

    words = global_chunk_words(
        chunk=chunk,
        words=(
            TimestampedWord(text="duplicate", start_seconds=0.1, end_seconds=0.3),
            TimestampedWord(text="kept", start_seconds=0.6, end_seconds=0.8),
            TimestampedWord(text="next", start_seconds=29.7, end_seconds=29.9),
        ),
    )

    assert words == (TimestampedWord(text="kept", start_seconds=29.6, end_seconds=29.8),)


def test_parakeet_chunks_are_grouped_into_bounded_inference_batches() -> None:
    sample_rate = 10
    audio = np.zeros(round(300.0 * sample_rate), dtype=np.float32)
    chunks = parakeet_audio_chunks(audio=audio, sample_rate=sample_rate)

    batches = inference_batches(chunks, PARAKEET_INFERENCE_BATCH_SIZE)

    assert len(batches) == 2
    assert len(batches[0]) == PARAKEET_INFERENCE_BATCH_SIZE
    assert len(batches[1]) == len(chunks) - PARAKEET_INFERENCE_BATCH_SIZE
    assert tuple(chunk for batch in batches for chunk in batch) == chunks
