from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.shared.asr import TimestampedWord

PARAKEET_CHUNK_DURATION_SECONDS = 30.0
PARAKEET_CHUNK_OVERLAP_SECONDS = 1.0
PARAKEET_INFERENCE_BATCH_SIZE = 8


@dataclass(frozen=True)
class ParakeetAudioChunk:
    samples: NDArray[np.float32]
    start_seconds: float
    keep_start_seconds: float
    keep_end_seconds: float
    is_final: bool


def parakeet_audio_chunks(
    audio: NDArray[np.float32],
    sample_rate: int,
) -> tuple[ParakeetAudioChunk, ...]:
    chunk_sample_count = round(PARAKEET_CHUNK_DURATION_SECONDS * sample_rate)
    overlap_sample_count = round(PARAKEET_CHUNK_OVERLAP_SECONDS * sample_rate)
    step_sample_count = chunk_sample_count - overlap_sample_count
    assert chunk_sample_count > overlap_sample_count > 0
    chunks: list[ParakeetAudioChunk] = []
    start_sample = 0
    while start_sample < len(audio):
        end_sample = min(start_sample + chunk_sample_count, len(audio))
        is_final = end_sample == len(audio)
        duration_seconds = (end_sample - start_sample) / sample_rate
        chunks.append(
            ParakeetAudioChunk(
                samples=audio[start_sample:end_sample],
                start_seconds=start_sample / sample_rate,
                keep_start_seconds=(
                    0.0 if start_sample == 0 else PARAKEET_CHUNK_OVERLAP_SECONDS / 2.0
                ),
                keep_end_seconds=(
                    duration_seconds
                    if is_final
                    else duration_seconds - PARAKEET_CHUNK_OVERLAP_SECONDS / 2.0
                ),
                is_final=is_final,
            )
        )
        start_sample += step_sample_count
    return tuple(chunks)


def parakeet_chunk_batches(
    chunks: tuple[ParakeetAudioChunk, ...],
    batch_size: int = PARAKEET_INFERENCE_BATCH_SIZE,
) -> tuple[tuple[ParakeetAudioChunk, ...], ...]:
    if batch_size <= 0:
        raise ValueError("Parakeet inference batch size must be positive.")
    return tuple(
        chunks[start_index : start_index + batch_size]
        for start_index in range(0, len(chunks), batch_size)
    )


def global_chunk_words(
    chunk: ParakeetAudioChunk,
    words: tuple[TimestampedWord, ...],
) -> tuple[TimestampedWord, ...]:
    global_words: list[TimestampedWord] = []
    for word in words:
        assert word.start_seconds is not None
        assert word.end_seconds is not None
        midpoint_seconds = (word.start_seconds + word.end_seconds) / 2.0
        within_left_boundary = midpoint_seconds >= chunk.keep_start_seconds
        within_right_boundary = (
            midpoint_seconds <= chunk.keep_end_seconds
            if chunk.is_final
            else midpoint_seconds < chunk.keep_end_seconds
        )
        if within_left_boundary and within_right_boundary:
            global_words.append(
                TimestampedWord(
                    text=word.text,
                    start_seconds=word.start_seconds + chunk.start_seconds,
                    end_seconds=word.end_seconds + chunk.start_seconds,
                    confidence=word.confidence,
                )
            )
    return tuple(global_words)
