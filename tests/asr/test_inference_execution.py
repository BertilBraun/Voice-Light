from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.compute.asr.models.base import (
    MAX_CONCURRENT_ASR_INFERENCE_CALLS,
    BatchInferenceExecutor,
    inference_batches,
)
from app.shared.asr import TimestampedWord


class BarrierTranscriber:
    def __init__(self, barrier: Barrier) -> None:
        self.barrier = barrier

    def transcribe_batch(
        self,
        inputs: tuple[int, ...],
    ) -> tuple[tuple[TimestampedWord, ...], ...]:
        self.barrier.wait(timeout=2.0)
        return tuple(
            (TimestampedWord(text=str(value * 2), start_seconds=0.0, end_seconds=1.0),)
            for value in inputs
        )


class InvalidCardinalityTranscriber:
    def transcribe_batch(
        self,
        inputs: tuple[str, ...],
    ) -> tuple[tuple[TimestampedWord, ...], ...]:
        del inputs
        return ((TimestampedWord(text="result", start_seconds=0.0, end_seconds=1.0),),)


def test_inference_batches_groups_typed_items() -> None:
    items = ("a", "b", "c", "d", "e")

    assert inference_batches(items, batch_size=2) == (
        ("a", "b"),
        ("c", "d"),
        ("e",),
    )


def test_inference_batches_reject_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch size must be positive"):
        inference_batches(("item",), batch_size=0)


def test_batch_inference_executor_allows_four_simultaneous_calls() -> None:
    executor = BatchInferenceExecutor()
    simultaneous_call_barrier = Barrier(MAX_CONCURRENT_ASR_INFERENCE_CALLS)
    model = BarrierTranscriber(simultaneous_call_barrier)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ASR_INFERENCE_CALLS) as workers:
        futures = tuple(
            workers.submit(executor.execute, (value,), model)
            for value in range(MAX_CONCURRENT_ASR_INFERENCE_CALLS)
        )

    assert tuple(future.result().outputs[0][0].text for future in futures) == (
        "0",
        "2",
        "4",
        "6",
    )


def test_batch_inference_executor_validates_output_cardinality() -> None:
    executor = BatchInferenceExecutor()

    with pytest.raises(ValueError, match="one output per batch item"):
        executor.execute(("a", "b"), InvalidCardinalityTranscriber())
