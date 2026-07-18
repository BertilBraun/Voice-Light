from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore
from typing import Protocol, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.audio import load_audio
from app.shared.audio.transport import CANONICAL_MODEL_AUDIO_SPEC
from app.shared.storage.local import LocalStorageBackend

MAX_CONCURRENT_ASR_INFERENCE_CALLS = 4

InferenceInput = TypeVar("InferenceInput")


@dataclass(frozen=True)
class PreparedAsrAudio:
    path: Path
    samples: NDArray[np.float32]
    sample_rate: int
    duration_seconds: float
    loading_time_seconds: float


@dataclass(frozen=True)
class BatchInferenceResult:
    outputs: tuple[tuple[TimestampedWord, ...], ...]
    queue_time_seconds: float
    execution_time_seconds: float


class BatchTranscriber(Protocol[InferenceInput]):
    def transcribe_batch(
        self,
        inputs: tuple[InferenceInput, ...],
    ) -> tuple[tuple[TimestampedWord, ...], ...]: ...


class BatchInferenceExecutor:
    def __init__(
        self,
        max_concurrent_calls: int = MAX_CONCURRENT_ASR_INFERENCE_CALLS,
    ) -> None:
        if max_concurrent_calls <= 0:
            raise ValueError("Maximum concurrent ASR inference calls must be positive.")
        self.inference_slots = BoundedSemaphore(max_concurrent_calls)

    def execute(
        self,
        inputs: tuple[InferenceInput, ...],
        model: BatchTranscriber[InferenceInput],
    ) -> BatchInferenceResult:
        if not inputs:
            raise ValueError("ASR inference batch must not be empty.")
        queue_start = time.perf_counter()
        with self.inference_slots:
            queue_time_seconds = time.perf_counter() - queue_start
            execution_start = time.perf_counter()
            outputs = model.transcribe_batch(inputs)
            execution_time_seconds = time.perf_counter() - execution_start
        if len(outputs) != len(inputs):
            raise ValueError("ASR model did not return one output per batch item.")
        return BatchInferenceResult(
            outputs=outputs,
            queue_time_seconds=queue_time_seconds,
            execution_time_seconds=execution_time_seconds,
        )


class LoadedAsrModel(Protocol):
    @property
    def model_id(self) -> AsrModelId: ...

    @property
    def package_names(self) -> tuple[str, ...]: ...

    @property
    def model_loading_time_seconds(self) -> float: ...

    def transcribe(self, audio: PreparedAsrAudio) -> ModelTranscription: ...


@dataclass(frozen=True)
class ModelTranscription:
    words: tuple[TimestampedWord, ...]
    inference_queue_time_seconds: float
    audio_loading_time_seconds: float
    model_execution_time_seconds: float


@dataclass(frozen=True)
class TimedTranscription:
    model_id: AsrModelId
    words: tuple[TimestampedWord, ...]
    model_loading_time_seconds: float
    inference_time_seconds: float
    package_names: tuple[str, ...]
    inference_queue_time_seconds: float = 0.0
    audio_loading_time_seconds: float = 0.0
    model_execution_time_seconds: float = 0.0


def prepare_asr_audio(audio_path: Path) -> PreparedAsrAudio:
    loading_start = time.perf_counter()
    track = load_audio(
        storage=LocalStorageBackend(),
        path=audio_path.as_posix(),
        target_sample_rate=CANONICAL_MODEL_AUDIO_SPEC.sample_rate,
    )
    if track.metadata.channels != CANONICAL_MODEL_AUDIO_SPEC.channels:
        raise ValueError("Prepared ASR audio must be mono.")
    if track.samples.ndim != 2 or track.samples.shape[1] != 1:
        raise ValueError("Prepared ASR audio samples must have one channel.")
    return PreparedAsrAudio(
        path=audio_path,
        samples=track.samples[:, 0],
        sample_rate=track.metadata.sample_rate,
        duration_seconds=track.metadata.duration_seconds,
        loading_time_seconds=time.perf_counter() - loading_start,
    )


def inference_batches(
    items: tuple[InferenceInput, ...],
    batch_size: int,
) -> tuple[tuple[InferenceInput, ...], ...]:
    if batch_size <= 0:
        raise ValueError("ASR inference batch size must be positive.")
    return tuple(
        items[start_index : start_index + batch_size]
        for start_index in range(0, len(items), batch_size)
    )


def timed_transcription(model: LoadedAsrModel, audio: PreparedAsrAudio) -> TimedTranscription:
    inference_start = time.perf_counter()
    transcription = model.transcribe(audio=audio)
    inference_time_seconds = time.perf_counter() - inference_start
    return TimedTranscription(
        model_id=model.model_id,
        words=transcription.words,
        model_loading_time_seconds=model.model_loading_time_seconds,
        inference_time_seconds=inference_time_seconds,
        inference_queue_time_seconds=transcription.inference_queue_time_seconds,
        audio_loading_time_seconds=transcription.audio_loading_time_seconds,
        model_execution_time_seconds=transcription.model_execution_time_seconds,
        package_names=model.package_names,
    )


def load_time_seconds(load_model: Callable[[], None]) -> float:
    load_start = time.perf_counter()
    load_model()
    return time.perf_counter() - load_start


def cuda_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for ASR inference.")
    return "cuda"
