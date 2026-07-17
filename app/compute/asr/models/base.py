from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from app.shared.asr import AsrModelId, TimestampedWord


class LoadedAsrModel(Protocol):
    @property
    def model_id(self) -> AsrModelId: ...

    @property
    def package_names(self) -> tuple[str, ...]: ...

    @property
    def model_loading_time_seconds(self) -> float: ...

    def transcribe(self, audio_path: Path) -> ModelTranscription: ...


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


def timed_transcription(model: LoadedAsrModel, audio_path: Path) -> TimedTranscription:
    inference_start = time.perf_counter()
    transcription = model.transcribe(audio_path=audio_path)
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
