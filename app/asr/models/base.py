from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.asr.schemas import AsrModelId, TimestampedWord
from app.asr_quality.schemas import Word


class LoadedAsrModel(Protocol):
    @property
    def model_id(self) -> AsrModelId: ...

    @property
    def package_names(self) -> tuple[str, ...]: ...

    @property
    def model_loading_time_seconds(self) -> float: ...

    def transcribe(self, audio_path: Path) -> tuple[TimestampedWord, ...]: ...


@dataclass(frozen=True)
class TimedTranscription:
    model_id: AsrModelId
    words: tuple[TimestampedWord, ...]
    model_loading_time_seconds: float
    inference_time_seconds: float
    package_names: tuple[str, ...]


def timed_transcription(model: LoadedAsrModel, audio_path: Path) -> TimedTranscription:
    inference_start = time.perf_counter()
    words = model.transcribe(audio_path=audio_path)
    inference_time_seconds = time.perf_counter() - inference_start
    return TimedTranscription(
        model_id=model.model_id,
        words=words,
        model_loading_time_seconds=model.model_loading_time_seconds,
        inference_time_seconds=inference_time_seconds,
        package_names=model.package_names,
    )


def load_time_seconds(load_model: Callable[[], None]) -> float:
    load_start = time.perf_counter()
    load_model()
    return time.perf_counter() - load_start


def timestamped_word_from_word(word: Word) -> TimestampedWord:
    return TimestampedWord(
        text=word.text,
        start_seconds=word.start_seconds,
        end_seconds=word.end_seconds,
        confidence=word.confidence,
    )
