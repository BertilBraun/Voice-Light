from __future__ import annotations

import base64
import binascii
import importlib.metadata
import tempfile
from pathlib import Path
from typing import Protocol

import torch
from fastapi import HTTPException

from app.compute.asr.models.base import TimedTranscription
from app.shared.asr import (
    AsrModelId,
    AsrRuntimeStats,
    AsrTranscriptResult,
    RemoteAsrRequest,
    RemoteAsrResponse,
    TimestampedWord,
)
from app.shared.audio import load_audio
from app.shared.storage.local import LocalStorageBackend


class RemoteAsrModelCache(Protocol):
    def transcribe(self, model_id: AsrModelId, audio_path: Path) -> TimedTranscription: ...


def transcribe_request(
    model_cache: RemoteAsrModelCache,
    request: RemoteAsrRequest,
) -> RemoteAsrResponse:
    audio_bytes = decode_audio(audio_base64=request.audio_base64)
    audio_suffix = Path(request.audio_filename).suffix
    if not audio_suffix:
        raise ValueError("ASR audio filename must have a file extension.")
    with tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = Path(audio_file.name)
    try:
        audio_duration_seconds = load_audio(
            LocalStorageBackend(),
            audio_path.as_posix(),
        ).metadata.duration_seconds
        return RemoteAsrResponse(
            results=transcribe_requested_models(
                model_cache=model_cache,
                model_ids=request.models,
                audio_path=audio_path,
                audio_duration_seconds=audio_duration_seconds,
            )
        )
    finally:
        audio_path.unlink(missing_ok=True)


def transcribe_requested_models(
    model_cache: RemoteAsrModelCache,
    model_ids: tuple[AsrModelId, ...],
    audio_path: Path,
    audio_duration_seconds: float,
) -> tuple[AsrTranscriptResult, ...]:
    return tuple(
        transcribe_model(
            model_cache=model_cache,
            model_id=model_id,
            audio_path=audio_path,
            audio_duration_seconds=audio_duration_seconds,
        )
        for model_id in model_ids
    )


def transcribe_model(
    model_cache: RemoteAsrModelCache,
    model_id: AsrModelId,
    audio_path: Path,
    audio_duration_seconds: float,
) -> AsrTranscriptResult:
    reset_peak_gpu_memory_stats()
    return transcript_result_from_model_output(
        transcription=model_cache.transcribe(model_id=model_id, audio_path=audio_path),
        audio_duration_seconds=audio_duration_seconds,
    )


def transcript_result_from_model_output(
    transcription: TimedTranscription,
    audio_duration_seconds: float,
) -> AsrTranscriptResult:
    processing_time_seconds = transcription.inference_time_seconds
    return AsrTranscriptResult(
        model_id=transcription.model_id,
        text=transcript_text(words=transcription.words),
        words=transcription.words,
        processing_time_seconds=processing_time_seconds,
        error=None,
        runtime=AsrRuntimeStats(
            processing_time_seconds=processing_time_seconds,
            model_loading_time_seconds=transcription.model_loading_time_seconds,
            inference_time_seconds=transcription.inference_time_seconds,
            real_time_factor=real_time_factor(
                processing_time_seconds=processing_time_seconds,
                audio_duration_seconds=audio_duration_seconds,
            ),
            peak_gpu_memory_mb=peak_gpu_memory_mb(),
            package_versions=package_versions(transcription.package_names),
        ),
    )


def decode_audio(audio_base64: str) -> bytes:
    try:
        return base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload.") from error


def transcript_text(words: tuple[TimestampedWord, ...]) -> str:
    return " ".join(word.text.strip() for word in words if word.text.strip())


def real_time_factor(
    processing_time_seconds: float,
    audio_duration_seconds: float,
) -> float | None:
    if audio_duration_seconds <= 0.0:
        return None
    return processing_time_seconds / audio_duration_seconds


def package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def peak_gpu_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1_048_576


def reset_peak_gpu_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
