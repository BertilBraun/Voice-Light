from __future__ import annotations

import base64
import binascii
import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Annotated

import modal
import torch
from fastapi import Header, HTTPException

from app.asr.models.base import TimedTranscription
from app.asr.models.registry import AsrModelCache
from app.asr.schemas import (
    AsrModelId,
    AsrRuntimeStats,
    AsrTranscriptResult,
    RemoteAsrRequest,
    RemoteAsrResponse,
    TimestampedWord,
)
from app.quality.audio import load_audio

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "fastapi[standard]>=0.115.0",
        "faster-whisper>=1.1.0",
        "librosa>=0.11.0",
        "numpy>=2.0.0",
        "pydantic>=2.0.0",
        "torch>=2.13.0",
        "transformers>=5.13.0",
        "whisperx>=3.7.0",
    )
    .add_local_python_source("app")
)

app = modal.App("VoiceLight")


@app.cls(
    image=modal_image,
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("voice-light-asr")],
)
@modal.concurrent(max_inputs=20, target_inputs=4)
class AsrModelServer:
    @modal.enter()
    def setup(self) -> None:
        self.model_cache = AsrModelCache()

    @modal.fastapi_endpoint(method="POST")
    def transcribe(
        self,
        request: RemoteAsrRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> RemoteAsrResponse:
        authorize_request(authorization=authorization)
        audio_bytes = decode_audio(audio_base64=request.audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_file.write(audio_bytes)
            audio_path = Path(audio_file.name)
        try:
            audio_duration_seconds = load_audio(audio_path).metadata.duration_seconds
            results = tuple(
                transcribe_model(
                    model_cache=self.model_cache,
                    model_id=model_id,
                    audio_path=audio_path,
                    audio_duration_seconds=audio_duration_seconds,
                )
                for model_id in request.models
            )
            return RemoteAsrResponse(results=results)
        finally:
            audio_path.unlink(missing_ok=True)


def transcribe_model(
    model_cache: AsrModelCache,
    model_id: AsrModelId,
    audio_path: Path,
    audio_duration_seconds: float,
) -> AsrTranscriptResult:
    reset_peak_gpu_memory_stats()
    return transcript_result_from_model_output(
        transcription=model_cache.transcribe(
            model_id=model_id,
            audio_path=audio_path,
        ),
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


def authorize_request(authorization: str | None) -> None:
    expected_api_key = os.environ["VOICE_LIGHT_REMOTE_ASR_API_KEY"]
    if authorization != f"Bearer {expected_api_key}":
        raise HTTPException(status_code=401, detail="Invalid ASR API key.")


def decode_audio(audio_base64: str) -> bytes:
    try:
        return base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload.") from error


def transcript_text(words: tuple[TimestampedWord, ...]) -> str:
    return " ".join(word.text.strip() for word in words if word.text.strip())


def real_time_factor(processing_time_seconds: float, audio_duration_seconds: float) -> float | None:
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
