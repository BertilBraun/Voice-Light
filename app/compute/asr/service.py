from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import importlib.metadata
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from app.compute.asr.models.base import PreparedAsrAudio, TimedTranscription, prepare_asr_audio
from app.shared.asr import (
    AsrModelId,
    AsrRequestStats,
    AsrRuntimeStats,
    AsrTranscriptResult,
    RemoteAsrRequest,
    RemoteAsrResponse,
    RemoteAsrUploadRequest,
    TimestampedWord,
)
from app.shared.audio import probe_local_audio_metadata
from app.shared.audio.transport import (
    CANONICAL_MODEL_AUDIO_SPEC,
    TRANSPORT_DURATION_TOLERANCE_SECONDS,
    AudioTransportCodec,
    AudioTransportMetadata,
    encoded_audio_codec,
    prepare_audio_transport,
)

UPLOAD_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class UploadedAudioStats:
    sha256: str
    size_bytes: int


class RemoteAsrModelCache(Protocol):
    def transcribe_prepared(
        self,
        model_id: AsrModelId,
        audio: PreparedAsrAudio,
    ) -> TimedTranscription: ...


def transcribe_request(
    model_cache: RemoteAsrModelCache,
    request: RemoteAsrRequest,
) -> RemoteAsrResponse:
    audio_bytes = decode_audio(audio_base64=request.audio_base64)
    audio_suffix = Path(request.audio_filename).suffix
    if not audio_suffix:
        raise ValueError("ASR audio filename must have a file extension.")
    with tempfile.TemporaryDirectory(prefix="voice-light-asr-request-") as directory_name:
        directory = Path(directory_name)
        source_path = directory / f"source{audio_suffix}"
        source_path.write_bytes(audio_bytes)
        canonical_path = directory / "canonical.flac"
        prepare_audio_transport(
            source_path=source_path,
            output_path=canonical_path,
            spec=CANONICAL_MODEL_AUDIO_SPEC,
        )
        audio = prepare_asr_audio(canonical_path)
        return RemoteAsrResponse(
            results=transcribe_requested_models(
                model_cache=model_cache,
                model_ids=request.models,
                audio=audio,
            )
        )


async def transcribe_uploaded_request(
    model_cache: RemoteAsrModelCache,
    request_json: str,
    audio: UploadFile,
) -> RemoteAsrResponse:
    request_start = time.perf_counter()
    try:
        request = RemoteAsrUploadRequest.model_validate_json(request_json)
    except ValidationError as error:
        raise HTTPException(status_code=400, detail="Invalid ASR upload metadata.") from error
    if audio.filename != request.audio.encoded_filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded ASR filename does not match its metadata.",
        )
    suffix = suffix_for_codec(request.audio.codec)
    with tempfile.TemporaryDirectory(prefix="voice-light-asr-upload-") as directory_name:
        directory = Path(directory_name)
        upload_path = directory / f"transport{suffix}"
        upload_stream_start = time.perf_counter()
        upload_stats = await write_uploaded_audio(upload=audio, path=upload_path)
        upload_stream_time_seconds = time.perf_counter() - upload_stream_start
        upload_validation_start = time.perf_counter()
        validate_uploaded_audio(
            path=upload_path,
            declared=request.audio,
            uploaded=upload_stats,
        )
        upload_validation_time_seconds = time.perf_counter() - upload_validation_start
        canonical_path = directory / "canonical.flac"
        audio_preparation_start = time.perf_counter()
        canonical_metadata = await asyncio.to_thread(
            prepare_audio_transport,
            source_path=upload_path,
            output_path=canonical_path,
            spec=CANONICAL_MODEL_AUDIO_SPEC,
        )
        validate_canonical_duration(
            declared_duration_seconds=request.audio.duration_seconds,
            canonical_duration_seconds=canonical_metadata.duration_seconds,
        )
        prepared_audio = await asyncio.to_thread(prepare_asr_audio, canonical_path)
        audio_preparation_time_seconds = time.perf_counter() - audio_preparation_start
        transcription_start = time.perf_counter()
        results = await asyncio.to_thread(
            transcribe_requested_models,
            model_cache=model_cache,
            model_ids=request.models,
            audio=prepared_audio,
        )
        transcription_time_seconds = time.perf_counter() - transcription_start
        return RemoteAsrResponse(
            results=results,
            request_stats=AsrRequestStats(
                upload_stream_time_seconds=upload_stream_time_seconds,
                upload_validation_time_seconds=upload_validation_time_seconds,
                audio_preparation_time_seconds=audio_preparation_time_seconds,
                transcription_time_seconds=transcription_time_seconds,
                request_time_seconds=time.perf_counter() - request_start,
            ),
        )


async def write_uploaded_audio(upload: UploadFile, path: Path) -> UploadedAudioStats:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("wb") as output_file:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            output_file.write(chunk)
            digest.update(chunk)
            size_bytes += len(chunk)
    return UploadedAudioStats(sha256=digest.hexdigest(), size_bytes=size_bytes)


def validate_uploaded_audio(
    path: Path,
    declared: AudioTransportMetadata,
    uploaded: UploadedAudioStats,
) -> None:
    if uploaded.sha256 != declared.sha256:
        raise HTTPException(status_code=400, detail="Uploaded ASR audio SHA-256 mismatch.")
    if uploaded.size_bytes != declared.size_bytes:
        raise HTTPException(status_code=400, detail="Uploaded ASR audio size mismatch.")
    if (
        declared.sample_rate != CANONICAL_MODEL_AUDIO_SPEC.sample_rate
        or declared.channels != CANONICAL_MODEL_AUDIO_SPEC.channels
    ):
        raise HTTPException(
            status_code=400,
            detail="ASR upload metadata must declare mono 16 kHz audio.",
        )
    try:
        codec = encoded_audio_codec(path)
        uploaded_metadata = probe_local_audio_metadata(path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if codec is not declared.codec:
        raise HTTPException(status_code=400, detail="Uploaded ASR audio codec mismatch.")
    if uploaded_metadata.channels != declared.channels:
        raise HTTPException(status_code=400, detail="Uploaded ASR channel count mismatch.")
    validate_canonical_duration(
        declared_duration_seconds=declared.duration_seconds,
        canonical_duration_seconds=uploaded_metadata.duration_seconds,
    )


def validate_canonical_duration(
    declared_duration_seconds: float,
    canonical_duration_seconds: float,
) -> None:
    difference_seconds = abs(declared_duration_seconds - canonical_duration_seconds)
    if difference_seconds > TRANSPORT_DURATION_TOLERANCE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Canonical ASR audio duration differs from upload metadata by "
                f"{difference_seconds:.3f} seconds."
            ),
        )


def suffix_for_codec(codec: AudioTransportCodec) -> str:
    match codec:
        case AudioTransportCodec.FLAC:
            return ".flac"
        case AudioTransportCodec.OGG_OPUS:
            return ".ogg"


def transcribe_requested_models(
    model_cache: RemoteAsrModelCache,
    model_ids: tuple[AsrModelId, ...],
    audio: PreparedAsrAudio,
) -> tuple[AsrTranscriptResult, ...]:
    return tuple(
        transcribe_model(
            model_cache=model_cache,
            model_id=model_id,
            audio=audio,
        )
        for model_id in model_ids
    )


def transcribe_model(
    model_cache: RemoteAsrModelCache,
    model_id: AsrModelId,
    audio: PreparedAsrAudio,
) -> AsrTranscriptResult:
    reset_peak_gpu_memory_stats()
    return transcript_result_from_model_output(
        transcription=model_cache.transcribe_prepared(model_id=model_id, audio=audio),
        audio_duration_seconds=audio.duration_seconds,
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
        language_estimate=transcription.language_estimate,
        processing_time_seconds=processing_time_seconds,
        error=None,
        runtime=AsrRuntimeStats(
            processing_time_seconds=processing_time_seconds,
            model_loading_time_seconds=transcription.model_loading_time_seconds,
            inference_time_seconds=transcription.inference_time_seconds,
            inference_queue_time_seconds=transcription.inference_queue_time_seconds,
            audio_loading_time_seconds=transcription.audio_loading_time_seconds,
            model_execution_time_seconds=transcription.model_execution_time_seconds,
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
