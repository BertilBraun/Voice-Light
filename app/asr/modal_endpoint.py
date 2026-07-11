from __future__ import annotations

import base64
import binascii
import importlib.metadata
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Annotated, Protocol

import librosa
import modal
import torch
import whisperx
from fastapi import Header, HTTPException
from faster_whisper import WhisperModel
from transformers import AutoModelForTDT, AutoProcessor

from app.analyses.asr.runners import (
    PARAKEET_IDENTIFIER,
    words_from_parakeet_timestamps,
    words_from_whisperx_output,
)
from app.asr.schemas import (
    AsrModelId,
    AsrRuntimeStats,
    AsrTranscriptResult,
    RemoteAsrRequest,
    RemoteAsrResponse,
    TranscriptSegment,
)
from app.asr_quality.schemas import Word
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


class LoadedAsrModel(Protocol):
    def transcribe(
        self, audio_path: Path, audio_duration_seconds: float
    ) -> AsrTranscriptResult: ...


class ModalAsrModelCache:
    def __init__(self) -> None:
        self.load_lock = Lock()
        self.loaded_models: dict[AsrModelId, LoadedAsrModel] = {}

    def transcribe(
        self,
        model_id: AsrModelId,
        audio_path: Path,
        audio_duration_seconds: float,
    ) -> AsrTranscriptResult:
        model = self.loaded_model(model_id=model_id)
        return model.transcribe(
            audio_path=audio_path,
            audio_duration_seconds=audio_duration_seconds,
        )

    def loaded_model(self, model_id: AsrModelId) -> LoadedAsrModel:
        with self.load_lock:
            cached_model = self.loaded_models.get(model_id)
            if cached_model is not None:
                return cached_model
            loaded_model = load_model(model_id=model_id)
            self.loaded_models[model_id] = loaded_model
            return loaded_model


class ParakeetLoadedAsrModel:
    def __init__(self) -> None:
        load_start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(PARAKEET_IDENTIFIER)
        self.model = AutoModelForTDT.from_pretrained(PARAKEET_IDENTIFIER, dtype="auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self.model_loading_time_seconds = time.perf_counter() - load_start
        self.inference_lock = Lock()

    def transcribe(self, audio_path: Path, audio_duration_seconds: float) -> AsrTranscriptResult:
        with self.inference_lock:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            processing_start = time.perf_counter()
            inference_start = time.perf_counter()
            audio, _sample_rate = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            inputs = self.processor([audio], sampling_rate=self.sample_rate)
            inputs.to(self.model.device, dtype=self.model.dtype)
            output = self.model.generate(**inputs, return_dict_in_generate=True)
            _decoded_output, decoded_timestamps = self.processor.decode(
                output.sequences,
                durations=output.durations,
                skip_special_tokens=True,
            )
            inference_time_seconds = time.perf_counter() - inference_start
            processing_time_seconds = time.perf_counter() - processing_start
        words = tuple(words_from_parakeet_timestamps(decoded_timestamps))
        return asr_transcript_result(
            model_id=AsrModelId.PARAKEET_TDT,
            words=words,
            processing_time_seconds=processing_time_seconds,
            model_loading_time_seconds=self.model_loading_time_seconds,
            inference_time_seconds=inference_time_seconds,
            audio_duration_seconds=audio_duration_seconds,
            error=None,
        )


class WhisperxLoadedAsrModel:
    def __init__(self) -> None:
        load_start = time.perf_counter()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel("large-v3", device=self.device, compute_type=self.compute_type)
        self.model_loading_time_seconds = time.perf_counter() - load_start
        self.alignment_models: dict[str, tuple[object, object]] = {}
        self.alignment_load_lock = Lock()
        self.inference_lock = Lock()

    def transcribe(self, audio_path: Path, audio_duration_seconds: float) -> AsrTranscriptResult:
        with self.inference_lock:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            processing_start = time.perf_counter()
            inference_start = time.perf_counter()
            segments, transcription_info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=False,
                vad_filter=False,
            )
            transcription_segments = [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segments
            ]
            language_code = str(transcription_info.language)
            align_model, metadata = self.alignment_model_for_language(language_code=language_code)
            audio = whisperx.load_audio(str(audio_path))
            aligned = whisperx.align(
                transcription_segments,
                align_model,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            inference_time_seconds = time.perf_counter() - inference_start
            processing_time_seconds = time.perf_counter() - processing_start
        words = tuple(words_from_whisperx_output(aligned))
        return asr_transcript_result(
            model_id=AsrModelId.WHISPERX,
            words=words,
            processing_time_seconds=processing_time_seconds,
            model_loading_time_seconds=self.model_loading_time_seconds,
            inference_time_seconds=inference_time_seconds,
            audio_duration_seconds=audio_duration_seconds,
            error=None,
        )

    def alignment_model_for_language(self, language_code: str) -> tuple[object, object]:
        with self.alignment_load_lock:
            cached_alignment_model = self.alignment_models.get(language_code)
            if cached_alignment_model is not None:
                return cached_alignment_model
            alignment_model = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
            )
            self.alignment_models[language_code] = alignment_model
            return alignment_model


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
        self.model_cache = ModalAsrModelCache()

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
                self.model_cache.transcribe(
                    model_id=model_id,
                    audio_path=audio_path,
                    audio_duration_seconds=audio_duration_seconds,
                )
                for model_id in request.models
            )
            return RemoteAsrResponse(results=results)
        finally:
            audio_path.unlink(missing_ok=True)


def load_model(model_id: AsrModelId) -> LoadedAsrModel:
    match model_id:
        case AsrModelId.PARAKEET_TDT:
            return ParakeetLoadedAsrModel()
        case AsrModelId.WHISPERX:
            return WhisperxLoadedAsrModel()


def authorize_request(authorization: str | None) -> None:
    expected_api_key = os.environ["VOICE_LIGHT_REMOTE_ASR_API_KEY"]
    if authorization != f"Bearer {expected_api_key}":
        raise HTTPException(status_code=401, detail="Invalid ASR API key.")


def decode_audio(audio_base64: str) -> bytes:
    try:
        return base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload.") from error


def asr_transcript_result(
    model_id: AsrModelId,
    words: tuple[Word, ...],
    processing_time_seconds: float,
    model_loading_time_seconds: float,
    inference_time_seconds: float,
    audio_duration_seconds: float,
    error: str | None,
) -> AsrTranscriptResult:
    return AsrTranscriptResult(
        model_id=model_id,
        text=transcript_text(words=words),
        segments=tuple(
            TranscriptSegment(
                text=word.text,
                start_seconds=word.start_seconds,
                end_seconds=word.end_seconds,
                confidence=word.confidence,
            )
            for word in words
        ),
        processing_time_seconds=processing_time_seconds,
        error=error,
        runtime=remote_asr_runtime_stats(
            processing_time_seconds=processing_time_seconds,
            model_loading_time_seconds=model_loading_time_seconds,
            inference_time_seconds=inference_time_seconds,
            audio_duration_seconds=audio_duration_seconds,
        ),
    )


def remote_asr_runtime_stats(
    processing_time_seconds: float,
    model_loading_time_seconds: float,
    inference_time_seconds: float,
    audio_duration_seconds: float,
) -> AsrRuntimeStats:
    return AsrRuntimeStats(
        processing_time_seconds=processing_time_seconds,
        model_loading_time_seconds=model_loading_time_seconds,
        inference_time_seconds=inference_time_seconds,
        real_time_factor=processing_time_seconds / audio_duration_seconds
        if audio_duration_seconds > 0.0
        else None,
        peak_gpu_memory_mb=peak_gpu_memory_mb(),
        package_versions=package_versions(
            ("torch", "transformers", "librosa", "whisperx", "faster-whisper")
        ),
    )


def transcript_text(words: tuple[Word, ...]) -> str:
    return " ".join(word.text.strip() for word in words if word.text.strip())


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
