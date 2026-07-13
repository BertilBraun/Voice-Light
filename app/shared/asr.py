from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from app.shared.base_model import FrozenBaseModel

PARAKEET_IDENTIFIER = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_REVISION = "7c35754d166cca382ad1e53e68b01e7c575f3a1d"
WHISPER_IDENTIFIER = "Systran/faster-whisper-large-v3"
WHISPER_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
CANARY_IDENTIFIER = "nvidia/canary-1b-v2"
CANARY_REVISION = "87bc52657add533cd0156b3fc1aef027280754bf"
NEMOTRON_3_5_IDENTIFIER = "nvidia/nemotron-3.5-asr-streaming-0.6b"
NEMOTRON_3_5_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"


class AsrModelId(StrEnum):
    PARAKEET_TDT = "parakeet_tdt_0_6b_v3"
    WHISPERX = "whisperx_large_v3"
    CANARY = "canary_1b_v2"
    NEMOTRON_3_5 = "nemotron_3_5_asr_streaming_0_6b"


class TimestampedWord(FrozenBaseModel):
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


class AsrRuntimeStats(FrozenBaseModel):
    processing_time_seconds: float
    model_loading_time_seconds: float | None = None
    inference_time_seconds: float | None = None
    real_time_factor: float | None = None
    peak_gpu_memory_mb: float | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)


class AsrTranscriptResult(FrozenBaseModel):
    model_id: AsrModelId
    text: str
    words: tuple[TimestampedWord, ...]
    processing_time_seconds: float | None = None
    error: str | None = None
    runtime: AsrRuntimeStats | None = None


class CachedAsrRequest(FrozenBaseModel):
    audio_path: str
    models: tuple[AsrModelId, ...] = Field(min_length=1)


class CachedAsrResponse(FrozenBaseModel):
    audio_sha256: str
    results: tuple[AsrTranscriptResult, ...]


class RemoteAsrRequest(FrozenBaseModel):
    audio_sha256: str
    audio_base64: str
    models: tuple[AsrModelId, ...] = Field(min_length=1)


class RemoteAsrResponse(FrozenBaseModel):
    results: tuple[AsrTranscriptResult, ...]
