from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.asr_quality.schemas import FileMetrics, SpeakerTrack, TranscriptionResult, Word


class AsrModelMode(StrEnum):
    PARAKEET_TDT = "parakeet_tdt_0_6b_v3"
    WHISPERX = "whisperx_large_v3"


class AsrModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: AsrModelMode
    label: str
    description: str


class AsrAnalysisRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    speaker_track: SpeakerTrack
    models: tuple[AsrModelMode, ...]
    reference_words: tuple[Word, ...] = ()


class AsrModelRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: AsrModelInfo
    transcription: TranscriptionResult
    metrics: FileMetrics | None


class AsrAnalysisResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    speaker_track: SpeakerTrack
    audio_url: str
    analyzed_duration_seconds: float
    reference_word_count: int
    runs: tuple[AsrModelRun, ...]


class AsrTranscriberContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_path: str
    speaker_track: SpeakerTrack
    audio_duration_seconds: float


class RunnerMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    model_identifier: str
    package_versions: dict[str, str] = Field(default_factory=dict)
