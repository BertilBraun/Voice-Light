from __future__ import annotations

from app.asr.schemas import AsrModelId
from app.asr_quality.schemas import FileMetrics, SpeakerTrack, TranscriptionResult, Word
from app.frozen_base_config import FrozenBaseModel

AsrModelMode = AsrModelId


class AsrModelInfo(FrozenBaseModel):
    mode: AsrModelMode
    label: str
    description: str


class AsrAnalysisRequest(FrozenBaseModel):
    session_id: str
    speaker_track: SpeakerTrack
    models: tuple[AsrModelMode, ...]
    reference_words: tuple[Word, ...] = ()


class AsrModelRun(FrozenBaseModel):
    model: AsrModelInfo
    transcription: TranscriptionResult
    metrics: FileMetrics | None


class AsrAnalysisResponse(FrozenBaseModel):
    session_id: str
    speaker_track: SpeakerTrack
    audio_url: str
    analyzed_duration_seconds: float
    reference_word_count: int
    runs: tuple[AsrModelRun, ...]
