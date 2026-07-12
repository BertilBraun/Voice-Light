from __future__ import annotations

from enum import StrEnum

from app.asr_quality.schemas import FileMetrics, SpeakerTrack, TranscriptionResult, Word
from app.frozen_base_config import FrozenBaseModel


class AsrModelMode(StrEnum):
    PARAKEET_TDT = "parakeet_tdt_0_6b_v3"
    WHISPERX = "whisperx_large_v3"
    CANARY = "canary_1b_v2"
    NEMOTRON_3_5 = "nemotron_3_5_asr_streaming_0_6b"
    PARAKEET_CANARY_CONSENSUS = "parakeet_canary_consensus"
    MERGED_CONSENSUS = "merged_consensus"


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
