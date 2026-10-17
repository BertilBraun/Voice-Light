from __future__ import annotations

import wave
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from app.compute.voice.client_metadata import VoiceClientMetadata
from app.compute.voice.conversation import ConversationMessage
from app.compute.voice.overlap import OverlapMetricsReport
from app.compute.voice.playback import PlaybackMetricsReport
from app.compute.voice.predictive import (
    CandidateInvalidationReason,
    CandidateLifecycle,
    PredictiveMetricsReport,
)
from app.compute.voice.tools import (
    SerializedToolCall,
    ToolExecutionFailure,
    ToolInvalidationReason,
    ToolLifecycle,
    ToolResultCommitStatus,
    ToolSuccess,
)


class SessionClosureReason(StrEnum):
    CLIENT_DISCONNECTED = "client_disconnected"
    CLIENT_STOPPED = "client_stopped"
    START_TIMEOUT = "start_timeout"
    INACTIVITY_TIMEOUT = "inactivity_timeout"
    MAXIMUM_DURATION = "maximum_duration"
    SESSION_FAILURE = "session_failure"


@dataclass(frozen=True)
class SessionCaptureConfig:
    directory: Path
    retention_days: int = 30

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Self | None:
        directory_value = environment.get("VOICE_LIGHT_SESSION_CAPTURE_DIRECTORY")
        if directory_value is None:
            return None
        if not directory_value.strip():
            raise ValueError("VOICE_LIGHT_SESSION_CAPTURE_DIRECTORY cannot be empty.")
        retention_value = environment.get("VOICE_LIGHT_SESSION_CAPTURE_RETENTION_DAYS", "30")
        try:
            retention_days = int(retention_value)
        except ValueError as error:
            raise ValueError(
                "VOICE_LIGHT_SESSION_CAPTURE_RETENTION_DAYS must be an integer."
            ) from error
        return cls(directory=Path(directory_value), retention_days=retention_days)

    def __post_init__(self) -> None:
        if self.retention_days <= 0:
            raise ValueError("The session capture retention must be positive.")


class SessionCaptureTool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int
    invocation_id: int
    call_id: str
    lifecycle: ToolLifecycle
    result_commit_status: ToolResultCommitStatus
    request: SerializedToolCall | None
    outcome: ToolSuccess | ToolExecutionFailure | None
    invalidation_reason: ToolInvalidationReason | None
    wasted: bool


class SessionCaptureGeneration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    generation_id: int
    user_turn_id: int
    speculative: bool
    lifecycle: CandidateLifecycle
    invalidation_reason: CandidateInvalidationReason | None
    response_text: str
    acknowledged_text_offset: int
    qwen_token_count: int
    synthesized_sample_count: int


class SessionCaptureRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    session_id: str
    request_id: str
    connected_at: datetime
    closed_at: datetime
    duration_seconds: float
    closure_reason: SessionClosureReason
    local_time_zone: str | None
    client: VoiceClientMetadata
    input_sample_rate: int
    input_sample_count: int
    audio_file: str
    conversation: tuple[ConversationMessage, ...]
    tools: tuple[SessionCaptureTool, ...]
    generations: tuple[SessionCaptureGeneration, ...]
    predictive_metrics: PredictiveMetricsReport
    playback_metrics: PlaybackMetricsReport
    overlap_metrics: OverlapMetricsReport


class SessionCaptureStore:
    def __init__(self, config: SessionCaptureConfig) -> None:
        self.config = config

    def save(self, record: SessionCaptureRecord, pcm16: bytes) -> None:
        self.config.directory.mkdir(parents=True, exist_ok=True)
        audio_path = self.config.directory / record.audio_file
        json_path = self.config.directory / f"voice-session-{record.session_id}.json"
        _write_wave(audio_path, pcm16, record.input_sample_rate)
        json_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        self._prune_expired_files(record.closed_at)

    def _prune_expired_files(self, reference_time: datetime) -> None:
        cutoff = reference_time - timedelta(days=self.config.retention_days)
        for capture_path in self.config.directory.glob("voice-session-*.*"):
            modified_at = datetime.fromtimestamp(capture_path.stat().st_mtime, tz=UTC)
            if modified_at < cutoff:
                capture_path.unlink()


def _write_wave(path: Path, pcm16: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(pcm16)
