from __future__ import annotations

from enum import StrEnum
from typing import Literal

from app.shared.base_model import FrozenBaseModel
from app.shared.quality import AudioMetadata, QualityResult


class HealthStatus(StrEnum):
    LIVE = "live"
    READY = "ready"
    NOT_READY = "not_ready"


class ModelStageStatus(StrEnum):
    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ModelStage(FrozenBaseModel):
    name: str
    status: ModelStageStatus
    load_time_seconds: float | None = None
    error: str | None = None


class GpuMemory(FrozenBaseModel):
    current_allocated_mb: float | None
    peak_allocated_mb: float | None


class LivenessResponse(FrozenBaseModel):
    status: Literal[HealthStatus.LIVE] = HealthStatus.LIVE


class ReadinessResponse(FrozenBaseModel):
    status: Literal[HealthStatus.READY, HealthStatus.NOT_READY]
    stages: tuple[ModelStage, ...]
    gpu_memory: GpuMemory


class QualityAnalysisUpload(FrozenBaseModel):
    sample_id: str
    speaker1_original_metadata: AudioMetadata | None
    speaker2_original_metadata: AudioMetadata | None


class QualityAnalysisResponse(FrozenBaseModel):
    speaker1_metadata: AudioMetadata
    speaker2_metadata: AudioMetadata
    quality_result: QualityResult
