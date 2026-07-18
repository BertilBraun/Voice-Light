from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field

from app.shared.base_model import FrozenBaseModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"
SIDECAR_SCHEMA_VERSION = "voice-light.audio-alignment/v1"


class AlignmentSide(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"


class AlignmentApplicationStatus(StrEnum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    RECOVERED = "recovered"


class AlignmentTrackRewrite(FrozenBaseModel):
    side: AlignmentSide
    filename: str = Field(min_length=1)
    original_sha256: str = Field(pattern=SHA256_PATTERN)
    aligned_sha256: str = Field(pattern=SHA256_PATTERN)
    original_frame_count: int = Field(ge=0)
    aligned_frame_count: int = Field(ge=0)
    prepended_silence_frame_count: int = Field(ge=0)
    appended_silence_frame_count: int = Field(ge=0)


class AlignmentSidecar(FrozenBaseModel):
    schema_version: Literal["voice-light.audio-alignment/v1"] = SIDECAR_SCHEMA_VERSION
    sample_external_id: str = Field(min_length=1)
    reviewed_speaker2_shift_seconds: float
    sample_rate: int = Field(gt=0)
    applied_speaker2_shift_frame_count: int
    speaker1: AlignmentTrackRewrite
    speaker2: AlignmentTrackRewrite
    applied_at: AwareDatetime
