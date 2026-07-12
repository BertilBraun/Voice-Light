from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from app.frozen_base_config import FrozenBaseModel
from app.quality.models import AudioMetadata, QualityResult


class UploadedAudioSource(FrozenBaseModel):
    kind: Literal["upload"] = "upload"
    filename: str
    audio_base64: str


class UriAudioSource(FrozenBaseModel):
    kind: Literal["uri"] = "uri"
    uri: str


AudioSource = Annotated[UploadedAudioSource | UriAudioSource, Field(discriminator="kind")]


class RemoteQualityRequest(FrozenBaseModel):
    sample_id: str
    speaker1: AudioSource
    speaker2: AudioSource


class RemoteQualityResponse(FrozenBaseModel):
    speaker1_metadata: AudioMetadata
    speaker2_metadata: AudioMetadata
    quality_result: QualityResult
