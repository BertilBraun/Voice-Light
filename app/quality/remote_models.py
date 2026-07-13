from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from app.frozen_base_config import FrozenBaseModel
from app.quality.models import AudioMetadata, QualityResult

QUALITY_INPUT_VOLUME_COUNT = 4


def quality_input_volume_name(volume_index: int) -> str:
    return f"voice-light-quality-inputs-{volume_index}"


class LocalAudioSource(FrozenBaseModel):
    kind: Literal["local"] = "local"
    filename: str
    path: str


class VolumeAudioSource(FrozenBaseModel):
    kind: Literal["volume"] = "volume"
    volume_index: int = Field(ge=0, lt=QUALITY_INPUT_VOLUME_COUNT)
    path: str


class UriAudioSource(FrozenBaseModel):
    kind: Literal["uri"] = "uri"
    uri: str


AudioSource = Annotated[
    LocalAudioSource | VolumeAudioSource | UriAudioSource,
    Field(discriminator="kind"),
]


class RemoteQualityRequest(FrozenBaseModel):
    sample_id: str
    speaker1: AudioSource
    speaker2: AudioSource


class RemoteQualityResponse(FrozenBaseModel):
    speaker1_metadata: AudioMetadata
    speaker2_metadata: AudioMetadata
    quality_result: QualityResult
