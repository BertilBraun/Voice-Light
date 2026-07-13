from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from app.shared.base_model import FrozenBaseModel
from app.shared.quality import AudioMetadata


class LocalAudioSource(FrozenBaseModel):
    kind: Literal["local"] = "local"
    filename: str
    path: str
    original_metadata: AudioMetadata


class UriAudioSource(FrozenBaseModel):
    kind: Literal["uri"] = "uri"
    uri: str


AudioSource = Annotated[
    LocalAudioSource | UriAudioSource,
    Field(discriminator="kind"),
]


class RemoteQualityRequest(FrozenBaseModel):
    sample_id: str
    speaker1: AudioSource
    speaker2: AudioSource
