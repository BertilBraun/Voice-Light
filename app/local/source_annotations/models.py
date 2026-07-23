from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.shared.base_model import FrozenBaseModel


class SourceAnnotationEvent(FrozenBaseModel):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    label: str
    text: str


class SourceAnnotationTrack(FrozenBaseModel):
    annotator_id: str
    events: tuple[SourceAnnotationEvent, ...]


class SourceSampleAnnotations(FrozenBaseModel):
    annotation_version: str
    speaker_1: tuple[SourceAnnotationTrack, ...]
    speaker_2: tuple[SourceAnnotationTrack, ...]


class SourceAnnotationImport(FrozenBaseModel):
    sample_track_id: UUID
    source_name: str
    annotation_version: str
    annotator_id: str
    annotation: SourceAnnotationTrack


class RegisteredSourceAnnotationSample(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    speaker_1_track_id: UUID
    speaker_2_track_id: UUID
