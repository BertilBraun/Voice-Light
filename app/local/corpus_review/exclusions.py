from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.shared.base_model import FrozenBaseModel


class CorpusExclusionScope(StrEnum):
    RECORDING = "recording"
    INTERVAL = "interval"


class RecordingCorpusExclusionRequest(FrozenBaseModel):
    scope: Literal[CorpusExclusionScope.RECORDING]
    reason: str = Field(min_length=1, max_length=4000)


class IntervalCorpusExclusionRequest(FrozenBaseModel):
    scope: Literal[CorpusExclusionScope.INTERVAL]
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    reason: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_interval(self) -> IntervalCorpusExclusionRequest:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Corpus exclusion end must follow its start.")
        return self


CorpusExclusionRequest = Annotated[
    RecordingCorpusExclusionRequest | IntervalCorpusExclusionRequest,
    Field(discriminator="scope"),
]


class CorpusIntervalExclusion(FrozenBaseModel):
    id: UUID
    review_item_id: UUID
    sample_id: UUID
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    reason: str
    created_at: datetime

    @model_validator(mode="after")
    def validate_interval(self) -> CorpusIntervalExclusion:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Corpus exclusion end must follow its start.")
        return self


class CorpusExclusionRecord(FrozenBaseModel):
    id: UUID
    review_item_id: UUID
    sample_id: UUID
    scope: CorpusExclusionScope
    start_seconds: float | None
    end_seconds: float | None
    reason: str
    created_at: datetime

    @model_validator(mode="after")
    def validate_scope(self) -> CorpusExclusionRecord:
        match self.scope:
            case CorpusExclusionScope.RECORDING:
                if self.start_seconds is not None or self.end_seconds is not None:
                    raise ValueError("Recording exclusions cannot define an interval.")
            case CorpusExclusionScope.INTERVAL:
                if self.start_seconds is None or self.end_seconds is None:
                    raise ValueError("Interval exclusions require both boundaries.")
                if self.end_seconds <= self.start_seconds:
                    raise ValueError("Corpus exclusion end must follow its start.")
        return self
