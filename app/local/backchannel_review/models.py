from __future__ import annotations

from uuid import UUID

from app.shared.base_model import FrozenBaseModel
from app.shared.quality import (
    AnnotationPoint,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    SegmentAnnotationTarget,
    SpeakerSide,
)


class ReviewWindowAnnotation(FrozenBaseModel):
    side: SpeakerSide
    speech_segments: tuple[AnnotationSpan, ...]
    pauses: tuple[AnnotationSpan, ...]
    backchannels: tuple[AnnotationSpan, ...]
    turns: tuple[AnnotationPoint, ...]
    interruptions: tuple[AnnotationPoint, ...]
    segment_targets: tuple[SegmentAnnotationTarget, ...]
    connection_targets: tuple[ConnectionAnnotationTarget, ...]


class BackchannelReviewCandidate(FrozenBaseModel):
    sample_id: UUID
    external_id: str
    floor_holder_side: SpeakerSide
    possible_backchannel_side: SpeakerSide
    window_start_seconds: float
    window_end_seconds: float
    floor_holder_before: SegmentAnnotationTarget
    possible_backchannel: SegmentAnnotationTarget
    floor_holder_after: SegmentAnnotationTarget
    floor_holder_connection: ConnectionAnnotationTarget
    speaker1: ReviewWindowAnnotation
    speaker2: ReviewWindowAnnotation


class BackchannelReviewListResponse(FrozenBaseModel):
    candidates: tuple[BackchannelReviewCandidate, ...]
    offset: int
    next_offset: int | None
