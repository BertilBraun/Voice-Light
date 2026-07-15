from __future__ import annotations

import random
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.local.config import DATABASE_URL
from app.local.db.models import TrackSide
from app.local.db.repository import Repository
from app.local.training_samples.models import TrainingSampleOption, TrainingSamplePreview
from app.local.training_samples.service import build_training_sample_preview

router = APIRouter(prefix="/api/training-samples", tags=["training-samples"])


def repository() -> Repository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for training sample APIs.")
    return Repository(DATABASE_URL)


@router.get("/preview")
def preview_training_sample(
    sample_id: UUID,
    user_side: TrackSide = TrackSide.SPEAKER2,
    start_seconds: float | None = Query(default=None, ge=0.0),
) -> TrainingSamplePreview:
    try:
        dashboard_sample = repository().get_dashboard_sample(sample_id)
        return build_training_sample_preview(
            dashboard_sample=dashboard_sample,
            user_side=user_side,
            requested_start_seconds=start_seconds,
            generator=random.SystemRandom(),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/options")
def list_training_sample_options(
    limit: int = Query(default=40, ge=1, le=100),
) -> tuple[TrainingSampleOption, ...]:
    records = repository().list_annotated_samples(limit)
    return tuple(
        TrainingSampleOption(
            sample_id=record.sample_id,
            external_id=record.external_id,
            represented_duration_seconds=record.represented_duration_seconds,
            usable_event_count=record.usable_event_count,
        )
        for record in records
    )


@router.get("/random-preview")
def random_training_sample_preview(current_sample_id: UUID) -> TrainingSamplePreview:
    try:
        sample_repository = repository()
        sample_id = sample_repository.random_annotated_sample_id(current_sample_id)
        generator = random.SystemRandom()
        user_sides = (TrackSide.SPEAKER1, TrackSide.SPEAKER2)
        user_side = user_sides[generator.randrange(len(user_sides))]
        return build_training_sample_preview(
            dashboard_sample=sample_repository.get_dashboard_sample(sample_id),
            user_side=user_side,
            requested_start_seconds=None,
            generator=generator,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
