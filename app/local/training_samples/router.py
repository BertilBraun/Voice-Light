from __future__ import annotations

import random
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.local.config import DATABASE_URL
from app.local.db.models import TrackSide
from app.local.db.repository import Repository
from app.local.training_samples.models import TrainingSamplePreview
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
