from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.local.config import DATABASE_URL
from app.local.synchronization_review.models import SynchronizationCandidateListResponse
from app.local.synchronization_review.repository import SynchronizationReviewRepository
from app.local.synchronization_review.service import synchronization_candidates

router = APIRouter(
    prefix="/api/synchronization-review",
    tags=["synchronization-review"],
)


@router.get("/candidates")
def list_synchronization_candidates() -> SynchronizationCandidateListResponse:
    try:
        return cached_synchronization_candidates(DATABASE_URL)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@lru_cache(maxsize=1)
def cached_synchronization_candidates(
    database_url: str,
) -> SynchronizationCandidateListResponse:
    if not database_url:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for synchronization review APIs.")
    return synchronization_candidates(
        repository=SynchronizationReviewRepository(database_url=database_url)
    )
