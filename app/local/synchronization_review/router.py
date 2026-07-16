from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.local.config import DATABASE_URL
from app.local.db.models import DashboardSample, TrackSide
from app.local.db.repository import Repository
from app.local.synchronization_review.models import (
    SynchronizationCandidateListResponse,
    SynchronizationGainResponse,
    SynchronizationReviewRequest,
    SynchronizationReviewSaveResponse,
)
from app.local.synchronization_review.repository import SynchronizationReviewRepository
from app.local.synchronization_review.service import (
    speech_only_gain_normalization,
    synchronization_candidates,
)
from app.shared.quality import QualityResult

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


@router.get("/gain/{sample_id}")
def synchronization_gain(sample_id: UUID) -> SynchronizationGainResponse:
    try:
        return cached_synchronization_gain(database_url=DATABASE_URL, sample_id=sample_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/reviews/{sample_id}")
def save_synchronization_review(
    sample_id: UUID,
    request: SynchronizationReviewRequest,
) -> SynchronizationReviewSaveResponse:
    try:
        reviewed = SynchronizationReviewRepository(
            database_url=DATABASE_URL
        ).save_reviewed_alignment(
            sample_id=sample_id,
            speaker2_shift_seconds=request.speaker2_shift_seconds,
        )
        cached_synchronization_candidates.cache_clear()
        return SynchronizationReviewSaveResponse(
            sample_id=sample_id,
            external_id=reviewed.external_id,
            speaker2_shift_seconds=reviewed.speaker2_shift_seconds,
        )
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


@lru_cache(maxsize=128)
def cached_synchronization_gain(
    database_url: str,
    sample_id: UUID,
) -> SynchronizationGainResponse:
    if not database_url:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for synchronization review APIs.")
    dashboard_sample = Repository(database_url).get_dashboard_sample(sample_id=sample_id)
    if dashboard_sample.latest_quality is None:
        raise ValueError(f"Sample has no stored quality result: {sample_id}")
    quality_result = QualityResult.model_validate(dashboard_sample.latest_quality.payload)
    annotation = quality_result.conversation_annotation
    if annotation is None:
        raise ValueError(f"Sample has no stored conversation annotation: {sample_id}")
    return SynchronizationGainResponse(
        sample_id=sample_id,
        speaker1_gain=speech_only_gain_normalization(
            wave_path=track_path(
                dashboard_sample=dashboard_sample,
                side=TrackSide.SPEAKER1,
            ),
            speech_segments=annotation.speaker1.speech_segments,
        ),
        speaker2_gain=speech_only_gain_normalization(
            wave_path=track_path(
                dashboard_sample=dashboard_sample,
                side=TrackSide.SPEAKER2,
            ),
            speech_segments=annotation.speaker2.speech_segments,
        ),
    )


def track_path(dashboard_sample: DashboardSample, side: TrackSide) -> Path:
    track = next(
        (candidate for candidate in dashboard_sample.tracks if candidate.side is side),
        None,
    )
    if track is None:
        raise ValueError(f"Sample has no {side.value} track: {dashboard_sample.sample.id}")
    path = Path(track.access_uri).resolve()
    if not path.is_file():
        raise ValueError(f"Track audio does not exist: {path}")
    return path
