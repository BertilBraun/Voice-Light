from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.local.backchannel_review.models import (
    BackchannelReviewCandidate,
    BackchannelReviewListResponse,
)
from app.local.backchannel_review.service import find_ambiguous_backchannel_candidates
from app.local.config import DATABASE_URL
from app.local.db.models import DashboardSample, SampleListFilter
from app.local.db.repository import Repository
from app.shared.quality import METRIC_VERSION, QualityResult

router = APIRouter(prefix="/api/backchannel-review", tags=["backchannel-review"])
SAMPLE_PAGE_SIZE = 200


def repository() -> Repository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for backchannel review APIs.")
    return Repository(DATABASE_URL)


@router.get("/candidates")
def list_backchannel_review_candidates() -> BackchannelReviewListResponse:
    try:
        candidates = tuple(
            sorted(
                (
                    candidate
                    for dashboard_sample in _all_dashboard_samples(repository())
                    for candidate in _candidates_for_sample(dashboard_sample)
                ),
                key=lambda candidate: (
                    candidate.external_id,
                    candidate.possible_backchannel.start_seconds,
                    candidate.floor_holder_side.value,
                ),
            )
        )
        return BackchannelReviewListResponse(candidates=candidates)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _all_dashboard_samples(sample_repository: Repository) -> tuple[DashboardSample, ...]:
    samples: list[DashboardSample] = []
    offset = 0
    while True:
        page = sample_repository.list_dashboard_samples(
            SampleListFilter(limit=SAMPLE_PAGE_SIZE, offset=offset)
        )
        samples.extend(page)
        if len(page) < SAMPLE_PAGE_SIZE:
            return tuple(samples)
        offset += SAMPLE_PAGE_SIZE


def _candidates_for_sample(
    dashboard_sample: DashboardSample,
) -> tuple[BackchannelReviewCandidate, ...]:
    quality_record = dashboard_sample.latest_quality
    if quality_record is None or quality_record.metric_version != METRIC_VERSION:
        return ()
    quality_result = QualityResult.model_validate(quality_record.payload)
    annotation = quality_result.conversation_annotation
    if annotation is None:
        return ()
    return find_ambiguous_backchannel_candidates(
        sample_id=dashboard_sample.sample.id,
        external_id=dashboard_sample.sample.external_id,
        annotation=annotation,
    )
