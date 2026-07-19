from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, HTTPException, Query

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
SAMPLE_PAGE_SIZE = 10
DEFAULT_CANDIDATE_PAGE_SIZE = 50
MAXIMUM_CANDIDATE_PAGE_SIZE = 200


class DashboardSampleRepository(Protocol):
    def list_dashboard_samples(
        self,
        sample_filter: SampleListFilter,
    ) -> list[DashboardSample]: ...


def repository() -> Repository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for backchannel review APIs.")
    return Repository(DATABASE_URL)


@router.get("/candidates")
def list_backchannel_review_candidates(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(
        default=DEFAULT_CANDIDATE_PAGE_SIZE,
        ge=1,
        le=MAXIMUM_CANDIDATE_PAGE_SIZE,
    ),
) -> BackchannelReviewListResponse:
    try:
        return _candidate_page(
            sample_repository=repository(),
            offset=offset,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _candidate_page(
    sample_repository: DashboardSampleRepository,
    offset: int,
    limit: int,
) -> BackchannelReviewListResponse:
    selected_candidates: list[BackchannelReviewCandidate] = []
    candidate_index = 0
    sample_offset = 0
    while True:
        page = sample_repository.list_dashboard_samples(
            SampleListFilter(limit=SAMPLE_PAGE_SIZE, offset=sample_offset)
        )
        for dashboard_sample in page:
            candidates = sorted(
                _candidates_for_sample(dashboard_sample),
                key=lambda candidate: (
                    candidate.external_id,
                    candidate.possible_backchannel.start_seconds,
                    candidate.floor_holder_side.value,
                ),
            )
            for candidate in candidates:
                if candidate_index >= offset:
                    selected_candidates.append(candidate)
                    if len(selected_candidates) > limit:
                        return BackchannelReviewListResponse(
                            candidates=tuple(selected_candidates[:limit]),
                            offset=offset,
                            next_offset=offset + limit,
                        )
                candidate_index += 1
        if len(page) < SAMPLE_PAGE_SIZE:
            return BackchannelReviewListResponse(
                candidates=tuple(selected_candidates),
                offset=offset,
                next_offset=None,
            )
        sample_offset += SAMPLE_PAGE_SIZE


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
