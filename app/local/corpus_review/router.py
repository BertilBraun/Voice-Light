from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Response

from app.local.config import DATABASE_URL
from app.local.conversation_regions.models import CONVERSATION_REGION_ANALYSIS_VERSION
from app.local.corpus_audit.repository import CorpusAuditRepository
from app.local.corpus_review.exclusions import CorpusExclusionRecord, CorpusExclusionRequest
from app.local.corpus_review.models import (
    CorpusReviewDecision,
    CorpusReviewItemRecord,
    CorpusReviewPlan,
    CorpusReviewReadiness,
    CorpusReviewSetRequest,
)
from app.local.corpus_review.repository import CorpusReviewRepository
from app.local.corpus_review.service import (
    corpus_review_readiness,
    plan_corpus_review,
    plan_failed_review_replacements,
)
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.shared.quality import METRIC_VERSION

router = APIRouter(prefix="/api/corpus-review", tags=["corpus-review"])


def review_repository() -> CorpusReviewRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for corpus review APIs.")
    return CorpusReviewRepository(DATABASE_URL)


def audit_repository() -> CorpusAuditRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for corpus review APIs.")
    return CorpusAuditRepository(DATABASE_URL)


@router.post("/sets")
def create_review_set(request: CorpusReviewSetRequest, response: Response) -> CorpusReviewPlan:
    try:
        response.headers["Cache-Control"] = "no-store"
        dataset_ids = tuple(selection.dataset_id for selection in request.datasets)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("A corpus review dataset may only be selected once.")
        evidence = tuple(
            item
            for selection in request.datasets
            for item in audit_repository().load_evidence(
                dataset_ids=(selection.dataset_id,),
                minimum_quality=selection.minimum_quality,
                metric_version=METRIC_VERSION,
                annotation_version=ANNOTATION_VERSION,
                region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
            )
        )
        planned_items = plan_corpus_review(evidence=evidence, request=request)
        return review_repository().create_or_get(request=request, planned_items=planned_items)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/sets/{name}")
def get_review_set(name: str, response: Response) -> CorpusReviewPlan:
    try:
        response.headers["Cache-Control"] = "no-store"
        return review_repository().get(name)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/sets/{name}/readiness")
def get_review_readiness(name: str, response: Response) -> CorpusReviewReadiness:
    try:
        response.headers["Cache-Control"] = "no-store"
        return corpus_review_readiness(review_repository().get(name))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.put("/items/{item_id}")
def update_review_item(
    item_id: UUID,
    decision: CorpusReviewDecision,
    response: Response,
) -> CorpusReviewItemRecord:
    try:
        response.headers["Cache-Control"] = "no-store"
        return review_repository().update_decision(item_id=item_id, decision=decision)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/items/{item_id}/exclusions")
def apply_review_exclusion(
    item_id: UUID,
    request: CorpusExclusionRequest,
    response: Response,
) -> CorpusExclusionRecord:
    try:
        response.headers["Cache-Control"] = "no-store"
        return review_repository().apply_exclusion(item_id=item_id, request=request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/sets/{name}/replace-failures")
def replace_failed_review_items(name: str, response: Response) -> CorpusReviewPlan:
    try:
        response.headers["Cache-Control"] = "no-store"
        repository = review_repository()
        plan = repository.get(name)
        evidence = tuple(
            item
            for selection in plan.review_set.config.datasets
            for item in audit_repository().load_evidence(
                dataset_ids=(selection.dataset_id,),
                minimum_quality=selection.minimum_quality,
                metric_version=METRIC_VERSION,
                annotation_version=ANNOTATION_VERSION,
                region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
            )
        )
        replacements = plan_failed_review_replacements(plan=plan, evidence=evidence)
        if not replacements:
            raise ValueError("The review set has no failed items to replace.")
        return repository.replace_failed_items(name=name, replacements=replacements)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
