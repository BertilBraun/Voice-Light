from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response

from app.local.config import DATABASE_URL
from app.local.db.models import DashboardSample, SampleListFilter
from app.local.db.repository import Repository
from app.local.misalignment_lab.models import (
    MisalignmentCandidatePreview,
    MisalignmentJudgment,
    MisalignmentJudgmentRequest,
    MisalignmentJudgmentResponse,
    MisalignmentQueueResponse,
    MisalignmentRepairJudgmentRequest,
    MisalignmentRepairJudgmentResponse,
    MisalignmentRepairQueueResponse,
)
from app.local.misalignment_lab.repository import MisalignmentLabRepository
from app.local.misalignment_lab.service import (
    DEFAULT_QUEUE_SEED,
    DEFAULT_QUEUE_SIZE,
    build_misalignment_preview,
    build_misalignment_queue,
    build_misalignment_repair_queue,
    eligible_annotated_session_count,
    misalignment_progress,
    misalignment_repair_progress,
    validate_judgment_candidate,
)
from app.local.synchronization_review.audit import SYNCHRONIZATION_AUDIT_PATH
from app.local.synchronization_review.models import SynchronizationAuditReport

router = APIRouter(prefix="/api/misalignment-lab", tags=["misalignment-lab"])
SAMPLE_PAGE_SIZE = 200


def sample_repository() -> Repository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for the misalignment lab.")
    return Repository(DATABASE_URL)


def judgment_repository() -> MisalignmentLabRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for the misalignment lab.")
    return MisalignmentLabRepository(DATABASE_URL)


@router.get("/queue")
def misalignment_queue(
    response: Response,
    seed: str = Query(default=DEFAULT_QUEUE_SEED, min_length=1, max_length=100),
    limit: int = Query(default=DEFAULT_QUEUE_SIZE, ge=1, le=100),
) -> MisalignmentQueueResponse:
    try:
        response.headers["Cache-Control"] = "no-store"
        judgments = judgment_repository().list_judgments()
        return build_misalignment_queue(
            dashboard_samples=_all_dashboard_samples(sample_repository()),
            audit_report=_audit_report(SYNCHRONIZATION_AUDIT_PATH),
            judgments=judgments,
            seed=seed,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/repair-queue")
def misalignment_repair_queue(response: Response) -> MisalignmentRepairQueueResponse:
    try:
        response.headers["Cache-Control"] = "no-store"
        reviews = judgment_repository()
        return build_misalignment_repair_queue(
            dashboard_samples=_all_dashboard_samples(sample_repository()),
            audit_report=_audit_report(SYNCHRONIZATION_AUDIT_PATH),
            judgments=reviews.list_judgments(),
            repair_judgments=reviews.list_repair_judgments(),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/preview/{sample_id}/{candidate_id}")
def misalignment_preview(
    sample_id: UUID,
    candidate_id: UUID,
    response: Response,
) -> MisalignmentCandidatePreview:
    try:
        response.headers["Cache-Control"] = "no-store"
        return build_misalignment_preview(
            dashboard_sample=sample_repository().get_dashboard_sample(sample_id),
            audit_report=_audit_report(SYNCHRONIZATION_AUDIT_PATH),
            candidate_id=candidate_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/judgments")
def save_misalignment_judgment(
    request: MisalignmentJudgmentRequest,
    response: Response,
) -> MisalignmentJudgmentResponse:
    try:
        response.headers["Cache-Control"] = "no-store"
        samples = _all_dashboard_samples(sample_repository())
        dashboard_sample = next(
            (candidate for candidate in samples if candidate.sample.id == request.sample_id),
            None,
        )
        if dashboard_sample is None:
            raise ValueError(f"Sample not found: {request.sample_id}")
        validate_judgment_candidate(
            dashboard_sample=dashboard_sample,
            audit_report=_audit_report(SYNCHRONIZATION_AUDIT_PATH),
            candidate_id=request.candidate_id,
            start_seconds=request.window_start_seconds,
            end_seconds=request.window_end_seconds,
        )
        reviews = judgment_repository()
        stored = reviews.save_judgment(request=request)
        judgments = reviews.list_judgments()
        return MisalignmentJudgmentResponse(
            stored=stored,
            session_quarantined=stored.judgment is MisalignmentJudgment.LIKELY_MISALIGNED,
            progress=misalignment_progress(
                judgments=judgments,
                eligible_session_count=eligible_annotated_session_count(samples),
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/repair-judgments")
def save_misalignment_repair_judgment(
    request: MisalignmentRepairJudgmentRequest,
    response: Response,
) -> MisalignmentRepairJudgmentResponse:
    try:
        response.headers["Cache-Control"] = "no-store"
        samples = _all_dashboard_samples(sample_repository())
        reviews = judgment_repository()
        repair_queue = build_misalignment_repair_queue(
            dashboard_samples=samples,
            audit_report=_audit_report(SYNCHRONIZATION_AUDIT_PATH),
            judgments=reviews.list_judgments(),
            repair_judgments=reviews.list_repair_judgments(),
        )
        repair_candidate = next(
            (
                candidate
                for candidate in repair_queue.candidates
                if candidate.candidate.sample_id == request.sample_id
            ),
            None,
        )
        if repair_candidate is None:
            raise ValueError("No conservative repair estimate exists for this session.")
        if repair_candidate.candidate.candidate_id != request.candidate_id:
            raise ValueError("Repair candidate does not match the quarantined review.")
        estimate = repair_candidate.repair_estimate
        if (
            estimate.estimator_version != request.estimator_version
            or abs(estimate.predicted_second_part_shift_seconds - request.predicted_shift_seconds)
            > 1e-6
        ):
            raise ValueError("Repair estimate is stale; reload the repair queue.")
        stored = reviews.save_repair_judgment(request=request)
        repair_judgments = reviews.list_repair_judgments()
        return MisalignmentRepairJudgmentResponse(
            stored=stored,
            progress=misalignment_repair_progress(
                quarantined_session_count=(repair_queue.progress.quarantined_session_count),
                repair_candidate_count=repair_queue.progress.repair_candidate_count,
                repair_judgments=repair_judgments,
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _all_dashboard_samples(repository: Repository) -> tuple[DashboardSample, ...]:
    samples: list[DashboardSample] = []
    offset = 0
    while True:
        page = repository.list_dashboard_samples(
            SampleListFilter(limit=SAMPLE_PAGE_SIZE, offset=offset)
        )
        samples.extend(page)
        if len(page) < SAMPLE_PAGE_SIZE:
            return tuple(samples)
        offset += SAMPLE_PAGE_SIZE


def _audit_report(path: Path) -> SynchronizationAuditReport | None:
    if not path.is_file():
        return None
    return SynchronizationAuditReport.model_validate_json(path.read_text(encoding="utf-8"))
