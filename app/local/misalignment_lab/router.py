from __future__ import annotations

from functools import lru_cache
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
    MisalignmentRepairJudgment,
    MisalignmentRepairJudgmentRequest,
    MisalignmentRepairJudgmentResponse,
    MisalignmentRepairQueueResponse,
    MisalignmentRepairScope,
    MisalignmentStoredJudgment,
    MisalignmentTransitionPreview,
)
from app.local.misalignment_lab.repository import MisalignmentLabRepository
from app.local.misalignment_lab.service import (
    DEFAULT_QUEUE_SEED,
    DEFAULT_QUEUE_SIZE,
    build_misalignment_preview,
    build_misalignment_queue,
    build_misalignment_repair_queue,
    build_transition_preview,
    estimate_piecewise_repair,
    misalignment_progress,
    validate_judgment_candidate,
)
from app.local.synchronization_review.audit import SYNCHRONIZATION_AUDIT_PATH
from app.local.synchronization_review.models import (
    SynchronizationAuditReport,
    SynchronizationAuditResult,
)

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
        audit_report = _audit_report(SYNCHRONIZATION_AUDIT_PATH)
        return build_misalignment_queue(
            dashboard_samples=_audited_dashboard_samples(
                repository=sample_repository(),
                audit_report=audit_report,
            ),
            audit_report=audit_report,
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
        judgments = reviews.list_judgments()
        return build_misalignment_repair_queue(
            dashboard_samples=_quarantined_dashboard_samples(
                repository=sample_repository(),
                judgments=judgments,
            ),
            audit_report=_audit_report(SYNCHRONIZATION_AUDIT_PATH),
            judgments=judgments,
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


@router.get("/transition-preview/{sample_id}/{candidate_id}")
def misalignment_transition_preview(
    sample_id: UUID,
    candidate_id: UUID,
    response: Response,
    center_seconds: float | None = Query(default=None, ge=0.0),
) -> MisalignmentTransitionPreview:
    try:
        response.headers["Cache-Control"] = "no-store"
        return build_transition_preview(
            dashboard_sample=sample_repository().get_dashboard_sample(sample_id),
            audit_report=_audit_report(SYNCHRONIZATION_AUDIT_PATH),
            candidate_id=candidate_id,
            center_seconds=center_seconds,
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
        repository = sample_repository()
        audit_report = _audit_report(SYNCHRONIZATION_AUDIT_PATH)
        dashboard_sample = repository.get_dashboard_sample(request.sample_id)
        validate_judgment_candidate(
            dashboard_sample=dashboard_sample,
            audit_report=audit_report,
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
                eligible_session_count=len(
                    _audited_dashboard_samples(
                        repository=repository,
                        audit_report=audit_report,
                    )
                ),
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
        reviews = judgment_repository()
        judgments = reviews.list_judgments()
        quarantine_judgment = next(
            (
                judgment
                for judgment in judgments
                if judgment.sample_id == request.sample_id
                and judgment.candidate_id == request.candidate_id
                and judgment.judgment is MisalignmentJudgment.LIKELY_MISALIGNED
            ),
            None,
        )
        if quarantine_judgment is None:
            raise ValueError("The offset recommendation must be attached to a quarantined review.")
        repository = sample_repository()
        audit_report = _audit_report(SYNCHRONIZATION_AUDIT_PATH)
        candidate = validate_judgment_candidate(
            dashboard_sample=repository.get_dashboard_sample(request.sample_id),
            audit_report=audit_report,
            candidate_id=request.candidate_id,
            start_seconds=quarantine_judgment.window_start_seconds,
            end_seconds=quarantine_judgment.window_end_seconds,
        )
        recommendation = candidate.offset_recommendation
        if recommendation is None:
            raise ValueError("No safe single-offset recommendation exists for this session.")
        if (
            recommendation.estimator_version != request.estimator_version
            or abs(recommendation.shift_seconds - request.predicted_shift_seconds) > 1e-6
        ):
            raise ValueError("Offset recommendation is stale; reload the review queue.")
        if recommendation.repair_scope is not request.repair_scope:
            raise ValueError("Repair scope is stale; reload the review queue.")
        audit_result = (
            next(
                (
                    result
                    for result in audit_report.results
                    if result.sample_id == request.sample_id
                ),
                None,
            )
            if audit_report is not None
            else None
        )
        _validate_repair_transition(
            request=request,
            audit_result=audit_result,
        )
        stored = reviews.save_repair_judgment(request=request)
        repair_judgments = reviews.list_repair_judgments()
        samples = _quarantined_dashboard_samples(
            repository=repository,
            judgments=judgments,
        )
        repair_queue = build_misalignment_repair_queue(
            dashboard_samples=samples,
            audit_report=audit_report,
            judgments=judgments,
            repair_judgments=repair_judgments,
        )
        return MisalignmentRepairJudgmentResponse(
            stored=stored,
            progress=repair_queue.progress,
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


def _audited_dashboard_samples(
    repository: Repository,
    audit_report: SynchronizationAuditReport | None,
) -> tuple[DashboardSample, ...]:
    if audit_report is None or not audit_report.results:
        return _all_dashboard_samples(repository)
    return _cached_audited_dashboard_samples(
        database_url=repository.database_url,
        audited_sample_ids=tuple(result.sample_id for result in audit_report.results),
    )


@lru_cache(maxsize=2)
def _cached_audited_dashboard_samples(
    database_url: str,
    audited_sample_ids: tuple[UUID, ...],
) -> tuple[DashboardSample, ...]:
    repository = Repository(database_url)
    audited_sample_id_set = set(audited_sample_ids)
    first_sample = repository.get_dashboard_sample(audited_sample_ids[0])
    samples: list[DashboardSample] = []
    offset = 0
    while True:
        page = repository.list_dashboard_samples(
            SampleListFilter(
                dataset_id=first_sample.sample.dataset_id,
                limit=SAMPLE_PAGE_SIZE,
                offset=offset,
            )
        )
        samples.extend(sample for sample in page if sample.sample.id in audited_sample_id_set)
        if len(page) < SAMPLE_PAGE_SIZE:
            return tuple(samples)
        offset += SAMPLE_PAGE_SIZE


def _quarantined_dashboard_samples(
    repository: Repository,
    judgments: tuple[MisalignmentStoredJudgment, ...],
) -> tuple[DashboardSample, ...]:
    sample_ids = sorted(
        {
            judgment.sample_id
            for judgment in judgments
            if judgment.judgment is MisalignmentJudgment.LIKELY_MISALIGNED
        },
        key=str,
    )
    return tuple(repository.get_dashboard_sample(sample_id) for sample_id in sample_ids)


def _audit_report(path: Path) -> SynchronizationAuditReport | None:
    if not path.is_file():
        return None
    return SynchronizationAuditReport.model_validate_json(path.read_text(encoding="utf-8"))


def _validate_repair_transition(
    request: MisalignmentRepairJudgmentRequest,
    audit_result: SynchronizationAuditResult | None,
) -> None:
    if request.repair_scope is MisalignmentRepairScope.GLOBAL_OFFSET:
        if any(
            value is not None
            for value in (
                request.first_part_shift_seconds,
                request.change_point_seconds,
                request.change_interval_start_seconds,
                request.change_interval_end_seconds,
            )
        ):
            raise ValueError("A global offset cannot contain a transition location.")
        if request.transition_confirmed is not (
            request.judgment is MisalignmentRepairJudgment.PLAUSIBLE
        ):
            raise ValueError("Global transition confirmation does not match the repair judgment.")
        return
    if audit_result is None:
        raise ValueError("No synchronization audit exists for the piecewise repair.")
    estimate = estimate_piecewise_repair(audit_result=audit_result)
    if estimate is None:
        raise ValueError("No conservative piecewise transition exists for this session.")
    if (
        request.first_part_shift_seconds != estimate.first_part_shift_seconds
        or request.change_interval_start_seconds != estimate.conservative_first_part_end_seconds
        or request.change_interval_end_seconds != estimate.conservative_second_part_start_seconds
    ):
        raise ValueError("Transition estimate is stale; reload the review queue.")
    if request.judgment is MisalignmentRepairJudgment.PLAUSIBLE:
        if not request.transition_confirmed or request.change_point_seconds is None:
            raise ValueError("A plausible piecewise repair requires a confirmed transition point.")
        if not (
            estimate.conservative_first_part_end_seconds
            <= request.change_point_seconds
            <= estimate.conservative_second_part_start_seconds
        ):
            raise ValueError("Confirmed transition point is outside the conservative interval.")
        return
    if request.transition_confirmed or request.change_point_seconds is not None:
        raise ValueError("An unaccepted repair cannot contain a confirmed transition point.")
