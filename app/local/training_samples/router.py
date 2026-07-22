from __future__ import annotations

import random
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response

from app.local.config import DATABASE_URL
from app.local.conversation_regions.models import CONVERSATION_REGION_ANALYSIS_VERSION
from app.local.conversation_regions.repository import ConversationRegionRepository
from app.local.db.models import TrackSide
from app.local.db.repository import Repository
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.local.timeline_repair.models import (
    TIMELINE_REPAIR_PLAN_VERSION,
    TimelineRepairPlanRecord,
)
from app.local.timeline_repair.repository import TimelineRepairRepository
from app.local.training_samples.models import (
    TrainingSampleOption,
    TrainingSamplePreview,
    TrainingSampleProposition,
    TrainingSampleSelectionMode,
)
from app.local.training_samples.service import (
    build_training_sample_preview,
    build_training_sample_propositions,
)
from app.shared.quality import METRIC_VERSION

router = APIRouter(prefix="/api/training-samples", tags=["training-samples"])


def repository() -> Repository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for training sample APIs.")
    return Repository(DATABASE_URL)


def conversation_region_repository() -> ConversationRegionRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for training sample APIs.")
    return ConversationRegionRepository(DATABASE_URL)


def timeline_repair_repository() -> TimelineRepairRepository:
    if not DATABASE_URL:
        raise ValueError("VOICE_LIGHT_DATABASE_URL is required for training sample APIs.")
    return TimelineRepairRepository(DATABASE_URL)


def training_timeline_plan(sample_id: UUID) -> TimelineRepairPlanRecord | None:
    plan_repository = timeline_repair_repository()
    plan = plan_repository.get_complete_plan(
        sample_id=sample_id,
        plan_version=TIMELINE_REPAIR_PLAN_VERSION,
    )
    if plan is None and plan_repository.sample_requires_repair(sample_id):
        raise ValueError("The selected sample requires a completed virtual timeline repair.")
    return plan


@router.get("/preview")
def preview_training_sample(
    sample_id: UUID,
    response: Response,
    user_side: TrackSide = TrackSide.SPEAKER2,
    start_seconds: float | None = Query(default=None, ge=0.0),
) -> TrainingSamplePreview:
    try:
        response.headers["Cache-Control"] = "no-store"
        dashboard_sample = repository().get_dashboard_sample(sample_id)
        if dashboard_sample.sample.is_unusable:
            raise ValueError("The selected sample is marked unusable.")
        repair_plan = training_timeline_plan(sample_id)
        region_record = (
            conversation_region_repository().get_current(
                sample_id=sample_id,
                analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
                annotation_version=ANNOTATION_VERSION,
            )
            if repair_plan is None
            else None
        )
        return build_training_sample_preview(
            dashboard_sample=dashboard_sample,
            user_side=user_side,
            requested_start_seconds=start_seconds,
            selection_mode=TrainingSampleSelectionMode.RANDOM,
            generator=random.SystemRandom(),
            conversation_regions=(
                repair_plan.conversation_regions
                if repair_plan is not None
                else region_record.analysis
                if region_record is not None
                else None
            ),
            repair_plan=repair_plan,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/options")
def list_training_sample_options(
    response: Response,
    dataset_id: UUID,
    limit: int = Query(default=40, ge=1, le=100),
    minimum_quality: float | None = Query(default=None, ge=0.0, le=1.0),
) -> tuple[TrainingSampleOption, ...]:
    response.headers["Cache-Control"] = "no-store"
    records = repository().list_annotated_samples(
        dataset_id=dataset_id,
        metric_version=METRIC_VERSION,
        annotation_version=ANNOTATION_VERSION,
        limit=limit,
        minimum_quality=minimum_quality,
    )
    return tuple(
        TrainingSampleOption(
            dataset_id=dataset_id,
            sample_id=record.sample_id,
            external_id=record.external_id,
            represented_duration_seconds=record.represented_duration_seconds,
            usable_event_count=record.usable_event_count,
            quality_score=record.quality_score,
        )
        for record in records
    )


@router.get("/propositions")
def training_sample_propositions(
    sample_id: UUID,
    response: Response,
    user_side: TrackSide = TrackSide.SPEAKER2,
    limit: int = Query(default=15, ge=1, le=30),
) -> tuple[TrainingSampleProposition, ...]:
    try:
        response.headers["Cache-Control"] = "no-store"
        dashboard_sample = repository().get_dashboard_sample(sample_id)
        if dashboard_sample.sample.is_unusable:
            raise ValueError("The selected sample is marked unusable.")
        repair_plan = training_timeline_plan(sample_id)
        region_record = (
            conversation_region_repository().get_current(
                sample_id=sample_id,
                analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
                annotation_version=ANNOTATION_VERSION,
            )
            if repair_plan is None
            else None
        )
        return build_training_sample_propositions(
            dashboard_sample=dashboard_sample,
            user_side=user_side,
            conversation_regions=(
                repair_plan.conversation_regions
                if repair_plan is not None
                else region_record.analysis
                if region_record is not None
                else None
            ),
            limit=limit,
            repair_plan=repair_plan,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/random-preview")
def random_training_sample_preview(
    dataset_id: UUID,
    current_sample_id: UUID,
    response: Response,
    minimum_quality: float | None = Query(default=None, ge=0.0, le=1.0),
    sampling_mode: TrainingSampleSelectionMode = TrainingSampleSelectionMode.RANDOM,
) -> TrainingSamplePreview:
    try:
        response.headers["Cache-Control"] = "no-store"
        sample_repository = repository()
        match sampling_mode:
            case TrainingSampleSelectionMode.RANDOM:
                sample_id = sample_repository.random_annotated_sample_id(
                    dataset_id=dataset_id,
                    metric_version=METRIC_VERSION,
                    annotation_version=ANNOTATION_VERSION,
                    current_sample_id=current_sample_id,
                    minimum_quality=minimum_quality,
                )
            case TrainingSampleSelectionMode.INTERESTING:
                sample_id = sample_repository.interesting_annotated_sample_id(
                    dataset_id=dataset_id,
                    metric_version=METRIC_VERSION,
                    annotation_version=ANNOTATION_VERSION,
                    current_sample_id=current_sample_id,
                    minimum_quality=minimum_quality,
                )
        generator = random.SystemRandom()
        user_sides = (TrackSide.SPEAKER1, TrackSide.SPEAKER2)
        user_side = user_sides[generator.randrange(len(user_sides))]
        dashboard_sample = sample_repository.get_dashboard_sample(sample_id)
        repair_plan = training_timeline_plan(sample_id)
        region_record = (
            conversation_region_repository().get_current(
                sample_id=sample_id,
                analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
                annotation_version=ANNOTATION_VERSION,
            )
            if repair_plan is None
            else None
        )
        return build_training_sample_preview(
            dashboard_sample=dashboard_sample,
            user_side=user_side,
            requested_start_seconds=None,
            selection_mode=sampling_mode,
            generator=generator,
            conversation_regions=(
                repair_plan.conversation_regions
                if repair_plan is not None
                else region_record.analysis
                if region_record is not None
                else None
            ),
            repair_plan=repair_plan,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
