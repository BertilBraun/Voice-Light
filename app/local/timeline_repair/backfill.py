from __future__ import annotations

from dataclasses import dataclass

from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.config import DATABASE_URL
from app.local.db.repository import Repository
from app.local.timeline_repair.generation import generate_derived_artifacts
from app.local.timeline_repair.models import (
    TIMELINE_REPAIR_PLAN_VERSION,
    TimelineRepairPlanCreate,
)
from app.local.timeline_repair.repository import TimelineRepairRepository


@dataclass(frozen=True)
class TimelineRepairBackfillSummary:
    eligible_count: int
    completed_count: int
    skipped_count: int
    represented_duration_seconds: float


def backfill_virtual_timeline_repairs(database_url: str) -> TimelineRepairBackfillSummary:
    repair_repository = TimelineRepairRepository(database_url)
    sample_repository = Repository(database_url)
    transcript_repository = FullRecordingAsrRepository(database_url)
    sources = repair_repository.list_eligible_sources()
    completed_count = 0
    skipped_count = 0
    for source in sources:
        plan = repair_repository.create_plan(
            TimelineRepairPlanCreate(plan_version=TIMELINE_REPAIR_PLAN_VERSION, source=source)
        )
        if plan.derived_annotation is not None and plan.conversation_regions is not None:
            skipped_count += 1
            continue
        dashboard_sample = sample_repository.get_dashboard_sample(source.sample_id)
        artifacts = generate_derived_artifacts(
            plan=plan,
            dashboard_sample=dashboard_sample,
            transcript_repository=transcript_repository,
        )
        repair_repository.save_derived_artifacts(plan_id=plan.id, artifacts=artifacts)
        completed_count += 1
        print(f"Generated repaired timeline for {source.external_id}", flush=True)
    return TimelineRepairBackfillSummary(
        eligible_count=len(sources),
        completed_count=completed_count,
        skipped_count=skipped_count,
        represented_duration_seconds=sum(source.duration_seconds for source in sources),
    )


def main() -> None:
    summary = backfill_virtual_timeline_repairs(DATABASE_URL)
    print(
        "Virtual timeline repair backfill: "
        f"{summary.completed_count} generated, {summary.skipped_count} already complete, "
        f"{summary.eligible_count} eligible, "
        f"{summary.represented_duration_seconds / 3600.0:.2f} hours."
    )


if __name__ == "__main__":
    main()
