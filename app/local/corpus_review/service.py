from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from uuid import UUID

from app.local.conversation_regions.models import CONVERSATION_REGION_ANALYSIS_VERSION
from app.local.corpus_audit.repository import CorpusAuditEvidence
from app.local.corpus_review.models import (
    CorpusReviewGateIssue,
    CorpusReviewGateIssueCode,
    CorpusReviewItemRecord,
    CorpusReviewPlan,
    CorpusReviewReadiness,
    CorpusReviewSelectionAlgorithm,
    CorpusReviewSetRequest,
    CorpusReviewStatus,
    PlannedCorpusReviewItem,
)
from app.local.db.models import TrackSide
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.local.training_samples.service import (
    FRAME_SECONDS,
    INPUT_DURATION_SECONDS,
    TRAINING_LABEL_VERSION,
)
from app.shared.quality import METRIC_VERSION


def plan_corpus_review(
    evidence: tuple[CorpusAuditEvidence, ...],
    request: CorpusReviewSetRequest,
) -> tuple[PlannedCorpusReviewItem, ...]:
    evidence_by_dataset: dict[UUID, list[CorpusAuditEvidence]] = defaultdict(list)
    for item in evidence:
        evidence_by_dataset[item.dataset_id].append(item)
    planned: list[PlannedCorpusReviewItem] = []
    for selection in request.datasets:
        candidates = sorted(
            (
                item
                for item in evidence_by_dataset[selection.dataset_id]
                if item.conversation_regions is not None
                and item.represented_duration_seconds >= INPUT_DURATION_SECONDS
            ),
            key=lambda item: (item.external_id, str(item.sample_id)),
        )
        if len(candidates) < request.items_per_dataset:
            raise ValueError(
                f"Dataset {selection.dataset_id} has only {len(candidates)} eligible recordings; "
                f"{request.items_per_dataset} are required."
            )
        dataset_names = {item.dataset_name for item in candidates}
        if len(dataset_names) != 1:
            raise ValueError(
                f"Dataset {selection.dataset_id} has inconsistent names in corpus evidence."
            )
        dataset_name = next(iter(dataset_names))
        match request.selection_algorithm:
            case CorpusReviewSelectionAlgorithm.DATASET_ID_SHA256_V1:
                dataset_identifier = selection.dataset_id.hex
            case CorpusReviewSelectionAlgorithm.DATASET_NAME_SHA256_V2:
                dataset_identifier = dataset_name
        generator = random.Random(stable_dataset_seed(request.seed, dataset_identifier))
        selected = generator.sample(candidates, request.items_per_dataset)
        for item in selected:
            start_seconds = random_review_start(item=item, generator=generator)
            sides = (TrackSide.SPEAKER1, TrackSide.SPEAKER2)
            planned.append(
                PlannedCorpusReviewItem(
                    dataset_id=item.dataset_id,
                    dataset_name=item.dataset_name,
                    sample_id=item.sample_id,
                    external_id=item.external_id,
                    quality_score=item.quality_score,
                    user_side=sides[generator.randrange(len(sides))],
                    start_seconds=start_seconds,
                )
            )
    return tuple(planned)


def plan_failed_review_replacements(
    plan: CorpusReviewPlan,
    evidence: tuple[CorpusAuditEvidence, ...],
) -> tuple[tuple[UUID, PlannedCorpusReviewItem], ...]:
    used_recordings = {(item.dataset_id, item.sample_id) for item in plan.items}
    replacements: list[tuple[UUID, PlannedCorpusReviewItem]] = []
    for failed in (item for item in plan.items if item.overall_status is CorpusReviewStatus.FAIL):
        candidates = sorted(
            (
                item
                for item in evidence
                if item.dataset_id == failed.dataset_id
                and (item.dataset_id, item.sample_id) not in used_recordings
                and item.conversation_regions is not None
                and item.represented_duration_seconds >= INPUT_DURATION_SECONDS
            ),
            key=lambda item: (item.external_id, str(item.sample_id)),
        )
        if not candidates:
            raise ValueError(
                f"No unused replacement recording remains for dataset {failed.dataset_name}."
            )
        generator = random.Random(
            stable_dataset_seed(plan.review_set.seed, f"replacement:{failed.id.hex}")
        )
        selected = candidates[generator.randrange(len(candidates))]
        sides = (TrackSide.SPEAKER1, TrackSide.SPEAKER2)
        replacement = PlannedCorpusReviewItem(
            dataset_id=selected.dataset_id,
            dataset_name=selected.dataset_name,
            sample_id=selected.sample_id,
            external_id=selected.external_id,
            quality_score=selected.quality_score,
            user_side=sides[generator.randrange(len(sides))],
            start_seconds=random_review_start(item=selected, generator=generator),
        )
        replacements.append((failed.id, replacement))
        used_recordings.add((selected.dataset_id, selected.sample_id))
    return tuple(replacements)


def random_review_start(
    item: CorpusAuditEvidence,
    generator: random.Random,
) -> float:
    maximum_start_seconds = max(
        0.0,
        item.represented_duration_seconds - INPUT_DURATION_SECONDS,
    )
    frame_count = math.floor(maximum_start_seconds / FRAME_SECONDS)
    candidates = tuple(
        frame_index * FRAME_SECONDS
        for frame_index in range(frame_count + 1)
        if not any(
            frame_index * FRAME_SECONDS < exclusion.end_seconds
            and frame_index * FRAME_SECONDS + INPUT_DURATION_SECONDS > exclusion.start_seconds
            for exclusion in item.interval_exclusions
        )
    )
    if not candidates:
        raise ValueError(f"Recording {item.external_id} has no reviewable 20-second window.")
    return candidates[generator.randrange(len(candidates))]


def stable_dataset_seed(seed: str, dataset_identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_identifier}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def quantized_start_seconds(start_seconds: float, maximum_start_seconds: float) -> float:
    quantized = round(start_seconds / FRAME_SECONDS) * FRAME_SECONDS
    return min(maximum_start_seconds, max(0.0, quantized))


def corpus_review_readiness(plan: CorpusReviewPlan) -> CorpusReviewReadiness:
    expected_item_count = plan.review_set.items_per_dataset * len(plan.review_set.config.datasets)
    passed_item_count = sum(item.overall_status is CorpusReviewStatus.PASS for item in plan.items)
    failed_item_count = sum(item.overall_status is CorpusReviewStatus.FAIL for item in plan.items)
    pending_item_count = len(plan.items) - passed_item_count - failed_item_count
    issues: list[CorpusReviewGateIssue] = []
    if len(plan.items) != expected_item_count or any(
        sum(item.dataset_id == selection.dataset_id for item in plan.items)
        != plan.review_set.items_per_dataset
        for selection in plan.review_set.config.datasets
    ):
        issues.append(
            CorpusReviewGateIssue(
                code=CorpusReviewGateIssueCode.INCOMPLETE_SET,
                message=(
                    f"Expected {expected_item_count} review items with "
                    f"{plan.review_set.items_per_dataset} per dataset; found {len(plan.items)}."
                ),
            )
        )
    recording_keys = tuple((item.dataset_id, item.sample_id) for item in plan.items)
    if len(set(recording_keys)) != len(recording_keys):
        issues.append(
            CorpusReviewGateIssue(
                code=CorpusReviewGateIssueCode.DUPLICATE_RECORDING,
                message="A review set must contain distinct recordings within each dataset.",
            )
        )
    stale_item_count = sum(not review_item_provenance_current(item) for item in plan.items)
    if stale_item_count:
        issues.append(
            CorpusReviewGateIssue(
                code=CorpusReviewGateIssueCode.STALE_PROVENANCE,
                message=f"{stale_item_count} review items no longer match current corpus evidence.",
            )
        )
    if pending_item_count:
        issues.append(
            CorpusReviewGateIssue(
                code=CorpusReviewGateIssueCode.INCOMPLETE_REVIEW,
                message=f"{pending_item_count} review items still require a decision.",
            )
        )
    if failed_item_count:
        issues.append(
            CorpusReviewGateIssue(
                code=CorpusReviewGateIssueCode.FAILED_REVIEW,
                message=(
                    f"{failed_item_count} review items failed and require correction or exclusion."
                ),
            )
        )
    return CorpusReviewReadiness(
        review_set_name=plan.review_set.name,
        expected_item_count=expected_item_count,
        item_count=len(plan.items),
        passed_item_count=passed_item_count,
        failed_item_count=failed_item_count,
        pending_item_count=pending_item_count,
        ready_to_publish=not issues,
        issues=tuple(issues),
    )


def review_item_provenance_current(item: CorpusReviewItemRecord) -> bool:
    return (
        item.provenance_current
        and item.quality_metric_version == METRIC_VERSION
        and item.annotation_version == ANNOTATION_VERSION
        and item.region_analysis_version == CONVERSATION_REGION_ANALYSIS_VERSION
        and item.training_label_version == TRAINING_LABEL_VERSION
        and item.input_duration_seconds == INPUT_DURATION_SECONDS
        and item.frame_seconds == FRAME_SECONDS
    )
