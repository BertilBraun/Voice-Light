from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from uuid import UUID

from app.local.corpus_audit.repository import CorpusAuditEvidence
from app.local.corpus_review.models import (
    CorpusReviewSetRequest,
    PlannedCorpusReviewItem,
)
from app.local.db.models import TrackSide
from app.local.training_samples.service import FRAME_SECONDS, INPUT_DURATION_SECONDS


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
        generator = random.Random(stable_dataset_seed(request.seed, selection.dataset_id.hex))
        selected = generator.sample(candidates, request.items_per_dataset)
        for item in selected:
            maximum_start_seconds = max(
                0.0,
                item.represented_duration_seconds - INPUT_DURATION_SECONDS,
            )
            start_seconds = quantized_start_seconds(
                generator.uniform(0.0, maximum_start_seconds),
                maximum_start_seconds,
            )
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


def stable_dataset_seed(seed: str, dataset_identifier: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_identifier}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def quantized_start_seconds(start_seconds: float, maximum_start_seconds: float) -> float:
    quantized = round(start_seconds / FRAME_SECONDS) * FRAME_SECONDS
    return min(maximum_start_seconds, max(0.0, quantized))
