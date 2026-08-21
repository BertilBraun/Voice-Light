from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.shared.base_model import FrozenBaseModel

SPLIT_POLICY_VERSION = "conversation-dataset-sha256-v1"


class TrainingCorpusSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ConversationSplitCandidate(FrozenBaseModel):
    dataset_id: UUID
    sample_id: UUID


class ConversationSplitAssignment(FrozenBaseModel):
    dataset_id: UUID
    sample_id: UUID
    split: TrainingCorpusSplit


class ConversationSplitPlan(FrozenBaseModel):
    policy_version: Literal["conversation-dataset-sha256-v1"] = SPLIT_POLICY_VERSION
    seed: str = Field(min_length=1)
    assignments: tuple[ConversationSplitAssignment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_samples(self) -> ConversationSplitPlan:
        sample_ids = tuple(assignment.sample_id for assignment in self.assignments)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Conversation split assignments contain duplicate sample IDs.")
        return self


def assign_conversation_splits(
    candidates: Sequence[ConversationSplitCandidate],
    seed: str,
) -> ConversationSplitPlan:
    if not seed:
        raise ValueError("Conversation split seed must not be empty.")
    if not candidates:
        raise ValueError("Conversation split assignment requires at least one candidate.")
    _validate_unique_candidates(candidates)
    candidates_by_dataset: defaultdict[UUID, list[ConversationSplitCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_dataset[candidate.dataset_id].append(candidate)

    assignments: list[ConversationSplitAssignment] = []
    for dataset_id in sorted(candidates_by_dataset, key=lambda value: value.hex):
        ranked = sorted(
            candidates_by_dataset[dataset_id],
            key=lambda candidate: (
                _candidate_rank(seed=seed, candidate=candidate),
                candidate.sample_id.hex,
            ),
        )
        quotas = _split_quotas(len(ranked))
        offset = 0
        for split in TrainingCorpusSplit:
            split_count = quotas[split]
            assignments.extend(
                ConversationSplitAssignment(
                    dataset_id=candidate.dataset_id,
                    sample_id=candidate.sample_id,
                    split=split,
                )
                for candidate in ranked[offset : offset + split_count]
            )
            offset += split_count
        assert offset == len(ranked)

    return ConversationSplitPlan(
        policy_version=SPLIT_POLICY_VERSION,
        seed=seed,
        assignments=tuple(assignments),
    )


def _validate_unique_candidates(candidates: Sequence[ConversationSplitCandidate]) -> None:
    dataset_by_sample: dict[UUID, UUID] = {}
    for candidate in candidates:
        existing_dataset = dataset_by_sample.get(candidate.sample_id)
        if existing_dataset is not None:
            raise ValueError(
                f"Duplicate conversation split candidate for sample {candidate.sample_id}."
            )
        dataset_by_sample[candidate.sample_id] = candidate.dataset_id


def _candidate_rank(seed: str, candidate: ConversationSplitCandidate) -> bytes:
    payload = f"{SPLIT_POLICY_VERSION}:{seed}:{candidate.dataset_id}:{candidate.sample_id}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _split_quotas(conversation_count: int) -> dict[TrainingCorpusSplit, int]:
    assert conversation_count > 0
    ratios = {
        TrainingCorpusSplit.TRAIN: 0.8,
        TrainingCorpusSplit.VALIDATION: 0.1,
        TrainingCorpusSplit.TEST: 0.1,
    }
    exact = {split: conversation_count * ratio for split, ratio in ratios.items()}
    quotas = {split: math.floor(value) for split, value in exact.items()}
    unassigned_count = conversation_count - sum(quotas.values())
    remainder_order = sorted(
        TrainingCorpusSplit,
        key=lambda split: (
            -(exact[split] - quotas[split]),
            tuple(TrainingCorpusSplit).index(split),
        ),
    )
    for split in remainder_order[:unassigned_count]:
        quotas[split] += 1

    if conversation_count >= len(TrainingCorpusSplit):
        _ensure_represented(quotas, TrainingCorpusSplit.VALIDATION)
        _ensure_represented(quotas, TrainingCorpusSplit.TEST)
    assert sum(quotas.values()) == conversation_count
    return quotas


def _ensure_represented(
    quotas: dict[TrainingCorpusSplit, int],
    required_split: TrainingCorpusSplit,
) -> None:
    if quotas[required_split] > 0:
        return
    donor = max(
        TrainingCorpusSplit,
        key=lambda split: (quotas[split], -tuple(TrainingCorpusSplit).index(split)),
    )
    assert quotas[donor] > 1
    quotas[donor] -= 1
    quotas[required_split] += 1
