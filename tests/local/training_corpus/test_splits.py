from __future__ import annotations

from collections import Counter
from uuid import UUID

import pytest

from app.local.training_corpus.splits import (
    SPLIT_POLICY_VERSION,
    ConversationSplitCandidate,
    TrainingCorpusSplit,
    assign_conversation_splits,
)

DATASET_1 = UUID("00000000-0000-0000-0000-000000000101")
DATASET_2 = UUID("00000000-0000-0000-0000-000000000102")


def test_split_assignment_is_independent_of_candidate_order() -> None:
    candidates = candidates_for_dataset(DATASET_1, 33)

    first = assign_conversation_splits(candidates, seed="pilot-v1")
    second = assign_conversation_splits(tuple(reversed(candidates)), seed="pilot-v1")

    assert first == second
    assert first.policy_version == SPLIT_POLICY_VERSION


@pytest.mark.parametrize(
    ("conversation_count", "expected_counts"),
    (
        (1, (1, 0, 0)),
        (2, (2, 0, 0)),
        (3, (1, 1, 1)),
        (4, (2, 1, 1)),
        (8, (6, 1, 1)),
        (10, (8, 1, 1)),
        (29, (23, 3, 3)),
        (33, (27, 3, 3)),
        (37, (29, 4, 4)),
    ),
)
def test_split_assignment_uses_per_dataset_80_10_10_quotas(
    conversation_count: int,
    expected_counts: tuple[int, int, int],
) -> None:
    plan = assign_conversation_splits(
        candidates_for_dataset(DATASET_1, conversation_count),
        seed="pilot-v1",
    )

    counts = Counter(assignment.split for assignment in plan.assignments)

    assert tuple(counts[split] for split in TrainingCorpusSplit) == expected_counts


def test_split_assignment_stratifies_each_dataset_independently() -> None:
    first_dataset = candidates_for_dataset(DATASET_1, 8, sample_offset=0)
    second_dataset = candidates_for_dataset(DATASET_2, 3, sample_offset=100)

    plan = assign_conversation_splits(
        (*first_dataset, *second_dataset),
        seed="pilot-v1",
    )

    first_counts = Counter(
        assignment.split for assignment in plan.assignments if assignment.dataset_id == DATASET_1
    )
    second_counts = Counter(
        assignment.split for assignment in plan.assignments if assignment.dataset_id == DATASET_2
    )
    assert tuple(first_counts[split] for split in TrainingCorpusSplit) == (6, 1, 1)
    assert tuple(second_counts[split] for split in TrainingCorpusSplit) == (1, 1, 1)
    assert {assignment.sample_id for assignment in plan.assignments} == {
        candidate.sample_id for candidate in (*first_dataset, *second_dataset)
    }


def test_split_assignment_changes_with_seed() -> None:
    candidates = candidates_for_dataset(DATASET_1, 20)

    first = assign_conversation_splits(candidates, seed="pilot-v1")
    second = assign_conversation_splits(candidates, seed="pilot-v2")

    first_split_by_sample = {
        assignment.sample_id: assignment.split for assignment in first.assignments
    }
    second_split_by_sample = {
        assignment.sample_id: assignment.split for assignment in second.assignments
    }
    assert first_split_by_sample != second_split_by_sample


def test_split_assignment_rejects_duplicate_samples() -> None:
    sample_id = UUID(int=1)
    candidates = (
        ConversationSplitCandidate(dataset_id=DATASET_1, sample_id=sample_id),
        ConversationSplitCandidate(dataset_id=DATASET_2, sample_id=sample_id),
    )

    with pytest.raises(ValueError, match="Duplicate conversation split candidate"):
        assign_conversation_splits(candidates, seed="pilot-v1")


def test_split_assignment_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        assign_conversation_splits((), seed="pilot-v1")


def test_split_assignment_rejects_empty_seed() -> None:
    with pytest.raises(ValueError, match="seed must not be empty"):
        assign_conversation_splits(candidates_for_dataset(DATASET_1, 1), seed="")


def candidates_for_dataset(
    dataset_id: UUID,
    count: int,
    sample_offset: int = 0,
) -> tuple[ConversationSplitCandidate, ...]:
    return tuple(
        ConversationSplitCandidate(
            dataset_id=dataset_id,
            sample_id=UUID(int=sample_offset + index + 1),
        )
        for index in range(count)
    )
