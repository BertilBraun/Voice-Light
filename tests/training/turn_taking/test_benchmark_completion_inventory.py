from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.local.db.models import TrackSide
from app.local.training_corpus.export import FRAMES_PER_SAMPLE, MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_completion_inventory import (
    build_turn_completion_inventory,
)
from app.training.turn_taking.benchmark_models import (
    CompletionBoundaryKind,
    read_inventory,
    read_turn_completion_inventory,
    write_turn_completion_inventory,
)

CORPUS_REVISION = "1" * 40
DATASET_ID = UUID("00000000-0000-0000-0000-000000000001")
SAMPLE_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_inventory_includes_terminal_and_continuation_boundaries(tmp_path: Path) -> None:
    floor = [0.0] * FRAMES_PER_SAMPLE
    floor[0:3] = [1.0, 1.0, 1.0]
    floor[8:10] = [1.0, 1.0]
    floor[12:15] = [1.0, 1.0, 1.0]
    completion = [-1.0] * FRAMES_PER_SAMPLE
    completion[3] = 0.9
    completion[10] = 0.1
    continuation = [-1.0] * FRAMES_PER_SAMPLE
    continuation[10] = 0.9
    first = _sample(
        "a" * 64,
        floor=floor,
        completion=completion,
        continuation=continuation,
        category="first",
    )
    second = _sample(
        "b" * 64,
        floor=floor,
        completion=completion,
        continuation=continuation,
        category="second",
    )

    inventory = build_turn_completion_inventory(
        samples=(second, first),
        corpus_repository="owner/corpus",
        corpus_revision=CORPUS_REVISION,
        split=TrainingCorpusSplit.VALIDATION,
        causal_horizon_seconds=0.32,
    )

    assert inventory.manifest.schema_version == (
        "voice-light-turn-completion-candidate-inventory-v2"
    )
    assert inventory.manifest.candidate_count == 2
    terminal, continuation = inventory.candidates
    assert terminal.boundary_kind is CompletionBoundaryKind.TERMINAL
    assert terminal.anchor_seconds == pytest.approx(0.24)
    assert terminal.end_seconds == pytest.approx(0.56)
    assert terminal.continuation_probability is None
    assert [point.completion_probability for point in terminal.target_points] == pytest.approx(
        [0.9, 0.9, 0.9, 0.9]
    )
    assert continuation.boundary_kind is CompletionBoundaryKind.CONTINUATION
    assert continuation.anchor_seconds == pytest.approx(0.8)
    assert continuation.end_seconds == pytest.approx(0.96)
    assert continuation.continuation_probability == pytest.approx(0.9)
    assert [point.elapsed_seconds for point in continuation.target_points] == pytest.approx(
        [0.08, 0.16]
    )
    assert continuation.categories == ("first", "second")
    assert continuation.source_window_ids == ("a" * 64, "b" * 64)

    path = tmp_path / "completion-inventory.json"
    write_turn_completion_inventory(path, inventory)
    assert read_turn_completion_inventory(path) == inventory
    with pytest.raises(ValidationError):
        read_inventory(path)


def test_inventory_masks_assistant_active_and_truncated_terminal_boundaries() -> None:
    floor = [0.0] * FRAMES_PER_SAMPLE
    floor[0:2] = [1.0, 1.0]
    floor[20:22] = [1.0, 1.0]
    floor[247:249] = [1.0, 1.0]
    assistant_floor = [0.0] * FRAMES_PER_SAMPLE
    assistant_floor[2] = 1.0
    completion = [-1.0] * FRAMES_PER_SAMPLE
    completion[2] = 0.9
    completion[22] = 0.2
    completion[249] = 0.8

    inventory = build_turn_completion_inventory(
        samples=(
            _sample(
                "a" * 64,
                floor=floor,
                completion=completion,
                assistant_floor=assistant_floor,
            ),
        ),
        corpus_repository="owner/corpus",
        corpus_revision=CORPUS_REVISION,
        split=TrainingCorpusSplit.VALIDATION,
    )

    assert inventory.manifest.causal_horizon_seconds == pytest.approx(2.0)
    assert [candidate.anchor_seconds for candidate in inventory.candidates] == pytest.approx([1.76])


def test_inventory_rejects_conflicting_sparse_completion_targets() -> None:
    floor = [0.0] * FRAMES_PER_SAMPLE
    floor[0:2] = [1.0, 1.0]
    first_completion = [-1.0] * FRAMES_PER_SAMPLE
    first_completion[2] = 0.2
    second_completion = first_completion.copy()
    second_completion[2] = 0.8

    with pytest.raises(ValueError, match="Conflicting labels"):
        build_turn_completion_inventory(
            samples=(
                _sample("a" * 64, floor=floor, completion=first_completion),
                _sample("b" * 64, floor=floor, completion=second_completion),
            ),
            corpus_repository="owner/corpus",
            corpus_revision=CORPUS_REVISION,
            split=TrainingCorpusSplit.VALIDATION,
        )


def test_inventory_rejects_conflicting_continuation_evidence() -> None:
    floor = [0.0] * FRAMES_PER_SAMPLE
    floor[0:2] = [1.0, 1.0]
    completion = [-1.0] * FRAMES_PER_SAMPLE
    completion[2] = 0.2
    first_continuation = [-1.0] * FRAMES_PER_SAMPLE
    first_continuation[2] = 0.9
    second_continuation = first_continuation.copy()
    second_continuation[2] = 0.1

    with pytest.raises(ValueError, match="Conflicting labels"):
        build_turn_completion_inventory(
            samples=(
                _sample(
                    "a" * 64,
                    floor=floor,
                    completion=completion,
                    continuation=first_continuation,
                ),
                _sample(
                    "b" * 64,
                    floor=floor,
                    completion=completion,
                    continuation=second_continuation,
                ),
            ),
            corpus_repository="owner/corpus",
            corpus_revision=CORPUS_REVISION,
            split=TrainingCorpusSplit.VALIDATION,
        )


def test_inventory_uses_explicit_synthetic_continuation_interval() -> None:
    floor = [0.0] * FRAMES_PER_SAMPLE
    floor[0:3] = [1.0] * 3
    completion = [-1.0] * FRAMES_PER_SAMPLE
    completion[3] = 0.0
    continuation = [-1.0] * FRAMES_PER_SAMPLE
    continuation[3:8] = [1.0] * 5

    inventory = build_turn_completion_inventory(
        samples=(
            _sample(
                "a" * 64,
                floor=floor,
                completion=completion,
                continuation=continuation,
                assistant_floor=[1.0] * FRAMES_PER_SAMPLE,
            ),
        ),
        corpus_repository="owner/corpus",
        corpus_revision=CORPUS_REVISION,
        split=TrainingCorpusSplit.VALIDATION,
        continuation_interval_targets=True,
    )

    assert inventory.manifest.candidate_count == 1
    candidate = inventory.candidates[0]
    assert candidate.boundary_kind is CompletionBoundaryKind.CONTINUATION
    assert candidate.anchor_seconds == pytest.approx(0.24)
    assert candidate.end_seconds == pytest.approx(0.64)
    assert len(candidate.target_points) == 5


def test_explicit_synthetic_inventory_retains_terminal_anchor_during_floor_overlap() -> None:
    floor = [0.0] * FRAMES_PER_SAMPLE
    floor[0:4] = [1.0] * 4
    completion = [-1.0] * FRAMES_PER_SAMPLE
    completion[3] = 1.0

    inventory = build_turn_completion_inventory(
        samples=(
            _sample(
                "a" * 64,
                floor=floor,
                completion=completion,
                assistant_floor=[1.0] * FRAMES_PER_SAMPLE,
            ),
        ),
        corpus_repository="owner/corpus",
        corpus_revision=CORPUS_REVISION,
        split=TrainingCorpusSplit.VALIDATION,
        causal_horizon_seconds=0.32,
        continuation_interval_targets=True,
    )

    assert inventory.manifest.candidate_count == 1
    assert inventory.candidates[0].boundary_kind is CompletionBoundaryKind.TERMINAL
    assert len(inventory.candidates[0].target_points) == 4


def _sample(
    window_id: str,
    floor: list[float],
    completion: list[float],
    continuation: list[float] | None = None,
    category: str = "boundary",
    assistant_floor: list[float] | None = None,
) -> MaterializedTrainingSample:
    zeros = tuple(0.0 for _ in range(FRAMES_PER_SAMPLE))
    return MaterializedTrainingSample(
        schema_version="voice-light-turn-taking-v1",
        training_label_version="turn-taking-frame-labels-v1",
        window_id=window_id,
        dataset_id=DATASET_ID,
        dataset_name="Mundo TurnBench",
        sample_id=SAMPLE_ID,
        external_id="conversation-1",
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        split=TrainingCorpusSplit.VALIDATION,
        user_audio_path="dataset/sample/speaker_1.flac",
        assistant_audio_path="dataset/sample/speaker_2.flac",
        start_seconds=0.0,
        end_seconds=20.0,
        quality_score=0.9,
        category=category,
        assistant_has_floor=tuple(assistant_floor) if assistant_floor is not None else zeros,
        p_user_has_floor=tuple(floor),
        p_user_yield=zeros,
        p_assistant_backchannel=zeros,
        future_activity_0_200=zeros,
        future_activity_200_500=zeros,
        future_activity_500_1000=zeros,
        future_activity_1000_1500=zeros,
        turn_completion=tuple(completion),
        continuation_pause=(
            tuple(continuation)
            if continuation is not None
            else tuple(-1.0 for _ in range(FRAMES_PER_SAMPLE))
        ),
        non_floor_feedback=zeros,
        floor_take=zeros,
    )
