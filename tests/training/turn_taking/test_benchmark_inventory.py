from pathlib import Path
from uuid import UUID

import pytest

from app.local.db.models import TrackSide
from app.local.training_corpus.export import FRAMES_PER_SAMPLE, MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.benchmark_inventory import build_candidate_inventory
from app.training.turn_taking.benchmark_models import read_inventory, write_inventory

CORPUS_REVISION = "1" * 40
DATASET_ID = UUID("00000000-0000-0000-0000-000000000001")
SAMPLE_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_inventory_deduplicates_windows_and_requires_bounded_silence(tmp_path: Path) -> None:
    floor = [1.0] * FRAMES_PER_SAMPLE
    user_yield = [0.0] * FRAMES_PER_SAMPLE
    floor[2:5] = [0.0, 0.0, 0.0]
    user_yield[2:5] = [0.2, 0.7, 0.9]
    first = _sample("a" * 64, 0.0, floor, user_yield, "hold_pause")
    second = _sample("b" * 64, 0.0, floor, user_yield, "turn_shift")

    inventory = build_candidate_inventory(
        samples=(second, first),
        corpus_repository="owner/corpus",
        corpus_revision=CORPUS_REVISION,
        split=TrainingCorpusSplit.VALIDATION,
    )

    assert inventory.manifest.sample_window_count == 2
    assert inventory.manifest.conversation_count == 1
    assert inventory.manifest.candidate_count == 1
    candidate = inventory.candidates[0]
    assert candidate.start_seconds == pytest.approx(0.16)
    assert candidate.end_seconds == pytest.approx(0.4)
    assert candidate.categories == ("hold_pause", "turn_shift")
    assert candidate.source_window_ids == ("a" * 64, "b" * 64)
    assert [point.silence_duration_seconds for point in candidate.target_points] == pytest.approx(
        [0.08, 0.16, 0.24]
    )
    assert [point.yield_probability for point in candidate.target_points] == pytest.approx(
        [0.2, 0.7, 0.9]
    )

    path = tmp_path / "inventory.json"
    write_inventory(path=path, inventory=inventory)
    assert read_inventory(path) == inventory


def test_inventory_rejects_conflicting_overlapping_labels() -> None:
    floor = [1.0] * FRAMES_PER_SAMPLE
    user_yield = [0.0] * FRAMES_PER_SAMPLE
    changed = user_yield.copy()
    changed[10] = 0.5

    with pytest.raises(ValueError, match="Conflicting labels"):
        build_candidate_inventory(
            samples=(
                _sample("a" * 64, 0.0, floor, user_yield, "background"),
                _sample("b" * 64, 0.0, floor, changed, "background"),
            ),
            corpus_repository="owner/corpus",
            corpus_revision=CORPUS_REVISION,
            split=TrainingCorpusSplit.VALIDATION,
        )


def _sample(
    window_id: str,
    start_seconds: float,
    floor: list[float],
    user_yield: list[float],
    category: str,
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
        start_seconds=start_seconds,
        end_seconds=start_seconds + 20.0,
        quality_score=0.9,
        category=category,
        assistant_has_floor=zeros,
        p_user_has_floor=tuple(floor),
        p_user_yield=tuple(user_yield),
        p_assistant_backchannel=zeros,
        future_activity_0_200=zeros,
        future_activity_200_500=zeros,
        future_activity_500_1000=zeros,
        future_activity_1000_1500=zeros,
        turn_completion=zeros,
        continuation_pause=zeros,
        non_floor_feedback=zeros,
        floor_take=zeros,
    )
