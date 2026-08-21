from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq
import pytest

from app.local.db.models import TrackSide
from app.local.training_corpus.export import (
    FRAMES_PER_SAMPLE,
    MaterializedTrainingSample,
    _validate_export_destination,
    _window_id,
    _write_training_shards,
)
from app.local.training_corpus.splits import TrainingCorpusSplit

DATASET_ID = UUID("00000000-0000-0000-0000-000000000001")
SAMPLE_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_training_shards_are_separated_and_use_fixed_size_frame_arrays(
    tmp_path: Path,
) -> None:
    samples = (
        training_sample(TrainingCorpusSplit.TRAIN, SAMPLE_ID),
        training_sample(
            TrainingCorpusSplit.VALIDATION,
            UUID("00000000-0000-0000-0000-000000000003"),
        ),
        training_sample(
            TrainingCorpusSplit.TEST,
            UUID("00000000-0000-0000-0000-000000000004"),
        ),
    )

    shards = _write_training_shards(output_directory=tmp_path, samples=samples)

    assert tuple(shard.path for shard in shards) == (
        "training/train/shard-00000.parquet",
        "training/validation/shard-00000.parquet",
        "training/test/shard-00000.parquet",
    )
    assert all(shard.row_count == 1 and shard.size_bytes > 0 for shard in shards)
    table = pq.read_table(tmp_path / shards[0].path)
    assert table.schema.field("p_user_yield").type.list_size == FRAMES_PER_SAMPLE
    assert table.column("split").to_pylist() == ["train"]


def test_window_id_is_stable_and_changes_with_orientation_or_frame() -> None:
    first = _window_id(DATASET_ID, SAMPLE_ID, TrackSide.SPEAKER1, 16.0)

    assert first == _window_id(DATASET_ID, SAMPLE_ID, TrackSide.SPEAKER1, 16.0)
    assert first != _window_id(DATASET_ID, SAMPLE_ID, TrackSide.SPEAKER2, 16.0)
    assert first != _window_id(DATASET_ID, SAMPLE_ID, TrackSide.SPEAKER1, 16.08)


@pytest.mark.parametrize("managed_name", ("training", "corpus.json"))
def test_export_destination_rejects_managed_output_collisions(
    tmp_path: Path,
    managed_name: str,
) -> None:
    managed_path = tmp_path / managed_name
    if managed_path.suffix:
        managed_path.write_text("occupied", encoding="utf-8")
    else:
        managed_path.mkdir()

    with pytest.raises(ValueError, match="Export-managed output already exists"):
        _validate_export_destination(tmp_path)


def test_export_destination_accepts_existing_staged_audio(tmp_path: Path) -> None:
    (tmp_path / "dataset_2" / "samples").mkdir(parents=True)

    _validate_export_destination(tmp_path)

    assert tmp_path.is_dir()


def training_sample(
    split: TrainingCorpusSplit,
    sample_id: UUID,
) -> MaterializedTrainingSample:
    values = tuple(0.0 for _ in range(FRAMES_PER_SAMPLE))
    return MaterializedTrainingSample(
        schema_version="voice-light-turn-taking-v1",
        training_label_version="turn-taking-frame-labels-v1",
        window_id=_window_id(DATASET_ID, sample_id, TrackSide.SPEAKER1, 0.0),
        dataset_id=DATASET_ID,
        dataset_name="dataset",
        sample_id=sample_id,
        external_id=f"sample-{sample_id.hex}",
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        split=split,
        user_audio_path="dataset/samples/sample/speaker_1.flac",
        assistant_audio_path="dataset/samples/sample/speaker_2.flac",
        start_seconds=0.0,
        end_seconds=20.0,
        quality_score=0.99,
        category="background",
        assistant_has_floor=values,
        p_user_has_floor=values,
        p_user_yield=values,
        p_assistant_backchannel=values,
        future_activity_0_200=values,
        future_activity_200_500=values,
        future_activity_500_1000=values,
        future_activity_1000_1500=values,
        turn_completion=values,
        continuation_pause=values,
        non_floor_feedback=values,
        floor_take=values,
    )
