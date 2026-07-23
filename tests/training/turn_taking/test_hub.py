from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from app.local.db.models import TrackSide
from app.local.training_corpus.export import ExportManifest, MaterializedTrainingSample
from app.training.turn_taking.hub import HuggingFaceTurnTakingDataset


def test_hub_dataset_indexes_parquet_and_downloads_audio_lazily(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "repository"
    manifest = ExportManifest(
        schema_version="voice-light-turn-taking-v1",
        generated_at=datetime.now(UTC),
        metric_version="quality-v1",
        annotation_version="annotation-v1",
        dataset_ids=(uuid4(),),
        minimum_quality=0.95,
        recording_count=1,
        training_sample_count=1,
        shard_count=1,
    )
    _write(root / "metadata" / "manifest.json", manifest.model_dump_json())
    sample = _sample()
    shard_path = root / "training" / "shards" / "shard-00000.parquet"
    shard_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([sample.model_dump(mode="json")]), shard_path)
    download_calls: list[str] = []

    def download(
        *, repo_id: str, repo_type: str, filename: str, cache_dir: str | Path | None
    ) -> str:
        assert repo_id == "test/corpus"
        assert repo_type == "dataset"
        assert cache_dir == tmp_path / "cache"
        download_calls.append(filename)
        return str(root / filename)

    monkeypatch.setattr(
        "app.training.turn_taking.hub.load_audio_window",
        lambda **_: torch.ones(320, dtype=torch.float32),
    )
    dataset = HuggingFaceTurnTakingDataset(
        repository_id="test/corpus",
        cache_directory=tmp_path / "cache",
        downloader=download,
    )

    assert len(dataset) == 1
    assert download_calls == ["metadata/manifest.json", "training/shards/shard-00000.parquet"]
    item = dataset[0]

    assert item.waveform.shape == (320,)
    assert item.assistant_speaking.shape == (250,)
    assert item.targets.primary_mask[1]
    assert not item.targets.primary_mask[0]
    assert item.targets.event_mask.shape == (250, 5)
    assert item.targets.future_activity_mask.shape == (250, 4)
    assert download_calls[-1] == sample.user_audio_path


def _sample() -> MaterializedTrainingSample:
    labels = tuple(-1.0 if index == 0 else 0.5 for index in range(250))
    return MaterializedTrainingSample(
        schema_version="voice-light-turn-taking-v1",
        dataset_name="test",
        sample_id=uuid4(),
        external_id="recording",
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        user_audio_path="dataset_1/recording/recording_speaker1.flac",
        assistant_audio_path="dataset_1/recording/recording_speaker2.flac",
        start_seconds=12.0,
        end_seconds=32.0,
        quality_score=0.99,
        category="dense_turn_taking",
        assistant_has_floor=labels,
        p_user_has_floor=labels,
        p_user_yield=labels,
        p_assistant_backchannel=labels,
        future_activity_0_200=labels,
        future_activity_200_500=labels,
        future_activity_500_1000=labels,
        future_activity_1000_1500=labels,
        turn_completion=labels,
        continuation_pause=labels,
        non_floor_feedback=labels,
        floor_take=labels,
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
