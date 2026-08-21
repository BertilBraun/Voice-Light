from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from app.local.db.models import TrackSide
from app.local.training_corpus.export import (
    ExportManifest,
    ExportShard,
    ExportSplitSummary,
    MaterializedTrainingSample,
)
from app.local.training_corpus.splits import (
    ConversationSplitAssignment,
    ConversationSplitPlan,
    TrainingCorpusSplit,
)
from app.training.turn_taking.hub import HuggingFaceTurnTakingDataset, validate_sample_contract


def test_hub_dataset_indexes_parquet_and_downloads_audio_lazily(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "repository"
    validation_sample = _sample(split=TrainingCorpusSplit.VALIDATION)
    train_sample = _sample(split=TrainingCorpusSplit.TRAIN)
    validation_shard = ExportShard(
        split=TrainingCorpusSplit.VALIDATION,
        path="training/validation/shard-00000.parquet",
        row_count=1,
        size_bytes=1,
        sha256="1" * 64,
    )
    train_shard = ExportShard(
        split=TrainingCorpusSplit.TRAIN,
        path="training/train/shard-00000.parquet",
        row_count=1,
        size_bytes=1,
        sha256="2" * 64,
    )
    manifest = _manifest(
        samples=(train_sample, validation_sample),
        shards=(train_shard, validation_shard),
    )
    _write(root / "corpus.json", manifest.model_dump_json())
    _write_shard(root / validation_shard.path, validation_sample)
    _write_shard(root / train_shard.path, train_sample)
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
        split=TrainingCorpusSplit.VALIDATION,
        repository_id="test/corpus",
        cache_directory=tmp_path / "cache",
        downloader=download,
    )

    assert len(dataset) == 1
    assert download_calls == ["corpus.json", validation_shard.path]
    item = dataset[0]

    assert item.sample_id == validation_sample.window_id
    assert item.waveform.shape == (320,)
    assert item.assistant_speaking.shape == (250,)
    assert item.targets.primary_mask[1]
    assert not item.targets.primary_mask[0]
    assert item.targets.event_mask.shape == (250, 5)
    assert item.targets.future_activity_mask.shape == (250, 4)
    assert download_calls[-1] == validation_sample.user_audio_path


def _sample(split: TrainingCorpusSplit = TrainingCorpusSplit.TRAIN) -> MaterializedTrainingSample:
    labels = tuple(-1.0 if index == 0 else 0.5 for index in range(250))
    assistant_floor = (0.5,) * 250
    return MaterializedTrainingSample(
        schema_version="voice-light-turn-taking-v1",
        training_label_version="labels-v1",
        window_id="a" * 64,
        dataset_id=uuid4(),
        dataset_name="test",
        sample_id=uuid4(),
        external_id="recording",
        user_side=TrackSide.SPEAKER1,
        assistant_side=TrackSide.SPEAKER2,
        split=split,
        user_audio_path="dataset_1/recording/recording_speaker1.flac",
        assistant_audio_path="dataset_1/recording/recording_speaker2.flac",
        start_seconds=12.0,
        end_seconds=32.0,
        quality_score=0.99,
        category="dense_turn_taking",
        assistant_has_floor=assistant_floor,
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


def test_training_sample_contract_rejects_mismatched_label_version() -> None:
    sample = _sample()
    manifest = _manifest(
        samples=(sample,),
        shards=(_shard(TrainingCorpusSplit.TRAIN),),
        training_label_version="labels-v2",
    )

    with pytest.raises(ValueError, match="does not match manifest label version"):
        validate_sample_contract(
            sample=sample,
            manifest=manifest,
            split=TrainingCorpusSplit.TRAIN,
        )


def test_training_sample_contract_rejects_mismatched_split() -> None:
    sample = _sample(split=TrainingCorpusSplit.VALIDATION)
    manifest = _manifest(
        samples=(sample,),
        shards=(_shard(TrainingCorpusSplit.VALIDATION),),
    )

    with pytest.raises(ValueError, match="does not match requested split"):
        validate_sample_contract(
            sample=sample,
            manifest=manifest,
            split=TrainingCorpusSplit.TRAIN,
        )


def test_materialized_sample_rejects_non_probability_training_target() -> None:
    payload = _sample().model_dump()
    payload["p_user_yield"] = (-0.5,) + (0.5,) * 249

    with pytest.raises(ValueError, match="masked with -1 or be probabilities"):
        MaterializedTrainingSample.model_validate(payload)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_shard(path: Path, sample: MaterializedTrainingSample) -> None:
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([sample.model_dump(mode="json")]), path)


def _shard(split: TrainingCorpusSplit) -> ExportShard:
    return ExportShard(
        split=split,
        path=f"training/{split.value}/shard-00000.parquet",
        row_count=1,
        size_bytes=1,
        sha256="1" * 64,
    )


def _manifest(
    samples: tuple[MaterializedTrainingSample, ...],
    shards: tuple[ExportShard, ...],
    training_label_version: str = "labels-v1",
) -> ExportManifest:
    return ExportManifest(
        schema_version="voice-light-turn-taking-v1",
        generated_at=datetime.now(UTC),
        metric_version="quality-v1",
        annotation_version="annotation-v1",
        region_analysis_version="regions-v1",
        training_label_version=training_label_version,
        input_duration_seconds=20.0,
        frame_seconds=0.08,
        review_set_name="test-review",
        split_plan=ConversationSplitPlan(
            seed="test-seed",
            assignments=tuple(
                ConversationSplitAssignment(
                    dataset_id=sample.dataset_id,
                    sample_id=sample.sample_id,
                    split=sample.split,
                )
                for sample in samples
            ),
        ),
        recording_count=len(samples),
        training_sample_count=len(samples),
        splits=tuple(
            ExportSplitSummary(
                split=split,
                recording_count=sum(sample.split is split for sample in samples),
                training_sample_count=sum(sample.split is split for sample in samples),
                source_duration_seconds=20.0 * sum(sample.split is split for sample in samples),
            )
            for split in TrainingCorpusSplit
        ),
        shards=shards,
    )
