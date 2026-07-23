from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from torch import Tensor
from torch.utils.data import Dataset

from app.local.training_corpus.export import (
    ExportManifest,
    MaterializedTrainingSample,
)
from app.training.turn_taking.data import FrameTargets, TrainingItem, load_audio_window

DEFAULT_HUB_REPOSITORY = "BertilBraun/voice-light-audio"
HUB_REPOSITORY_TYPE: Literal["dataset"] = "dataset"


class HubFileDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: Literal["dataset"],
        filename: str,
        cache_dir: str | Path | None,
    ) -> str: ...


class HuggingFaceTurnTakingDataset(Dataset[TrainingItem]):
    """Indexed Parquet supervision with lazy, Hugging Face-cached audio access."""

    def __init__(
        self,
        repository_id: str = DEFAULT_HUB_REPOSITORY,
        cache_directory: Path | None = None,
        sample_rate_hz: int = 16_000,
        downloader: HubFileDownloader = hf_hub_download,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        self.repository_id = repository_id
        self.cache_directory = cache_directory
        self.sample_rate_hz = sample_rate_hz
        self.downloader = downloader
        self.manifest = self._load_manifest()
        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingItem:
        sample = self.samples[index]
        audio_path = self._download(sample.user_audio_path)
        waveform = load_audio_window(
            path=audio_path,
            sample_rate_hz=self.sample_rate_hz,
            start_seconds=sample.start_seconds,
            end_seconds=sample.end_seconds,
        )
        return TrainingItem(
            sample_id=_training_item_id(sample),
            waveform=waveform,
            assistant_speaking=torch.tensor(sample.assistant_has_floor, dtype=torch.float32),
            targets=_frame_targets(sample),
        )

    def _load_manifest(self) -> ExportManifest:
        path = self._download("metadata/manifest.json")
        return ExportManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def _load_samples(self) -> tuple[MaterializedTrainingSample, ...]:
        samples: list[MaterializedTrainingSample] = []
        for shard_index in range(self.manifest.shard_count):
            path = self._download(f"training/shards/shard-{shard_index:05d}.parquet")
            rows = pq.read_table(path).to_pylist()
            samples.extend(MaterializedTrainingSample.model_validate(row) for row in rows)
        if len(samples) != self.manifest.training_sample_count:
            raise ValueError(
                "Expected "
                f"{self.manifest.training_sample_count} training samples, found {len(samples)}."
            )
        return tuple(samples)

    def _download(self, filename: str) -> Path:
        return Path(
            self.downloader(
                repo_id=self.repository_id,
                repo_type=HUB_REPOSITORY_TYPE,
                filename=filename,
                cache_dir=self.cache_directory,
            )
        )


def _frame_targets(sample: MaterializedTrainingSample) -> FrameTargets:
    yield_probability, primary_mask = _targets_and_mask(sample.p_user_yield)
    event_targets, event_mask = _stack_targets_and_masks(
        (
            sample.turn_completion,
            sample.continuation_pause,
            sample.p_assistant_backchannel,
            sample.non_floor_feedback,
            sample.floor_take,
        )
    )
    future_activity, future_activity_mask = _stack_targets_and_masks(
        (
            sample.future_activity_0_200,
            sample.future_activity_200_500,
            sample.future_activity_500_1000,
            sample.future_activity_1000_1500,
        )
    )
    return FrameTargets(
        yield_probability=yield_probability,
        primary_weight=primary_mask.float(),
        primary_mask=primary_mask,
        event_targets=event_targets,
        event_mask=event_mask,
        future_activity=future_activity,
        future_activity_mask=future_activity_mask,
    )


def _targets_and_mask(values: tuple[float, ...]) -> tuple[Tensor, Tensor]:
    raw = torch.tensor(values, dtype=torch.float32)
    mask = raw >= 0.0
    return raw.masked_fill(~mask, 0.0), mask


def _stack_targets_and_masks(values: tuple[tuple[float, ...], ...]) -> tuple[Tensor, Tensor]:
    columns = tuple(_targets_and_mask(column) for column in values)
    targets = torch.stack(tuple(item[0] for item in columns), dim=-1)
    mask = torch.stack(tuple(item[1] for item in columns), dim=-1)
    return targets, mask


def _training_item_id(sample: MaterializedTrainingSample) -> str:
    return f"{sample.sample_id}:{sample.user_side.value}:{sample.start_seconds:.2f}"
