from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from torch import Tensor
from torch.utils.data import Dataset

from app.local.training_corpus.export import (
    ExportManifest,
    ExportSplitSummary,
    MaterializedTrainingSample,
)
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.data import FrameTargets, TrainingItem, load_audio_window

DEFAULT_HUB_REPOSITORY = "BertilBraun/voice-light-audio"
HUB_REPOSITORY_TYPE: Literal["dataset"] = "dataset"
PINNED_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HASH_CHUNK_BYTES = 1024 * 1024


class HubFileDownloader(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: Literal["dataset"],
        filename: str,
        revision: str,
        cache_dir: str | Path | None,
    ) -> str: ...


class HuggingFaceTurnTakingDataset(Dataset[TrainingItem]):
    """Indexed Parquet supervision with lazy, Hugging Face-cached audio access."""

    def __init__(
        self,
        split: TrainingCorpusSplit,
        revision: str,
        repository_id: str = DEFAULT_HUB_REPOSITORY,
        cache_directory: Path | None = None,
        sample_rate_hz: int = 16_000,
        augmenter: Callable[[Tensor, random.Random], Tensor] | None = None,
        random_seed: int = 17,
        downloader: HubFileDownloader = hf_hub_download,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if PINNED_REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError("revision must be an immutable 40-character commit SHA.")
        self.split = split
        self.revision = revision
        self.repository_id = repository_id
        self.cache_directory = cache_directory
        self.sample_rate_hz = sample_rate_hz
        self.augmenter = augmenter
        self.random_seed = random_seed
        self.augmentation_worker_seed: int | None = None
        self.augmentation_generator = random.Random()
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
        if self.augmenter is not None:
            waveform = self.augmenter(waveform, self._worker_augmentation_generator())
        return TrainingItem(
            sample_id=_training_item_id(sample),
            waveform=waveform,
            assistant_speaking=torch.tensor(sample.assistant_has_floor, dtype=torch.float32),
            targets=frame_targets_from_sample(sample),
        )

    def _load_manifest(self) -> ExportManifest:
        path = self._download("corpus.json")
        return ExportManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def _load_samples(self) -> tuple[MaterializedTrainingSample, ...]:
        samples: list[MaterializedTrainingSample] = []
        shards = tuple(shard for shard in self.manifest.shards if shard.split is self.split)
        for shard in shards:
            path = self._download(shard.path)
            if path.stat().st_size != shard.size_bytes:
                raise ValueError(f"Shard {shard.path!r} size does not match the manifest.")
            if _file_sha256(path) != shard.sha256:
                raise ValueError(f"Shard {shard.path!r} hash does not match the manifest.")
            rows = pq.read_table(path).to_pylist()
            if len(rows) != shard.row_count:
                raise ValueError(
                    f"Shard {shard.path!r} declares {shard.row_count} rows, found {len(rows)}."
                )
            for row in rows:
                sample = MaterializedTrainingSample.model_validate(row)
                validate_sample_contract(
                    sample=sample,
                    manifest=self.manifest,
                    split=self.split,
                )
                samples.append(sample)
        split_summary = _split_summary(manifest=self.manifest, split=self.split)
        if len(samples) != split_summary.training_sample_count:
            raise ValueError(
                "Expected "
                f"{split_summary.training_sample_count} {self.split.value} samples, "
                f"found {len(samples)}."
            )
        return tuple(samples)

    def _worker_augmentation_generator(self) -> random.Random:
        worker_seed = torch.initial_seed()
        combined_seed = worker_seed ^ self.random_seed
        if self.augmentation_worker_seed != combined_seed:
            self.augmentation_generator.seed(combined_seed)
            self.augmentation_worker_seed = combined_seed
        return self.augmentation_generator

    def _download(self, filename: str) -> Path:
        return Path(
            self.downloader(
                repo_id=self.repository_id,
                repo_type=HUB_REPOSITORY_TYPE,
                filename=filename,
                revision=self.revision,
                cache_dir=self.cache_directory,
            )
        )


def frame_targets_from_sample(sample: MaterializedTrainingSample) -> FrameTargets:
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


def validate_sample_contract(
    sample: MaterializedTrainingSample,
    manifest: ExportManifest,
    split: TrainingCorpusSplit,
) -> None:
    if sample.schema_version != manifest.schema_version:
        raise ValueError(
            f"Training sample schema {sample.schema_version!r} does not match "
            f"manifest schema {manifest.schema_version!r}."
        )
    if sample.training_label_version != manifest.training_label_version:
        raise ValueError(
            f"Training sample label version {sample.training_label_version!r} does not match "
            f"manifest label version {manifest.training_label_version!r}."
        )
    if sample.split is not split:
        raise ValueError(
            f"Training sample split {sample.split.value!r} does not match "
            f"requested split {split.value!r}."
        )
    expected_frame_count = round(manifest.input_duration_seconds / manifest.frame_seconds)
    if expected_frame_count != len(sample.p_user_yield):
        raise ValueError(
            f"Training sample has {len(sample.p_user_yield)} frames; "
            f"manifest contract requires {expected_frame_count}."
        )
    sample_duration_seconds = sample.end_seconds - sample.start_seconds
    if abs(sample_duration_seconds - manifest.input_duration_seconds) > 1e-9:
        raise ValueError(
            f"Training sample duration {sample_duration_seconds} does not match "
            f"manifest duration {manifest.input_duration_seconds}."
        )


def _split_summary(
    manifest: ExportManifest,
    split: TrainingCorpusSplit,
) -> ExportSplitSummary:
    summaries = tuple(summary for summary in manifest.splits if summary.split is split)
    if len(summaries) != 1:
        raise ValueError(f"Corpus manifest must contain exactly one {split.value!r} split summary.")
    return summaries[0]


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
    return sample.window_id


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
