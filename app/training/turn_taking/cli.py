from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.backbone import NemotronStreamingBackbone
from app.training.turn_taking.config import TrainingConfig, TrainingPrecision
from app.training.turn_taking.data import (
    TurnTakingDataset,
    WaveformAugmenter,
    collate_training_items,
)
from app.training.turn_taking.hub import (
    DEFAULT_HUB_REPOSITORY,
    HuggingFaceTurnTakingDataset,
)
from app.training.turn_taking.model import TurnTakingAdapter
from app.training.turn_taking.schema import read_manifest
from app.training.turn_taking.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen-Nemotron turn-taking adapter.")
    parser.add_argument("checkpoint", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument("--hub-repository", default=DEFAULT_HUB_REPOSITORY)
    parser.add_argument("--hub-revision")
    parser.add_argument(
        "--hub-split",
        choices=tuple(split.value for split in TrainingCorpusSplit),
        default=TrainingCorpusSplit.TRAIN.value,
    )
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--max-steps", type=_positive_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int)
    parser.add_argument("--data-loader-workers", type=_nonnegative_int)
    parser.add_argument(
        "--precision",
        choices=tuple(precision.value for precision in TrainingPrecision),
    )
    arguments = parser.parse_args()
    config = TrainingConfig()
    if arguments.max_steps is not None:
        config = config.model_copy(update={"max_steps": arguments.max_steps})
    if arguments.batch_size is not None:
        config = config.model_copy(update={"batch_size": arguments.batch_size})
    if arguments.gradient_accumulation_steps is not None:
        config = config.model_copy(
            update={"gradient_accumulation_steps": arguments.gradient_accumulation_steps}
        )
    if arguments.data_loader_workers is not None:
        config = config.model_copy(update={"data_loader_workers": arguments.data_loader_workers})
    if arguments.precision is not None:
        config = config.model_copy(update={"precision": TrainingPrecision(arguments.precision)})
    torch.manual_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arguments.manifest is not None:
        samples = read_manifest(arguments.manifest)
        dataset = TurnTakingDataset(
            samples=samples,
            frame_seconds=config.encoder_frame_seconds,
            burn_in_seconds=config.burn_in_seconds,
            unmeasured_reliability_weight=config.unmeasured_reliability_weight,
            augmenter=WaveformAugmenter(),
            random_seed=config.random_seed,
        )
    else:
        if arguments.hub_revision is None:
            parser.error("--hub-revision is required when loading the Hub corpus.")
        hub_split = TrainingCorpusSplit(arguments.hub_split)
        dataset = HuggingFaceTurnTakingDataset(
            split=hub_split,
            revision=arguments.hub_revision,
            repository_id=arguments.hub_repository,
            cache_directory=arguments.hub_cache_directory,
            sample_rate_hz=config.sample_rate_hz,
            augmenter=WaveformAugmenter() if hub_split is TrainingCorpusSplit.TRAIN else None,
            random_seed=config.random_seed,
        )
    data_loader_generator = torch.Generator().manual_seed(config.random_seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_training_items,
        num_workers=config.data_loader_workers,
        prefetch_factor=(
            config.data_loader_prefetch_factor if config.data_loader_workers > 0 else None
        ),
        persistent_workers=config.data_loader_workers > 0,
        pin_memory=device.type == "cuda",
        generator=data_loader_generator,
    )
    backbone = NemotronStreamingBackbone(
        model_identifier=config.model_identifier,
        tap_layer_indices=config.adapter.tap_layer_indices,
        lookahead_tokens=config.lookahead_tokens,
    ).to(device)
    result = train(
        backbone=backbone,
        adapter=TurnTakingAdapter(config.adapter),
        batches=loader,
        config=config,
        checkpoint_path=arguments.checkpoint,
        device=device,
    )
    print(
        f"Completed {result.optimizer_steps} optimizer steps; "
        f"loss={result.final_loss:.4f}; "
        f"steps_per_second={result.optimizer_steps_per_second:.3f}; "
        f"peak_allocated_gib={_gibibytes(result.peak_device_memory_bytes)}; "
        f"peak_reserved_gib={_gibibytes(result.peak_reserved_device_memory_bytes)}; "
        f"checkpoint={result.checkpoint_path}"
    )


def _gibibytes(byte_count: int | None) -> str:
    return "n/a" if byte_count is None else f"{byte_count / 1024**3:.2f}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


if __name__ == "__main__":
    main()
