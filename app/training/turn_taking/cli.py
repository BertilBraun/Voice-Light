from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.backbone import NemotronStreamingBackbone
from app.training.turn_taking.benchmark_completion_inventory import (
    build_turn_completion_inventory,
)
from app.training.turn_taking.completion_training import (
    CompletionBoundaryDataset,
    balanced_completion_weights,
    build_completion_boundaries,
    build_inventory_completion_boundaries,
)
from app.training.turn_taking.completion_validation import CompletionValidator
from app.training.turn_taking.config import (
    TrainingConfig,
    TrainingPrecision,
    TurnCompletionObjectiveConfig,
    UserYieldObjectiveConfig,
    WaveformAugmentationProfile,
    waveform_augmentation_config,
)
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
    parser.add_argument("--model-revision")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=_positive_int)
    parser.add_argument("--minimum-steps-before-stopping", type=_nonnegative_int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int)
    parser.add_argument("--data-loader-workers", type=_nonnegative_int)
    parser.add_argument("--run-seed", type=_nonnegative_int)
    parser.add_argument(
        "--augmentation-profile",
        type=WaveformAugmentationProfile,
        choices=tuple(WaveformAugmentationProfile),
        default=WaveformAugmentationProfile.EXPANDED,
    )
    parser.add_argument(
        "--primary-objective",
        choices=("user_yield", "turn_completion"),
    )
    parser.add_argument(
        "--precision",
        choices=tuple(precision.value for precision in TrainingPrecision),
    )
    arguments = parser.parse_args()
    if arguments.resume_checkpoint is None:
        config = TrainingConfig()
        if arguments.max_steps is not None:
            config = config.model_copy(
                update={
                    "max_steps": arguments.max_steps,
                    "target_optimizer_step": arguments.max_steps,
                }
            )
    else:
        checkpoint = torch.load(arguments.resume_checkpoint, map_location="cpu", weights_only=False)
        config = TrainingConfig.model_validate(checkpoint["training_config"])
        if arguments.max_steps is None:
            parser.error("--max-steps is required when resuming a checkpoint.")
        config = config.model_copy(update={"target_optimizer_step": arguments.max_steps})
        if arguments.run_seed is None:
            config = config.model_copy(
                update={"random_seed": config.random_seed + checkpoint["optimizer_step"]}
            )
    if arguments.batch_size is not None:
        config = config.model_copy(update={"batch_size": arguments.batch_size})
    if arguments.gradient_accumulation_steps is not None:
        config = config.model_copy(
            update={"gradient_accumulation_steps": arguments.gradient_accumulation_steps}
        )
    if arguments.data_loader_workers is not None:
        config = config.model_copy(update={"data_loader_workers": arguments.data_loader_workers})
    if arguments.minimum_steps_before_stopping is not None:
        config = config.model_copy(
            update={"minimum_steps_before_stopping": arguments.minimum_steps_before_stopping}
        )
    if arguments.precision is not None:
        config = config.model_copy(update={"precision": TrainingPrecision(arguments.precision)})
    if arguments.model_revision is not None:
        config = config.model_copy(update={"model_revision": arguments.model_revision})
    if arguments.run_seed is not None:
        config = config.model_copy(update={"random_seed": arguments.run_seed})
    config = config.model_copy(
        update={"augmentation": waveform_augmentation_config(arguments.augmentation_profile)}
    )
    if arguments.primary_objective is not None:
        primary_objective = (
            TurnCompletionObjectiveConfig()
            if arguments.primary_objective == "turn_completion"
            else UserYieldObjectiveConfig()
        )
        config = config.model_copy(
            update={"loss": config.loss.model_copy(update={"primary_objective": primary_objective})}
        )
    torch.manual_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arguments.manifest is not None:
        samples = read_manifest(arguments.manifest)
        dataset = TurnTakingDataset(
            samples=samples,
            frame_seconds=config.encoder_frame_seconds,
            burn_in_seconds=config.burn_in_seconds,
            unmeasured_reliability_weight=config.unmeasured_reliability_weight,
            augmenter=WaveformAugmenter(config.augmentation, config.sample_rate_hz),
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
            augmenter=(
                WaveformAugmenter(config.augmentation, config.sample_rate_hz)
                if hub_split is TrainingCorpusSplit.TRAIN
                else None
            ),
            random_seed=config.random_seed,
        )
    sampler: WeightedRandomSampler[int] | None = None
    match config.loss.primary_objective:
        case TurnCompletionObjectiveConfig() as completion_objective:
            if arguments.manifest is not None:
                parser.error("Turn-completion training requires the pinned Hub corpus.")
            if hub_split is not TrainingCorpusSplit.TRAIN:
                parser.error("Turn-completion training requires --hub-split train.")
            boundaries = build_completion_boundaries(dataset.samples, completion_objective)
            dataset = CompletionBoundaryDataset(dataset, boundaries)
            sampler = WeightedRandomSampler(
                weights=balanced_completion_weights(boundaries),
                num_samples=len(boundaries),
                replacement=True,
                generator=torch.Generator().manual_seed(config.random_seed + 1),
            )
            hold_count = sum(boundary.completion_class.value == "hold" for boundary in boundaries)
            print(
                f"completion_boundaries={len(boundaries)}; hold={hold_count}; "
                f"eot={len(boundaries) - hold_count}",
                flush=True,
            )
        case UserYieldObjectiveConfig():
            pass
    data_loader_generator = torch.Generator().manual_seed(config.random_seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_training_items,
        num_workers=config.data_loader_workers,
        prefetch_factor=(
            config.data_loader_prefetch_factor if config.data_loader_workers > 0 else None
        ),
        persistent_workers=config.data_loader_workers > 0,
        pin_memory=device.type == "cuda",
        generator=data_loader_generator,
    )
    optimizer_steps_per_epoch = len(loader) / config.gradient_accumulation_steps
    print(
        f"training_items={len(dataset)}; optimizer_steps_per_epoch="
        f"{optimizer_steps_per_epoch:.2f}; target_epochs="
        f"{(config.target_optimizer_step or config.max_steps) / optimizer_steps_per_epoch:.2f}; "
        f"run_seed={config.random_seed}",
        flush=True,
    )
    backbone = NemotronStreamingBackbone(
        model_identifier=config.model_identifier,
        tap_layer_indices=config.adapter.tap_layer_indices,
        lookahead_tokens=config.lookahead_tokens,
        model_revision=config.model_revision,
        cache_directory=arguments.hub_cache_directory,
    ).to(device)
    validation_callback: CompletionValidator | None = None
    match config.loss.primary_objective:
        case TurnCompletionObjectiveConfig() as completion_objective:
            validation_source = HuggingFaceTurnTakingDataset(
                split=TrainingCorpusSplit.VALIDATION,
                revision=arguments.hub_revision,
                repository_id=arguments.hub_repository,
                cache_directory=arguments.hub_cache_directory,
                sample_rate_hz=config.sample_rate_hz,
                pad_missing_audio_suffix=True,
            )
            validation_inventory = build_turn_completion_inventory(
                samples=validation_source.samples,
                corpus_repository=arguments.hub_repository,
                corpus_revision=arguments.hub_revision,
                split=TrainingCorpusSplit.VALIDATION,
            )
            validation_boundaries = build_inventory_completion_boundaries(
                validation_source.samples,
                validation_inventory,
                completion_objective,
            )
            validation_dataset = CompletionBoundaryDataset(
                validation_source,
                validation_boundaries,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_training_items,
                num_workers=config.data_loader_workers,
                prefetch_factor=(
                    config.data_loader_prefetch_factor if config.data_loader_workers > 0 else None
                ),
                persistent_workers=config.data_loader_workers > 0,
                pin_memory=device.type == "cuda",
            )
            validation_callback = CompletionValidator(backbone, validation_loader, device)
        case UserYieldObjectiveConfig():
            pass
    result = train(
        backbone=backbone,
        adapter=TurnTakingAdapter(config.adapter),
        batches=loader,
        config=config,
        checkpoint_path=arguments.checkpoint,
        device=device,
        resume_checkpoint_path=arguments.resume_checkpoint,
        validation_callback=validation_callback,
    )
    print(
        f"Completed optimizer steps {result.starting_optimizer_step} "
        f"through {result.optimizer_steps}; "
        f"loss={result.final_loss:.4f}; "
        f"steps_per_second={result.optimizer_steps_per_second:.3f}; "
        f"peak_allocated_gib={_gibibytes(result.peak_device_memory_bytes)}; "
        f"peak_reserved_gib={_gibibytes(result.peak_reserved_device_memory_bytes)}; "
        f"checkpoint={result.checkpoint_path}; best_checkpoint={result.best_checkpoint_path}; "
        f"best_validation_step={result.best_validation_step}; "
        f"best_validation_score={result.best_validation_score}; "
        f"stopped_early={result.stopped_early}"
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
