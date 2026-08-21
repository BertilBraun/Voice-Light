from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.backbone import NemotronStreamingBackbone
from app.training.turn_taking.config import TrainingConfig
from app.training.turn_taking.data import collate_training_items
from app.training.turn_taking.evaluation import (
    EvaluationReport,
    evaluate_models,
    evaluate_oracle_vad,
    fit_class_priors,
)
from app.training.turn_taking.hub import (
    DEFAULT_HUB_REPOSITORY,
    HuggingFaceTurnTakingDataset,
    frame_targets_from_sample,
)
from app.training.turn_taking.model import TurnTakingAdapter

HASH_CHUNK_BYTES = 1024 * 1024
RANDOM_BASELINE_SEED = 1_729


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a turn-taking adapter checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hub-repository", default=DEFAULT_HUB_REPOSITORY)
    parser.add_argument("--hub-revision", required=True)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--data-loader-workers", type=_nonnegative_int, default=4)
    arguments = parser.parse_args()

    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    config = TrainingConfig.model_validate(checkpoint["training_config"])
    trained_adapter = TurnTakingAdapter(config.adapter)
    trained_adapter.load_state_dict(checkpoint["adapter_state"], strict=True)
    torch.manual_seed(RANDOM_BASELINE_SEED)
    random_adapter = TurnTakingAdapter(config.adapter)

    training_dataset = HuggingFaceTurnTakingDataset(
        split=TrainingCorpusSplit.TRAIN,
        revision=arguments.hub_revision,
        repository_id=arguments.hub_repository,
        cache_directory=arguments.hub_cache_directory,
        sample_rate_hz=config.sample_rate_hz,
    )
    priors = fit_class_priors(
        frame_targets_from_sample(sample) for sample in training_dataset.samples
    )
    validation_dataset = HuggingFaceTurnTakingDataset(
        split=TrainingCorpusSplit.VALIDATION,
        revision=arguments.hub_revision,
        repository_id=arguments.hub_repository,
        cache_directory=arguments.hub_cache_directory,
        sample_rate_hz=config.sample_rate_hz,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_training_items,
        num_workers=arguments.data_loader_workers,
        prefetch_factor=2 if arguments.data_loader_workers > 0 else None,
        persistent_workers=arguments.data_loader_workers > 0,
        pin_memory=device.type == "cuda",
    )
    backbone = NemotronStreamingBackbone(
        model_identifier=config.model_identifier,
        tap_layer_indices=config.adapter.tap_layer_indices,
        lookahead_tokens=config.lookahead_tokens,
    ).to(device)
    backbone.eval()
    models = evaluate_models(
        backbone=backbone,
        trained_adapter=trained_adapter,
        random_adapter=random_adapter,
        batches=loader,
        priors=priors,
        loss_config=config.loss,
        device=device,
    )
    models = models + (evaluate_oracle_vad(validation_dataset.samples, priors, config.loss),)
    report = EvaluationReport(
        generated_at=datetime.now(UTC),
        hub_revision=arguments.hub_revision,
        checkpoint_sha256=_file_sha256(arguments.checkpoint),
        optimizer_step=checkpoint["optimizer_step"],
        validation_sample_count=len(validation_dataset),
        random_seed=RANDOM_BASELINE_SEED,
        class_priors=priors,
        models=models,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    for model in report.models:
        print(
            f"{model.model.value}: total={model.total_loss:.6f}; "
            f"primary={model.primary_loss:.6f}; events={model.event_loss:.6f}; "
            f"future={model.future_activity_loss:.6f}",
            flush=True,
        )
    print(f"Wrote evaluation report to {arguments.output}", flush=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


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
