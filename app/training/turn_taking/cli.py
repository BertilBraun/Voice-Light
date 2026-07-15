from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.training.turn_taking.backbone import NemotronStreamingBackbone
from app.training.turn_taking.config import TrainingConfig
from app.training.turn_taking.data import (
    TurnTakingDataset,
    WaveformAugmenter,
    collate_training_items,
)
from app.training.turn_taking.model import TurnTakingAdapter
from app.training.turn_taking.schema import read_manifest
from app.training.turn_taking.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen-Nemotron turn-taking adapter.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int)
    arguments = parser.parse_args()
    base_config = TrainingConfig()
    config = (
        base_config.model_copy(update={"max_steps": arguments.max_steps})
        if arguments.max_steps is not None
        else base_config
    )
    torch.manual_seed(config.random_seed)
    samples = read_manifest(arguments.manifest)
    dataset = TurnTakingDataset(
        samples=samples,
        frame_seconds=config.encoder_frame_seconds,
        burn_in_seconds=config.burn_in_seconds,
        unmeasured_reliability_weight=config.unmeasured_reliability_weight,
        augmenter=WaveformAugmenter(),
        random_seed=config.random_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_training_items,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        f"loss={result.final_loss:.4f}; checkpoint={result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
