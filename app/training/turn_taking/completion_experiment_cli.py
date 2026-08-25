from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from app.training.turn_taking.benchmark_cli import PINNED_CORPUS_REVISION
from app.training.turn_taking.config import PINNED_NEMOTRON_REVISION


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    final_checkpoint: Path
    best_checkpoint: Path
    inventory: Path
    predictions: Path
    reports: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate one completion-primary turn detector on a CUDA node."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--hub-cache-directory", type=Path, required=True)
    parser.add_argument("--hub-revision", default=PINNED_CORPUS_REVISION)
    parser.add_argument("--model-revision", default=PINNED_NEMOTRON_REVISION)
    parser.add_argument("--max-steps", type=_positive_int, default=2_500)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=2)
    parser.add_argument("--data-loader-workers", type=_nonnegative_int, default=4)
    parser.add_argument("--run-seed", type=_nonnegative_int, default=20_260_825)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for the completion experiment.")

    paths = _experiment_paths(arguments.output_directory)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.predictions.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    common_hub_arguments = (
        "--hub-revision",
        arguments.hub_revision,
        "--hub-cache-directory",
        str(arguments.hub_cache_directory),
    )
    _run_module(
        "app.training.turn_taking.cli",
        str(paths.final_checkpoint),
        *common_hub_arguments,
        "--model-revision",
        arguments.model_revision,
        "--primary-objective",
        "turn_completion",
        "--max-steps",
        str(arguments.max_steps),
        "--batch-size",
        str(arguments.batch_size),
        "--gradient-accumulation-steps",
        str(arguments.gradient_accumulation_steps),
        "--data-loader-workers",
        str(arguments.data_loader_workers),
        "--run-seed",
        str(arguments.run_seed),
    )
    if not paths.best_checkpoint.exists():
        raise ValueError("Training finished without producing a best validation checkpoint.")

    _run_module(
        "app.training.turn_taking.benchmark_cli",
        "completion-inventory-v2",
        str(paths.inventory),
        *common_hub_arguments,
    )
    for detector in ("silero", "smart-turn", "livekit"):
        prediction_path = paths.predictions / f"validation-{detector}.json"
        report_path = paths.reports / f"validation-{detector}.json"
        _run_module(
            "app.training.turn_taking.benchmark_cli",
            "predict-completion-baseline-v2",
            str(paths.inventory),
            str(prediction_path),
            "--detector",
            detector,
            *common_hub_arguments,
        )
        _analyze(paths.inventory, prediction_path, report_path)

    checkpoint = torch.load(paths.best_checkpoint, map_location="cpu", weights_only=False)
    optimizer_step = int(checkpoint["optimizer_step"])
    _run_module(
        "app.training.turn_taking.benchmark_cli",
        "predict-voice-light-completion-v2",
        str(paths.inventory),
        str(paths.predictions),
        str(paths.best_checkpoint),
        *common_hub_arguments,
        "--model-revision",
        arguments.model_revision,
        "--batch-size",
        str(arguments.batch_size),
        "--data-loader-workers",
        str(arguments.data_loader_workers),
    )
    voice_light_predictions = paths.predictions / (
        f"validation-v2-turn-completion-voice-light-step-{optimizer_step:06d}-predictions.json"
    )
    _analyze(
        paths.inventory,
        voice_light_predictions,
        paths.reports / "validation-voice-light-best.json",
    )
    print(
        f"experiment_complete; best_checkpoint={paths.best_checkpoint}; reports={paths.reports}",
        flush=True,
    )


def _experiment_paths(root: Path) -> ExperimentPaths:
    return ExperimentPaths(
        root=root,
        final_checkpoint=root / "adapter-final.pt",
        best_checkpoint=root / "adapter-final-best.pt",
        inventory=root / "validation-completion-inventory.json",
        predictions=root / "predictions",
        reports=root / "reports",
    )


def _analyze(inventory: Path, predictions: Path, report: Path) -> None:
    _run_module(
        "app.training.turn_taking.benchmark_cli",
        "analyze-completion-v2",
        str(inventory),
        str(predictions),
        str(report),
    )


def _run_module(module: str, *arguments: str) -> None:
    command = (sys.executable, "-m", module, *arguments)
    print(f"running={' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


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
