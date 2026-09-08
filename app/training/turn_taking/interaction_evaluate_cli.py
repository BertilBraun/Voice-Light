from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.training.turn_taking.backbone import NemotronStreamingBackbone
from app.training.turn_taking.benchmark_voice_light import load_voice_light_checkpoint
from app.training.turn_taking.interaction_baselines import (
    InteractionBaselineArtifact,
    InteractionBaselineKind,
    evaluate_interaction_baseline,
    predict_interaction_baseline,
    sweep_interaction_baseline,
)
from app.training.turn_taking.interaction_evaluation import (
    InteractionPredictionArtifact,
    evaluate_interaction_predictions,
    merge_interaction_prediction_artifacts,
)
from app.training.turn_taking.interaction_prediction import predict_synthetic_interactions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Voice Light backchannel and interruption decisions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="Run both auxiliary heads on synthetic events.")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("output", type=Path)
    predict.add_argument("--dynamic-synthetic-corpus", type=Path, action="append", required=True)
    predict.add_argument("--model-cache-directory", type=Path)
    predict.add_argument("--batch-size", type=_positive_int, default=4)
    predict.add_argument("--detection-horizon-seconds", type=_positive_float, default=0.8)
    predict.add_argument("--skip-events", type=_nonnegative_int, default=0)
    predict.add_argument("--maximum-events", type=_positive_int)
    analyze = subparsers.add_parser("analyze", help="Score an existing interaction artifact.")
    analyze.add_argument("predictions", type=Path)
    analyze.add_argument("output", type=Path)
    analyze.add_argument("--threshold", type=_probability, default=0.5)
    merge = subparsers.add_parser("merge", help="Merge durable prediction shards.")
    merge.add_argument("output", type=Path)
    merge.add_argument("predictions", type=Path, nargs="+")
    predict_baseline = subparsers.add_parser(
        "predict-baseline", help="Run an operational baseline on synthetic interaction events."
    )
    predict_baseline.add_argument("detector", type=InteractionBaselineKind)
    predict_baseline.add_argument("output", type=Path)
    predict_baseline.add_argument(
        "--dynamic-synthetic-corpus", type=Path, action="append", required=True
    )
    predict_baseline.add_argument("--model-cache-directory", type=Path)
    predict_baseline.add_argument("--detection-horizon-seconds", type=_positive_float, default=0.8)
    predict_baseline.add_argument("--crop-random-seed", type=_nonnegative_int, default=17)
    analyze_baseline = subparsers.add_parser(
        "analyze-baseline", help="Score saved operational baseline predictions."
    )
    analyze_baseline.add_argument("predictions", type=Path)
    analyze_baseline.add_argument("output", type=Path)
    analyze_baseline.add_argument("--threshold", type=_probability, required=True)
    sweep_baseline = subparsers.add_parser(
        "sweep-baseline", help="Sweep thresholds over saved operational predictions."
    )
    sweep_baseline.add_argument("predictions", type=Path)
    sweep_baseline.add_argument("output", type=Path)
    sweep_baseline.add_argument("--threshold-step", type=_positive_float, default=0.01)
    arguments = parser.parse_args()
    match arguments.command:
        case "predict":
            _predict(arguments)
        case "analyze":
            _analyze(arguments)
        case "merge":
            _merge(arguments)
        case "predict-baseline":
            _predict_baseline(arguments)
        case "analyze-baseline":
            _analyze_baseline(arguments)
        case "sweep-baseline":
            _sweep_baseline(arguments)
        case _:
            raise AssertionError(f"Unhandled interaction command {arguments.command!r}.")


def _predict(arguments: argparse.Namespace) -> None:
    checkpoint = load_voice_light_checkpoint(arguments.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = NemotronStreamingBackbone(
        model_identifier=checkpoint.config.model_identifier,
        model_revision=checkpoint.config.model_revision,
        tap_layer_indices=checkpoint.config.adapter.tap_layer_indices,
        lookahead_tokens=checkpoint.config.lookahead_tokens,
        cache_directory=arguments.model_cache_directory,
    ).to(device)
    backbone.eval()
    artifact = predict_synthetic_interactions(
        backbone=backbone,
        checkpoint=checkpoint,
        source_roots=tuple(arguments.dynamic_synthetic_corpus),
        batch_size=arguments.batch_size,
        detection_horizon_seconds=arguments.detection_horizon_seconds,
        device=device,
        skip_events=arguments.skip_events,
        maximum_events=arguments.maximum_events,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(artifact.predictions)} predictions and "
        f"{len(artifact.missing_events)} missing-event records to {arguments.output}",
        flush=True,
    )


def _analyze(arguments: argparse.Namespace) -> None:
    artifact = InteractionPredictionArtifact.model_validate_json(
        arguments.predictions.read_text(encoding="utf-8")
    )
    report = evaluate_interaction_predictions(artifact, arguments.threshold)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"backchannel_false_cancel={report.backchannel_false_cancel_rate}; "
        f"interruption_recall={report.interruption_floor_take_recall}; "
        f"coverage={report.evaluated_event_count}/{report.eligible_event_count}",
        flush=True,
    )


def _merge(arguments: argparse.Namespace) -> None:
    artifacts = tuple(
        InteractionPredictionArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        for path in arguments.predictions
    )
    merged = merge_interaction_prediction_artifacts(artifacts)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(merged.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"Merged {len(artifacts)} shards with {len(merged.predictions)} predictions and "
        f"{len(merged.missing_events)} missing-event records into {arguments.output}",
        flush=True,
    )


def _predict_baseline(arguments: argparse.Namespace) -> None:
    artifact = predict_interaction_baseline(
        baseline_kind=arguments.detector,
        source_roots=tuple(arguments.dynamic_synthetic_corpus),
        detection_horizon_seconds=arguments.detection_horizon_seconds,
        model_cache_directory=arguments.model_cache_directory,
        crop_random_seed=arguments.crop_random_seed,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(artifact.predictions)} {artifact.manifest.detector.baseline_kind.value} "
        f"interaction predictions to {arguments.output}",
        flush=True,
    )


def _analyze_baseline(arguments: argparse.Namespace) -> None:
    artifact = InteractionBaselineArtifact.model_validate_json(
        arguments.predictions.read_text(encoding="utf-8")
    )
    report = evaluate_interaction_baseline(artifact, arguments.threshold)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"backchannel_false_action={report.backchannel_false_action_rate}; "
        f"interruption_recall={report.interruption_recall}; "
        f"coverage={report.backchannel_count + report.interruption_count}",
        flush=True,
    )


def _sweep_baseline(arguments: argparse.Namespace) -> None:
    artifact = InteractionBaselineArtifact.model_validate_json(
        arguments.predictions.read_text(encoding="utf-8")
    )
    sweep = sweep_interaction_baseline(artifact, arguments.threshold_step)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(sweep.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sweep.reports)} threshold reports to {arguments.output}", flush=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


if __name__ == "__main__":
    main()
