from __future__ import annotations

import argparse
import hashlib
import sys
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Subset

from app.local.analyses.end_of_turn.detectors.livekit_v1_mini import (
    EOT_MAX_SAMPLES,
    MODEL_SAMPLE_RATE,
    load_livekit_v1_mini_inference,
)
from app.local.analyses.end_of_turn.detectors.pipecat_smart_turn_v3 import (
    MAX_MODEL_WINDOW_SECONDS,
    MODEL_FILENAME,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    load_smart_turn_v3_inference,
)
from app.local.training_corpus.export import MaterializedTrainingSample
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.backbone import NemotronStreamingBackbone
from app.training.turn_taking.benchmark_adapters import (
    CandidateAudioProvider,
    HubAudioPathResolver,
    LiveKitInferenceScorer,
    ResolvedCandidateAudioProvider,
    RootAudioPathResolver,
    SmartTurnInferenceScorer,
    livekit_candidate_predictions,
    livekit_completion_candidate_predictions,
    load_causal_silero_model,
    silero_candidate_predictions,
    silero_completion_candidate_predictions,
    smart_turn_candidate_predictions,
    smart_turn_completion_candidate_predictions,
)
from app.training.turn_taking.benchmark_completion_audit import (
    CompletionAuditConfiguration,
    CompletionAuditManifest,
    CompletionAuditReviewArtifact,
    analyze_completion_audit_reviews,
    build_completion_audit_manifest,
)
from app.training.turn_taking.benchmark_completion_audit_export import (
    ResolvedCompletionAuditAudioLoader,
    refresh_completion_audit_review_page,
    refresh_corpus_quality_audit_review_page,
    write_completion_audit_package,
    write_corpus_quality_audit_package,
)
from app.training.turn_taking.benchmark_completion_inventory import (
    build_turn_completion_inventory,
)
from app.training.turn_taking.benchmark_completion_merge import (
    merge_completion_inventories,
    merge_completion_predictions,
)
from app.training.turn_taking.benchmark_completion_metrics import (
    CompletionAnalysisReport,
    CompletionEvaluationConfiguration,
    completion_breakdowns,
    completion_calibration_metrics,
    completion_discrimination_metrics,
    completion_pareto_frontier,
    select_completion_validation_point,
    sweep_completion_policies,
    validate_completion_predictions,
)
from app.training.turn_taking.benchmark_corpus_quality_audit import (
    CorpusQualityAuditConfiguration,
    build_corpus_quality_audit_manifest,
)
from app.training.turn_taking.benchmark_gate import (
    ValidationLockManifest,
    create_validation_lock,
    evaluate_locked_test_artifact,
    file_sha256,
    read_analysis_report,
    read_validation_lock,
    write_validation_lock,
)
from app.training.turn_taking.benchmark_inventory import build_candidate_inventory
from app.training.turn_taking.benchmark_metrics import (
    EvaluationConfiguration,
    PolicySweepPoint,
    ScorePersistence,
    best_at_cutoff_budget,
    best_at_latency_budget,
    calibration_metrics,
    evaluate_breakdowns,
    pareto_frontier,
    sweep_policies,
)
from app.training.turn_taking.benchmark_models import (
    AudioProvenanceRecord,
    CompletionDetectorKind,
    CompletionDetectorProvenance,
    CompletionPredictionArtifact,
    CompletionPredictionManifest,
    DetectorKind,
    DetectorProvenance,
    LiveKitCompletionDetectorProvenance,
    LiveKitDetectorConfiguration,
    LiveKitDetectorProvenance,
    OverlapAuditReport,
    PredictionArtifact,
    PredictionManifest,
    SileroCompletionDetectorProvenance,
    SileroDetectorConfiguration,
    SileroDetectorProvenance,
    SmartTurnCompletionDetectorProvenance,
    SmartTurnDetectorConfiguration,
    SmartTurnDetectorProvenance,
    completion_prediction_rows_sha256,
    prediction_rows_sha256,
    read_completion_predictions,
    read_inventory,
    read_predictions,
    read_turn_completion_inventory,
    write_completion_predictions,
    write_inventory,
    write_predictions,
    write_turn_completion_inventory,
)
from app.training.turn_taking.benchmark_overlap import audit_smart_turn_overlap
from app.training.turn_taking.benchmark_report import (
    BenchmarkAnalysisReport,
    OperatingPoints,
)
from app.training.turn_taking.benchmark_voice_light import (
    load_voice_light_checkpoint,
    predict_voice_light_checkpoints,
    predict_voice_light_completion_checkpoints,
)
from app.training.turn_taking.data import collate_training_items
from app.training.turn_taking.hub import (
    DEFAULT_HUB_REPOSITORY,
    HuggingFaceTurnTakingDataset,
    LocalMaterializedTurnTakingDataset,
)
from app.training.turn_taking.synthetic_completion_export import (
    export_synthetic_completion_validation,
)

PINNED_CORPUS_REVISION = "56e68eb8fb1d42159483612f508b9ce27672f724"
PINNED_NEMOTRON_REVISION = "ebe59e5a817142986528bbbee5dba8db7b38ed50"
SMART_TURN_TRAINING_REPOSITORY = "pipecat-ai/smart-turn-data-v3.2-train"
PINNED_SMART_TURN_TRAINING_REVISION = "e564e2ac567f774d1880aa1db6ce97afb8c519b7"
IMPLEMENTATION_VERSION = "voice-light-causal-adapters-v1"
DEFAULT_THRESHOLDS = tuple(index / 20 for index in range(1, 20))
DEFAULT_ACTION_DELAYS_SECONDS = (0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64)
DEFAULT_TIMEOUTS_SECONDS = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0)
DEFAULT_COMPLETION_THRESHOLDS = (*DEFAULT_THRESHOLDS, 0.99, 1.0)
DEFAULT_COMPLETION_ACTION_DELAYS_SECONDS = (
    *DEFAULT_ACTION_DELAYS_SECONDS,
    0.8,
    1.0,
    1.2,
    1.6,
    2.0,
)
HASH_CHUNK_BYTES = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the causal turn-detection benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_inventory_parser(subparsers)
    _add_predict_parser(subparsers)
    _add_voice_light_parser(subparsers)
    _add_analyze_parser(subparsers)
    _add_overlap_parser(subparsers)
    _add_lock_parser(subparsers)
    _add_final_test_parser(subparsers)
    _add_completion_inventory_parser(subparsers)
    _add_completion_baseline_parser(subparsers)
    _add_completion_voice_light_parser(subparsers)
    _add_completion_analyze_parser(subparsers)
    _add_completion_merge_inventory_parser(subparsers)
    _add_completion_merge_predictions_parser(subparsers)
    _add_synthetic_completion_export_parser(subparsers)
    _add_completion_audit_parser(subparsers)
    _add_completion_audit_refresh_parser(subparsers)
    _add_completion_audit_analysis_parser(subparsers)
    _add_corpus_quality_audit_parser(subparsers)
    _add_corpus_quality_audit_refresh_parser(subparsers)
    arguments = parser.parse_args()
    match arguments.command:
        case "inventory":
            _create_inventory(arguments)
        case "predict-baseline":
            _predict_baseline(arguments)
        case "predict-voice-light":
            _predict_voice_light(arguments)
        case "analyze":
            _analyze(arguments)
        case "overlap-audit":
            _overlap_audit(arguments)
        case "lock-validation":
            _lock_validation(arguments)
        case "final-test":
            _final_test(arguments)
        case "completion-inventory-v2":
            _create_completion_inventory(arguments)
        case "predict-completion-baseline-v2":
            _predict_completion_baseline(arguments)
        case "predict-voice-light-completion-v2":
            _predict_voice_light_completion(arguments)
        case "analyze-completion-v2":
            _analyze_completion(arguments)
        case "merge-completion-inventories-v2":
            _merge_completion_inventories(arguments)
        case "merge-completion-predictions-v2":
            _merge_completion_predictions(arguments)
        case "export-synthetic-completion-validation-v1":
            _export_synthetic_completion_validation(arguments)
        case "completion-label-audit-v2":
            _completion_label_audit(arguments)
        case "refresh-completion-label-audit-ui-v2":
            _refresh_completion_label_audit_ui(arguments)
        case "analyze-completion-label-audit-v2":
            _analyze_completion_label_audit(arguments)
        case "corpus-quality-audit-v1":
            _corpus_quality_audit(arguments)
        case "refresh-corpus-quality-audit-ui-v1":
            _refresh_corpus_quality_audit_ui(arguments)
        case _:
            raise AssertionError(f"Unhandled command {arguments.command!r}.")


def _add_inventory_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("inventory", help="Build a causal candidate inventory.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--hub-repository", default=DEFAULT_HUB_REPOSITORY)
    parser.add_argument("--hub-revision", default=PINNED_CORPUS_REVISION)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--local-export-root", type=Path)
    parser.add_argument(
        "--split",
        type=TrainingCorpusSplit,
        choices=tuple(TrainingCorpusSplit),
        default="validation",
    )
    parser.add_argument("--validation-lock", type=Path)


def _add_predict_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "predict-baseline", help="Cache sparse validation baseline predictions."
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--detector",
        choices=("silero", "smart-turn", "livekit"),
        required=True,
    )
    parser.add_argument("--hub-repository", default=DEFAULT_HUB_REPOSITORY)
    parser.add_argument("--hub-revision", default=PINNED_CORPUS_REVISION)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--validation-lock", type=Path)


def _add_analyze_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("analyze", help="Sweep validation endpointing policies.")
    parser.add_argument("inventory", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-score-point-seconds", type=_positive_float, default=0.2)
    parser.add_argument("--target-yield-threshold", type=_probability, default=0.5)
    parser.add_argument("--thresholds", type=_float_tuple, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--action-delays-seconds",
        type=_float_tuple,
        default=DEFAULT_ACTION_DELAYS_SECONDS,
    )
    parser.add_argument("--timeouts-seconds", type=_float_tuple, default=DEFAULT_TIMEOUTS_SECONDS)


def _add_voice_light_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "predict-voice-light",
        help="Cache both Voice Light checkpoints in one shared Nemotron pass.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--hub-repository", default=DEFAULT_HUB_REPOSITORY)
    parser.add_argument("--hub-revision", default=PINNED_CORPUS_REVISION)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--model-revision", default=PINNED_NEMOTRON_REVISION)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--data-loader-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--validation-lock", type=Path)


def _add_overlap_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "overlap-audit",
        help="Report Smart Turn provenance and exact-hash overlap evidence.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--smart-turn-training-revision",
        default=PINNED_SMART_TURN_TRAINING_REVISION,
    )


def _add_lock_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "lock-validation",
        help="Freeze validation-selected policies before opening the test split.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("overlap_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--primary-checkpoint-sha256", required=True)


def _add_final_test_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "final-test",
        help="Evaluate one locked policy on one matching test prediction artifact.",
    )
    parser.add_argument("validation_lock", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)


def _add_completion_inventory_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "completion-inventory-v2",
        help="Build the validation-only turn-completion candidate inventory.",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--hub-repository", default=DEFAULT_HUB_REPOSITORY)
    parser.add_argument("--hub-revision", default=PINNED_CORPUS_REVISION)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--local-export-root", type=Path)


def _add_completion_baseline_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "predict-completion-baseline-v2",
        help="Cache a baseline against v2 completion boundaries.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--detector", choices=("silero", "smart-turn", "livekit"), required=True)
    parser.add_argument("--hub-repository", default=DEFAULT_HUB_REPOSITORY)
    parser.add_argument("--hub-revision", default=PINNED_CORPUS_REVISION)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--audio-root", type=Path)


def _add_completion_voice_light_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "predict-voice-light-completion-v2",
        help="Cache completion-head scores from multiple checkpoints in one Nemotron pass.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--hub-repository", default=DEFAULT_HUB_REPOSITORY)
    parser.add_argument("--hub-revision", default=PINNED_CORPUS_REVISION)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--local-export-root", type=Path)
    parser.add_argument("--model-revision", default=PINNED_NEMOTRON_REVISION)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--data-loader-workers", type=_nonnegative_int, default=0)


def _add_completion_analyze_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "analyze-completion-v2",
        help="Sweep validation policies using confident completion label bands.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--thresholds", type=_float_tuple, default=DEFAULT_COMPLETION_THRESHOLDS)
    parser.add_argument(
        "--action-delays-seconds",
        type=_float_tuple,
        default=DEFAULT_COMPLETION_ACTION_DELAYS_SECONDS,
    )
    parser.add_argument("--timeouts-seconds", type=_float_tuple, default=DEFAULT_TIMEOUTS_SECONDS)


def _add_completion_merge_inventory_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "merge-completion-inventories-v2",
        help="Merge compatible validation completion inventories.",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("inventories", type=Path, nargs="+")


def _add_completion_merge_predictions_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "merge-completion-predictions-v2",
        help="Merge compatible completion predictions for a merged inventory.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("predictions", type=Path, nargs="+")


def _add_synthetic_completion_export_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "export-synthetic-completion-validation-v1",
        help="Materialize every deterministic synthetic EOT/HOLD validation anchor.",
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--random-seed", type=int, required=True)


def _add_completion_audit_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "completion-label-audit-v2",
        help="Build a deterministic validation label-audit package with stereo review clips.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("voice_light_predictions", type=Path)
    parser.add_argument("smart_turn_predictions", type=Path)
    parser.add_argument("livekit_predictions", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--local-export-root", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--ambiguous-count", type=_nonnegative_int, default=80)
    parser.add_argument("--confident-hold-count", type=_nonnegative_int, default=120)
    parser.add_argument("--confident-eot-count", type=_nonnegative_int, default=120)
    parser.add_argument("--double-review-count", type=_nonnegative_int, default=100)


def _add_completion_audit_analysis_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "analyze-completion-label-audit-v2",
        help="Analyze one or more exported completion-label review files.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reviews", type=Path, nargs="+")


def _add_completion_audit_refresh_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "refresh-completion-label-audit-ui-v2",
        help="Rebuild the visual review page from an existing audit package.",
    )
    parser.add_argument("audit_directory", type=Path)


def _add_corpus_quality_audit_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "corpus-quality-audit-v1",
        help="Build a source-stratified corpus-quality control review package.",
    )
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--hub-cache-directory", type=Path)
    parser.add_argument("--local-export-root", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--items-per-dataset", type=_positive_int, default=5)


def _add_corpus_quality_audit_refresh_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "refresh-corpus-quality-audit-ui-v1",
        help="Rebuild the corpus-quality review page from an existing audit package.",
    )
    parser.add_argument("audit_directory", type=Path)


def _create_inventory(arguments: argparse.Namespace) -> None:
    split: TrainingCorpusSplit = arguments.split
    _validate_split_access(
        split=split,
        validation_lock_path=arguments.validation_lock,
        corpus_repository=arguments.hub_repository,
        corpus_revision=arguments.hub_revision,
    )
    if arguments.local_export_root is None:
        dataset = HuggingFaceTurnTakingDataset(
            split=split,
            revision=arguments.hub_revision,
            repository_id=arguments.hub_repository,
            cache_directory=arguments.hub_cache_directory,
        )
        samples = dataset.samples
    else:
        samples = _read_local_samples(arguments.local_export_root, split)
    inventory = build_candidate_inventory(
        samples=samples,
        corpus_repository=arguments.hub_repository,
        corpus_revision=arguments.hub_revision,
        split=split,
    )
    write_inventory(path=arguments.output, inventory=inventory)
    print(inventory.manifest.model_dump_json(indent=2), flush=True)


def _predict_baseline(arguments: argparse.Namespace) -> None:
    inventory = read_inventory(arguments.inventory)
    lock = _validation_lock_for_predictions(inventory.manifest.split, arguments.validation_lock)
    audio_provider = _audio_provider(arguments)
    match arguments.detector:
        case "silero":
            model = load_causal_silero_model(use_onnx=True)
            predictions = silero_candidate_predictions(
                candidates=inventory.candidates,
                audio_provider=audio_provider,
                model=model,
            )
            provenance: DetectorProvenance = SileroDetectorProvenance(
                display_name="Silero VAD 6.2.1",
                implementation_version=IMPLEMENTATION_VERSION,
                package_name="silero-vad",
                package_version=version("silero-vad"),
                configuration=SileroDetectorConfiguration(
                    speech_threshold=0.5,
                    minimum_speech_seconds=0.1,
                    minimum_silence_seconds=SILERO_FRAME_SECONDS,
                    use_onnx=True,
                ),
            )
        case "smart-turn":
            inference = load_smart_turn_v3_inference(
                model_repository=MODEL_REPOSITORY,
                model_revision=MODEL_REVISION,
                model_filename=MODEL_FILENAME,
                cache_directory=arguments.hub_cache_directory,
            )
            predictions = smart_turn_candidate_predictions(
                candidates=inventory.candidates,
                audio_provider=audio_provider,
                scorer=SmartTurnInferenceScorer(inference),
                candidate_silence_seconds=0.2,
                maximum_window_seconds=MAX_MODEL_WINDOW_SECONDS,
            )
            model_path = Path(
                hf_hub_download(
                    repo_id=MODEL_REPOSITORY,
                    filename=MODEL_FILENAME,
                    revision=MODEL_REVISION,
                    cache_dir=arguments.hub_cache_directory,
                )
            )
            provenance = SmartTurnDetectorProvenance(
                display_name="Pipecat Smart Turn v3.2 CPU ONNX",
                implementation_version=IMPLEMENTATION_VERSION,
                runtime_package_name="onnxruntime",
                runtime_package_version=version("onnxruntime"),
                model_repository=MODEL_REPOSITORY,
                model_revision=MODEL_REVISION,
                model_filename=MODEL_FILENAME,
                model_sha256=_file_sha256(model_path),
                configuration=SmartTurnDetectorConfiguration(
                    sample_rate_hz=MODEL_SAMPLE_RATE,
                    maximum_window_seconds=MAX_MODEL_WINDOW_SECONDS,
                    candidate_silence_seconds=0.2,
                ),
            )
        case "livekit":
            inference = load_livekit_v1_mini_inference()
            predictions = livekit_candidate_predictions(
                candidates=inventory.candidates,
                audio_provider=audio_provider,
                scorer=LiveKitInferenceScorer(inference.end_of_turn_model),
                candidate_silence_seconds=0.3,
                maximum_window_seconds=EOT_MAX_SAMPLES / MODEL_SAMPLE_RATE,
            )
            provenance = LiveKitDetectorProvenance(
                display_name="LiveKit Turn Detector v1-mini",
                implementation_version=IMPLEMENTATION_VERSION,
                package_name="livekit-local-inference",
                package_version=version("livekit-local-inference"),
                model_sha256=None,
                configuration=LiveKitDetectorConfiguration(
                    sample_rate_hz=MODEL_SAMPLE_RATE,
                    vad_speech_threshold=0.5,
                    candidate_silence_seconds=0.3,
                ),
            )
        case _:
            raise AssertionError(f"Unhandled detector {arguments.detector!r}.")
    if lock is not None and not any(policy.detector == provenance for policy in lock.policies):
        raise ValueError("Baseline detector provenance is absent from the validation lock.")
    ordered_predictions = tuple(
        sorted(
            predictions,
            key=lambda prediction: (
                prediction.candidate_id,
                prediction.absolute_time_seconds,
            ),
        )
    )
    artifact = PredictionArtifact(
        manifest=PredictionManifest(
            inventory_sha256=inventory.manifest.candidate_sha256,
            split=inventory.manifest.split,
            detector=provenance,
            prediction_count=len(ordered_predictions),
            predictions_sha256=prediction_rows_sha256(ordered_predictions),
        ),
        predictions=ordered_predictions,
    )
    write_predictions(path=arguments.output, artifact=artifact)
    print(artifact.manifest.model_dump_json(indent=2), flush=True)


def _predict_voice_light(arguments: argparse.Namespace) -> None:
    inventory = read_inventory(arguments.inventory)
    lock = _validation_lock_for_predictions(inventory.manifest.split, arguments.validation_lock)
    checkpoints = tuple(load_voice_light_checkpoint(path) for path in arguments.checkpoints)
    reference_config = checkpoints[0].config
    dataset = HuggingFaceTurnTakingDataset(
        split=inventory.manifest.split,
        revision=arguments.hub_revision,
        repository_id=arguments.hub_repository,
        cache_directory=arguments.hub_cache_directory,
        sample_rate_hz=reference_config.sample_rate_hz,
        pad_missing_audio_suffix=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_training_items,
        num_workers=arguments.data_loader_workers,
        prefetch_factor=2 if arguments.data_loader_workers > 0 else None,
        persistent_workers=arguments.data_loader_workers > 0,
        pin_memory=device.type == "cuda",
    )
    backbone = NemotronStreamingBackbone(
        model_identifier=reference_config.model_identifier,
        tap_layer_indices=reference_config.adapter.tap_layer_indices,
        lookahead_tokens=reference_config.lookahead_tokens,
        model_revision=arguments.model_revision,
        cache_directory=arguments.hub_cache_directory,
    ).to(device)
    backbone.eval()
    artifacts = predict_voice_light_checkpoints(
        backbone=backbone,
        checkpoints=checkpoints,
        batches=loader,
        samples=dataset.samples,
        inventory=inventory,
        model_repository=reference_config.model_identifier,
        model_revision=arguments.model_revision,
        device=device,
        total_batch_count=len(loader),
        progress_output=sys.stderr,
    )
    for checkpoint, artifact in zip(checkpoints, artifacts, strict=True):
        if lock is not None and not any(
            policy.detector == artifact.manifest.detector for policy in lock.policies
        ):
            raise ValueError("Voice Light detector provenance is absent from the validation lock.")
        output_path = arguments.output_directory / (
            f"{inventory.manifest.split.value}-voice-light-step-"
            f"{checkpoint.optimizer_step:06d}-predictions.json"
        )
        write_predictions(path=output_path, artifact=artifact)
        print(artifact.manifest.model_dump_json(indent=2), flush=True)


def _analyze(arguments: argparse.Namespace) -> None:
    inventory = read_inventory(arguments.inventory)
    artifact = read_predictions(arguments.predictions)
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("Policy tuning is restricted to validation.")
    if artifact.manifest.inventory_sha256 != inventory.manifest.candidate_sha256:
        raise ValueError("Prediction artifact does not match the candidate inventory.")
    evaluation = EvaluationConfiguration(
        target_score_point_seconds=arguments.target_score_point_seconds,
        target_yield_threshold=arguments.target_yield_threshold,
        score_persistence=_score_persistence(artifact.manifest.detector.detector_kind),
    )
    sweep = sweep_policies(
        candidates=inventory.candidates,
        predictions=artifact.predictions,
        thresholds=arguments.thresholds,
        action_delays_seconds=arguments.action_delays_seconds,
        timeouts_seconds=arguments.timeouts_seconds,
        evaluation=evaluation,
    )
    selected = _select_validation_point(sweep)
    calibration = (
        None
        if artifact.manifest.detector.detector_kind is DetectorKind.SILERO_TIMEOUT
        else calibration_metrics(
            candidates=inventory.candidates,
            predictions=artifact.predictions,
        )
    )
    report = BenchmarkAnalysisReport(
        inventory_sha256=inventory.manifest.candidate_sha256,
        predictions_sha256=artifact.manifest.predictions_sha256,
        detector=artifact.manifest.detector,
        evaluation=evaluation,
        calibration=calibration,
        sweep=sweep,
        pareto_frontier=pareto_frontier(sweep),
        operating_points=OperatingPoints(
            best_at_300ms_latency=best_at_latency_budget(sweep, 0.3),
            best_at_600ms_latency=best_at_latency_budget(sweep, 0.6),
            best_at_5_percent_cutoff=best_at_cutoff_budget(sweep, 0.05),
            best_at_10_percent_cutoff=best_at_cutoff_budget(sweep, 0.10),
        ),
        selected_validation_point=selected,
        selected_breakdowns=evaluate_breakdowns(
            candidates=inventory.candidates,
            predictions=artifact.predictions,
            policy=selected.policy,
            evaluation=evaluation,
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(selected.model_dump_json(indent=2), flush=True)


def _create_completion_inventory(arguments: argparse.Namespace) -> None:
    split = TrainingCorpusSplit.VALIDATION
    if arguments.local_export_root is None:
        samples = HuggingFaceTurnTakingDataset(
            split=split,
            revision=arguments.hub_revision,
            repository_id=arguments.hub_repository,
            cache_directory=arguments.hub_cache_directory,
        ).samples
    else:
        samples = _read_local_samples(arguments.local_export_root, split)
    inventory = build_turn_completion_inventory(
        samples=samples,
        corpus_repository=arguments.hub_repository,
        corpus_revision=arguments.hub_revision,
        split=split,
    )
    write_turn_completion_inventory(arguments.output, inventory)
    print(inventory.manifest.model_dump_json(indent=2), flush=True)


def _predict_completion_baseline(arguments: argparse.Namespace) -> None:
    inventory = read_turn_completion_inventory(arguments.inventory)
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("Turn-completion prediction is restricted to validation.")
    audio_provider = _audio_provider(arguments)
    match arguments.detector:
        case "silero":
            predictions = silero_completion_candidate_predictions(
                inventory.candidates,
                audio_provider,
                load_causal_silero_model(use_onnx=True),
            )
            provenance: CompletionDetectorProvenance = SileroCompletionDetectorProvenance(
                display_name="Silero VAD 6.2.1 silence proxy",
                implementation_version="voice-light-turn-completion-adapters-v2",
                package_name="silero-vad",
                package_version=version("silero-vad"),
                configuration=SileroDetectorConfiguration(
                    speech_threshold=0.5,
                    minimum_speech_seconds=0.1,
                    minimum_silence_seconds=SILERO_FRAME_SECONDS,
                    use_onnx=True,
                ),
            )
        case "smart-turn":
            inference = load_smart_turn_v3_inference(
                model_repository=MODEL_REPOSITORY,
                model_revision=MODEL_REVISION,
                model_filename=MODEL_FILENAME,
                cache_directory=arguments.hub_cache_directory,
            )
            predictions = smart_turn_completion_candidate_predictions(
                inventory.candidates,
                audio_provider,
                SmartTurnInferenceScorer(inference),
                candidate_silence_seconds=0.2,
                maximum_window_seconds=MAX_MODEL_WINDOW_SECONDS,
            )
            model_path = Path(
                hf_hub_download(
                    repo_id=MODEL_REPOSITORY,
                    filename=MODEL_FILENAME,
                    revision=MODEL_REVISION,
                    cache_dir=arguments.hub_cache_directory,
                )
            )
            provenance = SmartTurnCompletionDetectorProvenance(
                display_name="Pipecat Smart Turn v3.2 CPU ONNX completion",
                implementation_version="voice-light-turn-completion-adapters-v2",
                runtime_package_name="onnxruntime",
                runtime_package_version=version("onnxruntime"),
                model_repository=MODEL_REPOSITORY,
                model_revision=MODEL_REVISION,
                model_filename=MODEL_FILENAME,
                model_sha256=_file_sha256(model_path),
                configuration=SmartTurnDetectorConfiguration(
                    sample_rate_hz=MODEL_SAMPLE_RATE,
                    maximum_window_seconds=MAX_MODEL_WINDOW_SECONDS,
                    candidate_silence_seconds=0.2,
                ),
            )
        case "livekit":
            inference = load_livekit_v1_mini_inference()
            predictions = livekit_completion_candidate_predictions(
                inventory.candidates,
                audio_provider,
                LiveKitInferenceScorer(inference.end_of_turn_model),
                candidate_silence_seconds=0.3,
                maximum_window_seconds=EOT_MAX_SAMPLES / MODEL_SAMPLE_RATE,
            )
            provenance = LiveKitCompletionDetectorProvenance(
                display_name="LiveKit Turn Detector v1-mini completion",
                implementation_version="voice-light-turn-completion-adapters-v2",
                package_name="livekit-local-inference",
                package_version=version("livekit-local-inference"),
                model_sha256=None,
                configuration=LiveKitDetectorConfiguration(
                    sample_rate_hz=MODEL_SAMPLE_RATE,
                    vad_speech_threshold=0.5,
                    candidate_silence_seconds=0.3,
                ),
            )
        case _:
            raise AssertionError(f"Unhandled detector {arguments.detector!r}.")
    ordered = tuple(
        sorted(
            predictions,
            key=lambda prediction: (
                prediction.candidate_id,
                prediction.absolute_time_seconds,
            ),
        )
    )
    validate_completion_predictions(inventory.candidates, ordered)
    artifact = CompletionPredictionArtifact(
        manifest=CompletionPredictionManifest(
            inventory_sha256=inventory.manifest.candidate_sha256,
            split=inventory.manifest.split,
            detector=provenance,
            prediction_count=len(ordered),
            predictions_sha256=completion_prediction_rows_sha256(ordered),
        ),
        predictions=ordered,
    )
    write_completion_predictions(arguments.output, artifact)
    print(artifact.manifest.model_dump_json(indent=2), flush=True)


def _predict_voice_light_completion(arguments: argparse.Namespace) -> None:
    inventory = read_turn_completion_inventory(arguments.inventory)
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("Turn-completion prediction is restricted to validation.")
    checkpoints = tuple(load_voice_light_checkpoint(path) for path in arguments.checkpoints)
    reference_config = checkpoints[0].config
    dataset = (
        LocalMaterializedTurnTakingDataset(
            root=arguments.local_export_root,
            split=TrainingCorpusSplit.VALIDATION,
            sample_rate_hz=reference_config.sample_rate_hz,
            pad_missing_audio_suffix=True,
        )
        if arguments.local_export_root is not None
        else HuggingFaceTurnTakingDataset(
            split=TrainingCorpusSplit.VALIDATION,
            revision=arguments.hub_revision,
            repository_id=arguments.hub_repository,
            cache_directory=arguments.hub_cache_directory,
            sample_rate_hz=reference_config.sample_rate_hz,
            pad_missing_audio_suffix=True,
        )
    )
    relevant_window_ids = {
        window_id for candidate in inventory.candidates for window_id in candidate.source_window_ids
    }
    relevant_indices = tuple(
        index
        for index, sample in enumerate(dataset.samples)
        if sample.window_id in relevant_window_ids
    )
    loaded_window_ids = {dataset.samples[index].window_id for index in relevant_indices}
    missing_window_ids = relevant_window_ids - loaded_window_ids
    if missing_window_ids:
        raise ValueError("Completion inventory references windows absent from the corpus split.")
    relevant_dataset = Subset(dataset, relevant_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        relevant_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        collate_fn=collate_training_items,
        num_workers=arguments.data_loader_workers,
        prefetch_factor=2 if arguments.data_loader_workers > 0 else None,
        persistent_workers=arguments.data_loader_workers > 0,
        pin_memory=device.type == "cuda",
    )
    backbone = NemotronStreamingBackbone(
        model_identifier=reference_config.model_identifier,
        tap_layer_indices=reference_config.adapter.tap_layer_indices,
        lookahead_tokens=reference_config.lookahead_tokens,
        model_revision=arguments.model_revision,
        cache_directory=arguments.hub_cache_directory,
    ).to(device)
    backbone.eval()
    artifacts = predict_voice_light_completion_checkpoints(
        backbone=backbone,
        checkpoints=checkpoints,
        batches=loader,
        samples=dataset.samples,
        inventory=inventory,
        model_repository=reference_config.model_identifier,
        model_revision=arguments.model_revision,
        device=device,
        total_batch_count=len(loader),
        progress_output=sys.stderr,
    )
    for checkpoint, artifact in zip(checkpoints, artifacts, strict=True):
        validate_completion_predictions(inventory.candidates, artifact.predictions)
        output_path = arguments.output_directory / (
            "validation-v2-turn-completion-voice-light-step-"
            f"{checkpoint.optimizer_step:06d}-predictions.json"
        )
        write_completion_predictions(output_path, artifact)
        print(artifact.manifest.model_dump_json(indent=2), flush=True)


def _analyze_completion(arguments: argparse.Namespace) -> None:
    inventory = read_turn_completion_inventory(arguments.inventory)
    artifact = read_completion_predictions(arguments.predictions)
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("Turn-completion analysis is restricted to validation.")
    if artifact.manifest.split is not inventory.manifest.split:
        raise ValueError("Prediction split does not match the completion inventory split.")
    if artifact.manifest.inventory_sha256 != inventory.manifest.candidate_sha256:
        raise ValueError("Prediction artifact does not match the completion inventory.")
    persistence = (
        ScorePersistence.CURRENT
        if artifact.manifest.detector.detector_kind is CompletionDetectorKind.SILERO_SILENCE
        else ScorePersistence.LATCHED
    )
    evaluation = CompletionEvaluationConfiguration(
        hold_completion_maximum=0.2,
        hold_continuation_minimum=0.8,
        eot_completion_minimum=0.8,
        conflicting_continuation_minimum=0.8,
        score_persistence=persistence,
    )
    sweep = sweep_completion_policies(
        inventory.candidates,
        artifact.predictions,
        arguments.thresholds,
        arguments.action_delays_seconds,
        arguments.timeouts_seconds,
        evaluation,
    )
    selected = select_completion_validation_point(sweep)
    coverage = validate_completion_predictions(inventory.candidates, artifact.predictions)
    calibration = (
        None
        if artifact.manifest.detector.detector_kind is CompletionDetectorKind.SILERO_SILENCE
        else completion_calibration_metrics(inventory.candidates, artifact.predictions)
    )
    report = CompletionAnalysisReport(
        inventory_sha256=inventory.manifest.candidate_sha256,
        predictions_sha256=artifact.manifest.predictions_sha256,
        detector=artifact.manifest.detector,
        evaluation=evaluation,
        prediction_coverage=coverage,
        discrimination=completion_discrimination_metrics(
            inventory.candidates,
            artifact.predictions,
            evaluation,
        ),
        calibration=calibration,
        sweep=sweep,
        pareto_frontier=completion_pareto_frontier(sweep),
        selected_validation_point=selected,
        selected_breakdowns=completion_breakdowns(
            inventory.candidates,
            artifact.predictions,
            selected.policy,
            evaluation,
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(selected.model_dump_json(indent=2), flush=True)


def _merge_completion_inventories(arguments: argparse.Namespace) -> None:
    inventory = merge_completion_inventories(
        tuple(read_turn_completion_inventory(path) for path in arguments.inventories)
    )
    write_turn_completion_inventory(arguments.output, inventory)
    print(inventory.manifest.model_dump_json(indent=2), flush=True)


def _merge_completion_predictions(arguments: argparse.Namespace) -> None:
    artifact = merge_completion_predictions(
        inventory=read_turn_completion_inventory(arguments.inventory),
        artifacts=tuple(read_completion_predictions(path) for path in arguments.predictions),
    )
    write_completion_predictions(arguments.output, artifact)
    print(artifact.manifest.model_dump_json(indent=2), flush=True)


def _export_synthetic_completion_validation(arguments: argparse.Namespace) -> None:
    manifest = export_synthetic_completion_validation(
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        random_seed=arguments.random_seed,
    )
    print(manifest.model_dump_json(indent=2), flush=True)


def _completion_label_audit(arguments: argparse.Namespace) -> None:
    inventory = read_turn_completion_inventory(arguments.inventory)
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("Completion label audits are restricted to validation.")
    if arguments.local_export_root is None:
        samples = HuggingFaceTurnTakingDataset(
            split=TrainingCorpusSplit.VALIDATION,
            revision=inventory.manifest.corpus_revision,
            repository_id=inventory.manifest.corpus_repository,
            cache_directory=arguments.hub_cache_directory,
        ).samples
    else:
        samples = _read_local_samples(
            arguments.local_export_root,
            TrainingCorpusSplit.VALIDATION,
        )
    manifest = build_completion_audit_manifest(
        inventory=inventory,
        voice_light_predictions=read_completion_predictions(arguments.voice_light_predictions),
        smart_turn_predictions=read_completion_predictions(arguments.smart_turn_predictions),
        livekit_predictions=read_completion_predictions(arguments.livekit_predictions),
        configuration=CompletionAuditConfiguration(
            ambiguous_count=arguments.ambiguous_count,
            confident_hold_count=arguments.confident_hold_count,
            confident_eot_count=arguments.confident_eot_count,
            double_review_count=arguments.double_review_count,
        ),
    )
    resolver = (
        RootAudioPathResolver(arguments.audio_root)
        if arguments.audio_root is not None
        else HubAudioPathResolver(
            repository_id=inventory.manifest.corpus_repository,
            revision=inventory.manifest.corpus_revision,
            cache_directory=arguments.hub_cache_directory,
        )
    )
    write_completion_audit_package(
        output_directory=arguments.output_directory,
        manifest=manifest,
        inventory=inventory,
        samples=samples,
        audio_loader=ResolvedCompletionAuditAudioLoader(path_resolver=resolver),
        progress_output=sys.stderr,
    )
    print(
        f"Wrote {manifest.item_count} completion-label audit cases to {arguments.output_directory}",
        flush=True,
    )


def _refresh_completion_label_audit_ui(arguments: argparse.Namespace) -> None:
    refresh_completion_audit_review_page(arguments.audit_directory)
    print(f"Refreshed completion-label reviewer at {arguments.audit_directory}", flush=True)


def _corpus_quality_audit(arguments: argparse.Namespace) -> None:
    inventory = read_turn_completion_inventory(arguments.inventory)
    if inventory.manifest.split is not TrainingCorpusSplit.VALIDATION:
        raise ValueError("Corpus quality audits are restricted to validation.")
    if arguments.local_export_root is None:
        samples = HuggingFaceTurnTakingDataset(
            split=TrainingCorpusSplit.VALIDATION,
            revision=inventory.manifest.corpus_revision,
            repository_id=inventory.manifest.corpus_repository,
            cache_directory=arguments.hub_cache_directory,
        ).samples
    else:
        samples = _read_local_samples(
            arguments.local_export_root,
            TrainingCorpusSplit.VALIDATION,
        )
    manifest = build_corpus_quality_audit_manifest(
        inventory=inventory,
        configuration=CorpusQualityAuditConfiguration(
            items_per_dataset=arguments.items_per_dataset,
        ),
    )
    resolver = (
        RootAudioPathResolver(arguments.audio_root)
        if arguments.audio_root is not None
        else HubAudioPathResolver(
            repository_id=inventory.manifest.corpus_repository,
            revision=inventory.manifest.corpus_revision,
            cache_directory=arguments.hub_cache_directory,
        )
    )
    write_corpus_quality_audit_package(
        output_directory=arguments.output_directory,
        manifest=manifest,
        inventory=inventory,
        samples=samples,
        audio_loader=ResolvedCompletionAuditAudioLoader(path_resolver=resolver),
        progress_output=sys.stderr,
    )
    print(
        f"Wrote {manifest.item_count} corpus-quality audit cases to {arguments.output_directory}",
        flush=True,
    )


def _refresh_corpus_quality_audit_ui(arguments: argparse.Namespace) -> None:
    refresh_corpus_quality_audit_review_page(arguments.audit_directory)
    print(f"Refreshed corpus-quality reviewer at {arguments.audit_directory}", flush=True)


def _analyze_completion_label_audit(arguments: argparse.Namespace) -> None:
    manifest = CompletionAuditManifest.model_validate_json(
        arguments.manifest.read_text(encoding="utf-8")
    )
    reviews = tuple(
        CompletionAuditReviewArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        for path in arguments.reviews
    )
    report = analyze_completion_audit_reviews(manifest, reviews)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.agreement.model_dump_json(indent=2), flush=True)


def _overlap_audit(arguments: argparse.Namespace) -> None:
    inventory = read_inventory(arguments.inventory)
    unique_local_records = {
        (candidate.dataset_name, candidate.external_id) for candidate in inventory.candidates
    }
    local_records = tuple(
        AudioProvenanceRecord(
            source_name=dataset_name,
            external_id=external_id,
            audio_sha256=None,
            pcm_sha256=None,
        )
        for dataset_name, external_id in sorted(unique_local_records)
    )
    smart_turn_records = tuple(
        AudioProvenanceRecord(
            source_name=source_name,
            external_id=None,
            audio_sha256=None,
            pcm_sha256=None,
        )
        for source_name in ("Liva AI", "Midcentury", "MundoAI", "Pipecat")
    )
    report = audit_smart_turn_overlap(
        local_records=local_records,
        smart_turn_records=smart_turn_records,
        external_repository=SMART_TURN_TRAINING_REPOSITORY,
        external_revision=arguments.smart_turn_training_revision,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2), flush=True)


def _lock_validation(arguments: argparse.Namespace) -> None:
    inventory = read_inventory(arguments.inventory)
    reports = tuple(read_analysis_report(path) for path in arguments.reports)
    overlap_report = OverlapAuditReport.model_validate_json(
        arguments.overlap_audit.read_text(encoding="utf-8")
    )
    lock = create_validation_lock(
        inventory=inventory,
        reports=reports,
        primary_checkpoint_sha256=arguments.primary_checkpoint_sha256,
        overlap_report=overlap_report,
        overlap_report_sha256=file_sha256(arguments.overlap_audit),
    )
    write_validation_lock(arguments.output, lock)
    print(lock.model_dump_json(indent=2), flush=True)


def _final_test(arguments: argparse.Namespace) -> None:
    lock = read_validation_lock(arguments.validation_lock)
    report = evaluate_locked_test_artifact(
        lock=lock,
        inventory=read_inventory(arguments.inventory),
        artifact=read_predictions(arguments.predictions),
        lock_sha256=file_sha256(arguments.validation_lock),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2), flush=True)


def _read_local_samples(
    export_root: Path, split: TrainingCorpusSplit
) -> tuple[MaterializedTrainingSample, ...]:
    shard_paths = tuple(sorted((export_root / "training" / split.value).glob("*.parquet")))
    if not shard_paths:
        raise ValueError(f"No local {split.value} Parquet shards found under {export_root}.")
    table = pa.concat_tables(tuple(pq.read_table(path) for path in shard_paths))
    return tuple(MaterializedTrainingSample.model_validate(row) for row in table.to_pylist())


def _validate_split_access(
    split: TrainingCorpusSplit,
    validation_lock_path: Path | None,
    corpus_repository: str,
    corpus_revision: str,
) -> None:
    if split is not TrainingCorpusSplit.TEST:
        return
    if validation_lock_path is None:
        raise ValueError("Test inventory access requires --validation-lock.")
    lock = read_validation_lock(validation_lock_path)
    if lock.corpus_repository != corpus_repository or lock.corpus_revision != corpus_revision:
        raise ValueError("Validation lock does not match the requested corpus revision.")


def _validation_lock_for_predictions(
    split: TrainingCorpusSplit, validation_lock_path: Path | None
) -> ValidationLockManifest | None:
    if split is TrainingCorpusSplit.VALIDATION:
        return None
    if split is not TrainingCorpusSplit.TEST:
        raise ValueError("Benchmark predictions support only validation or locked test data.")
    if validation_lock_path is None:
        raise ValueError("Test prediction generation requires --validation-lock.")
    return read_validation_lock(validation_lock_path)


def _audio_provider(arguments: argparse.Namespace) -> CandidateAudioProvider:
    resolver = (
        RootAudioPathResolver(arguments.audio_root)
        if arguments.audio_root is not None
        else HubAudioPathResolver(
            repository_id=arguments.hub_repository,
            revision=arguments.hub_revision,
            cache_directory=arguments.hub_cache_directory,
        )
    )
    return ResolvedCandidateAudioProvider(path_resolver=resolver)


def _score_persistence(detector_kind: DetectorKind) -> ScorePersistence:
    match detector_kind:
        case DetectorKind.PIPECAT_SMART_TURN_V3_2 | DetectorKind.LIVEKIT_V1_MINI:
            return ScorePersistence.LATCHED
        case DetectorKind.VOICE_LIGHT | DetectorKind.SILERO_TIMEOUT:
            return ScorePersistence.CURRENT


def _select_validation_point(sweep: tuple[PolicySweepPoint, ...]) -> PolicySweepPoint:
    selected = best_at_cutoff_budget(sweep, 0.05)
    if selected is None:
        selected = best_at_cutoff_budget(sweep, 0.10)
    if selected is None:
        frontier = pareto_frontier(sweep)
        if not frontier:
            raise ValueError("Policy sweep did not produce an eligible operating point.")
        selected = min(
            frontier,
            key=lambda point: (
                point.metrics.false_cutoff_rate,
                point.metrics.mean_latency_seconds or float("inf"),
            ),
        )
    return selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative floats")
    return values


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


SILERO_FRAME_SECONDS = 512 / MODEL_SAMPLE_RATE


if __name__ == "__main__":
    main()
