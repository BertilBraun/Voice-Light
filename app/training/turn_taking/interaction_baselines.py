from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from livekit.local_inference import VAD_WINDOW_SAMPLES
from numpy.typing import NDArray
from pydantic import Field, model_validator

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
    smart_turn_completion_probability,
)
from app.local.synthetic_generation.conversation_compiler import (
    FRAME_SECONDS,
    BackchannelAnchor,
    ConversationCompilerConfig,
    InterruptionAnchor,
)
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.shared.base_model import FrozenBaseModel
from app.training.turn_taking.benchmark_adapters import load_causal_silero_model
from app.training.turn_taking.config import SyntheticAnchorSamplingConfig
from app.training.turn_taking.interaction_evaluation import (
    InteractionKind,
    LatencyMetrics,
)
from app.training.turn_taking.synthetic_dataset import (
    AnchoredSyntheticTurnTakingDataset,
)

SILERO_WINDOW_SAMPLES = 512
SILERO_CONTEXT_SECONDS = 1.0
SILERO_MINIMUM_SPEECH_SECONDS = 0.1
SMART_TURN_GATE_SECONDS = 0.2
LIVEKIT_GATE_SECONDS = 0.3
IMPLEMENTATION_VERSION = "voice-light-interaction-baselines-v1"


class InteractionBaselineKind(StrEnum):
    SILERO_SPEECH_CANCEL = "silero_speech_cancel"
    LIVEKIT_VAD_CANCEL = "livekit_vad_cancel"
    SMART_TURN_RESPONSE = "smart_turn_completion_as_response"
    LIVEKIT_RESPONSE = "livekit_completion_as_response"


class InteractionActionSemantic(StrEnum):
    CANCEL_ASSISTANT = "cancel_assistant"
    START_RESPONSE = "start_response"


class SileroInteractionProvenance(FrozenBaseModel):
    baseline_kind: Literal[InteractionBaselineKind.SILERO_SPEECH_CANCEL] = (
        InteractionBaselineKind.SILERO_SPEECH_CANCEL
    )
    display_name: str = "Silero VAD 6.2.1 immediate cancellation"
    implementation_version: str = IMPLEMENTATION_VERSION
    package_version: str
    sample_rate_hz: Literal[16000] = 16000
    frame_seconds: Literal[0.032] = 0.032
    context_seconds: Literal[1.0] = SILERO_CONTEXT_SECONDS
    minimum_action_seconds: Literal[0.1] = SILERO_MINIMUM_SPEECH_SECONDS
    action_semantic: Literal[InteractionActionSemantic.CANCEL_ASSISTANT] = (
        InteractionActionSemantic.CANCEL_ASSISTANT
    )


class SmartTurnInteractionProvenance(FrozenBaseModel):
    baseline_kind: Literal[InteractionBaselineKind.SMART_TURN_RESPONSE] = (
        InteractionBaselineKind.SMART_TURN_RESPONSE
    )
    display_name: str = "Pipecat Smart Turn v3.2 completed utterance as response"
    implementation_version: str = IMPLEMENTATION_VERSION
    runtime_version: str
    model_repository: str = MODEL_REPOSITORY
    model_revision: str = MODEL_REVISION
    model_filename: str = MODEL_FILENAME
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: Literal[16000] = 16000
    maximum_window_seconds: Literal[8.0] = MAX_MODEL_WINDOW_SECONDS
    gate_seconds: Literal[0.2] = SMART_TURN_GATE_SECONDS
    minimum_action_seconds: Literal[0.0] = 0.0
    action_semantic: Literal[InteractionActionSemantic.START_RESPONSE] = (
        InteractionActionSemantic.START_RESPONSE
    )


class LiveKitVadInteractionProvenance(FrozenBaseModel):
    baseline_kind: Literal[InteractionBaselineKind.LIVEKIT_VAD_CANCEL] = (
        InteractionBaselineKind.LIVEKIT_VAD_CANCEL
    )
    display_name: str = "LiveKit v1-mini VAD immediate cancellation"
    implementation_version: str = IMPLEMENTATION_VERSION
    package_version: str
    sample_rate_hz: Literal[16000] = 16000
    frame_seconds: Literal[0.032] = 0.032
    minimum_action_seconds: Literal[0.09] = 0.09
    action_semantic: Literal[InteractionActionSemantic.CANCEL_ASSISTANT] = (
        InteractionActionSemantic.CANCEL_ASSISTANT
    )


class LiveKitInteractionProvenance(FrozenBaseModel):
    baseline_kind: Literal[InteractionBaselineKind.LIVEKIT_RESPONSE] = (
        InteractionBaselineKind.LIVEKIT_RESPONSE
    )
    display_name: str = "LiveKit v1-mini completed utterance as response"
    implementation_version: str = IMPLEMENTATION_VERSION
    package_version: str
    model_version: Literal["v1-mini"] = "v1-mini"
    sample_rate_hz: Literal[16000] = 16000
    maximum_window_samples: int = EOT_MAX_SAMPLES
    gate_seconds: Literal[0.3] = LIVEKIT_GATE_SECONDS
    minimum_action_seconds: Literal[0.0] = 0.0
    action_semantic: Literal[InteractionActionSemantic.START_RESPONSE] = (
        InteractionActionSemantic.START_RESPONSE
    )


InteractionBaselineProvenance = Annotated[
    SileroInteractionProvenance
    | LiveKitVadInteractionProvenance
    | SmartTurnInteractionProvenance
    | LiveKitInteractionProvenance,
    Field(discriminator="baseline_kind"),
]


class InteractionBaselineObservation(FrozenBaseModel):
    elapsed_seconds: float = Field(ge=0.0)
    action_probability: float = Field(ge=0.0, le=1.0)


class InteractionBaselineEventPrediction(FrozenBaseModel):
    event_id: str
    interaction_kind: InteractionKind
    observations: tuple[InteractionBaselineObservation, ...] = Field(min_length=1)
    inference_duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_observation_times(self) -> InteractionBaselineEventPrediction:
        times = tuple(observation.elapsed_seconds for observation in self.observations)
        if times != tuple(sorted(set(times))):
            raise ValueError("Baseline observations must have unique increasing times.")
        return self


class InteractionBaselineManifest(FrozenBaseModel):
    schema_version: str = "voice-light-interaction-baseline-predictions-v1"
    generated_at: datetime
    detector: InteractionBaselineProvenance
    source_roots: tuple[str, ...] = Field(min_length=1)
    split: Literal["validation"] = "validation"
    detection_horizon_seconds: float = Field(gt=0.0)
    crop_random_seed: int = Field(ge=0)
    event_count: int = Field(ge=0)
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InteractionBaselineArtifact(FrozenBaseModel):
    manifest: InteractionBaselineManifest
    predictions: tuple[InteractionBaselineEventPrediction, ...]

    @model_validator(mode="after")
    def validate_predictions(self) -> InteractionBaselineArtifact:
        if len(self.predictions) != self.manifest.event_count:
            raise ValueError("Baseline prediction count does not match its manifest.")
        event_ids = tuple(prediction.event_id for prediction in self.predictions)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Baseline predictions contain duplicate event IDs.")
        if baseline_predictions_sha256(self.predictions) != self.manifest.predictions_sha256:
            raise ValueError("Baseline predictions do not match their manifest hash.")
        return self


class InteractionBaselineReport(FrozenBaseModel):
    schema_version: str = "voice-light-interaction-baseline-evaluation-v1"
    detector: InteractionBaselineProvenance
    threshold: float = Field(ge=0.0, le=1.0)
    backchannel_count: int = Field(ge=0)
    backchannel_false_action_count: int = Field(ge=0)
    backchannel_false_action_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    interruption_count: int = Field(ge=0)
    interruption_action_count: int = Field(ge=0)
    interruption_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    backchannel_action_latency: LatencyMetrics
    interruption_action_latency: LatencyMetrics


class InteractionBaselineSweep(FrozenBaseModel):
    schema_version: str = "voice-light-interaction-baseline-sweep-v1"
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reports: tuple[InteractionBaselineReport, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_thresholds(self) -> InteractionBaselineSweep:
        thresholds = tuple(report.threshold for report in self.reports)
        if thresholds != tuple(sorted(set(thresholds))):
            raise ValueError("Baseline sweep thresholds must be unique and increasing.")
        detectors = tuple(report.detector for report in self.reports)
        if any(detector != detectors[0] for detector in detectors[1:]):
            raise ValueError("Baseline sweep reports must use one detector configuration.")
        return self


@dataclass(frozen=True)
class _InteractionExample:
    event_id: str
    interaction_kind: InteractionKind
    waveform: NDArray[np.float32]
    anchor_seconds: float
    active_end_seconds: float


def predict_interaction_baseline(
    baseline_kind: InteractionBaselineKind,
    source_roots: tuple[Path, ...],
    detection_horizon_seconds: float,
    model_cache_directory: Path | None,
    crop_random_seed: int,
) -> InteractionBaselineArtifact:
    if not source_roots:
        raise ValueError("Interaction baseline prediction requires a synthetic corpus.")
    if detection_horizon_seconds <= 0.0:
        raise ValueError("Interaction detection horizon must be positive.")
    if crop_random_seed < 0:
        raise ValueError("Interaction crop random seed must be nonnegative.")
    examples = _interaction_examples(source_roots, crop_random_seed)
    match baseline_kind:
        case InteractionBaselineKind.SILERO_SPEECH_CANCEL:
            detector = SileroInteractionProvenance(package_version=version("silero-vad"))
            predictions = _predict_silero(examples, detection_horizon_seconds)
        case InteractionBaselineKind.LIVEKIT_VAD_CANCEL:
            detector = LiveKitVadInteractionProvenance(
                package_version=version("livekit-local-inference")
            )
            predictions = _predict_livekit_vad(examples, detection_horizon_seconds)
        case InteractionBaselineKind.SMART_TURN_RESPONSE:
            model_path = Path(
                hf_hub_download(
                    repo_id=MODEL_REPOSITORY,
                    filename=MODEL_FILENAME,
                    revision=MODEL_REVISION,
                    cache_dir=model_cache_directory,
                )
            )
            detector = SmartTurnInteractionProvenance(
                runtime_version=version("onnxruntime"),
                model_sha256=_file_sha256(model_path),
            )
            predictions = _predict_smart_turn(examples, model_cache_directory)
        case InteractionBaselineKind.LIVEKIT_RESPONSE:
            detector = LiveKitInteractionProvenance(
                package_version=version("livekit-local-inference")
            )
            predictions = _predict_livekit(examples)
    ordered = tuple(sorted(predictions, key=lambda prediction: prediction.event_id))
    return InteractionBaselineArtifact(
        manifest=InteractionBaselineManifest(
            generated_at=datetime.now(UTC),
            detector=detector,
            source_roots=tuple(str(root.resolve()) for root in source_roots),
            detection_horizon_seconds=detection_horizon_seconds,
            crop_random_seed=crop_random_seed,
            event_count=len(ordered),
            predictions_sha256=baseline_predictions_sha256(ordered),
        ),
        predictions=ordered,
    )


def evaluate_interaction_baseline(
    artifact: InteractionBaselineArtifact,
    threshold: float,
) -> InteractionBaselineReport:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Interaction threshold must be between zero and one.")
    minimum_action_seconds = artifact.manifest.detector.minimum_action_seconds
    backchannels = tuple(
        prediction
        for prediction in artifact.predictions
        if prediction.interaction_kind is InteractionKind.BACKCHANNEL
    )
    interruptions = tuple(
        prediction
        for prediction in artifact.predictions
        if prediction.interaction_kind is InteractionKind.INTERRUPTION
    )
    backchannel_latencies = tuple(
        latency
        for prediction in backchannels
        if (latency := _first_action_latency(prediction, threshold, minimum_action_seconds))
        is not None
    )
    interruption_latencies = tuple(
        latency
        for prediction in interruptions
        if (latency := _first_action_latency(prediction, threshold, minimum_action_seconds))
        is not None
    )
    return InteractionBaselineReport(
        detector=artifact.manifest.detector,
        threshold=threshold,
        backchannel_count=len(backchannels),
        backchannel_false_action_count=len(backchannel_latencies),
        backchannel_false_action_rate=_rate(len(backchannel_latencies), len(backchannels)),
        interruption_count=len(interruptions),
        interruption_action_count=len(interruption_latencies),
        interruption_recall=_rate(len(interruption_latencies), len(interruptions)),
        backchannel_action_latency=_latency_metrics(len(backchannels), backchannel_latencies),
        interruption_action_latency=_latency_metrics(len(interruptions), interruption_latencies),
    )


def sweep_interaction_baseline(
    artifact: InteractionBaselineArtifact,
    threshold_step: float,
) -> InteractionBaselineSweep:
    if not 0.0 < threshold_step <= 1.0:
        raise ValueError("Baseline threshold step must be in (0, 1].")
    thresholds = tuple(
        min(1.0, index * threshold_step) for index in range(round(1.0 / threshold_step) + 1)
    )
    if thresholds[-1] != 1.0:
        thresholds = (*thresholds, 1.0)
    return InteractionBaselineSweep(
        predictions_sha256=artifact.manifest.predictions_sha256,
        reports=tuple(
            evaluate_interaction_baseline(artifact, threshold) for threshold in thresholds
        ),
    )


def baseline_predictions_sha256(
    predictions: Sequence[InteractionBaselineEventPrediction],
) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        digest.update(prediction.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _interaction_examples(
    source_roots: tuple[Path, ...],
    crop_random_seed: int,
) -> tuple[_InteractionExample, ...]:
    examples: list[_InteractionExample] = []
    for source_root in source_roots:
        dataset = AnchoredSyntheticTurnTakingDataset(
            root=source_root,
            split=TrainingCorpusSplit.VALIDATION,
            compiler_config=ConversationCompilerConfig(crop_variant_count=1),
            sampling_config=SyntheticAnchorSamplingConfig(),
            augmenter=None,
            random_seed=crop_random_seed,
            randomize=False,
        )
        for index, entry in enumerate(dataset.anchors):
            if not isinstance(entry.anchor, BackchannelAnchor | InterruptionAnchor):
                continue
            anchored = dataset.anchored_item(index)
            kind = (
                InteractionKind.BACKCHANNEL
                if isinstance(entry.anchor, BackchannelAnchor)
                else InteractionKind.INTERRUPTION
            )
            if kind is InteractionKind.BACKCHANNEL:
                active_frames = anchored.item.targets.event_mask[:, 3] & (
                    anchored.item.targets.event_targets[:, 3] >= 0.5
                )
            else:
                active_frames = anchored.item.targets.primary_mask & (
                    anchored.item.targets.yield_probability < 0.5
                )
            active_indices = torch.nonzero(active_frames, as_tuple=False).flatten()
            following = active_indices[active_indices >= anchored.anchor_frame_index]
            if following.numel() == 0:
                raise ValueError(
                    f"Interaction event has no active target: {anchored.item.sample_id}"
                )
            contiguous_end = int(following[0])
            for frame_index in following[1:].tolist():
                if frame_index != contiguous_end + 1:
                    break
                contiguous_end = frame_index
            examples.append(
                _InteractionExample(
                    event_id=anchored.item.sample_id,
                    interaction_kind=kind,
                    waveform=anchored.item.waveform.numpy().astype(np.float32, copy=False),
                    anchor_seconds=anchored.anchor_frame_index * FRAME_SECONDS,
                    active_end_seconds=(contiguous_end + 1) * FRAME_SECONDS,
                )
            )
    return tuple(examples)


def _predict_silero(
    examples: tuple[_InteractionExample, ...],
    detection_horizon_seconds: float,
) -> tuple[InteractionBaselineEventPrediction, ...]:
    model = load_causal_silero_model(use_onnx=True)
    predictions: list[InteractionBaselineEventPrediction] = []
    for example in examples:
        model.reset_states()
        context_start = max(0.0, example.anchor_seconds - SILERO_CONTEXT_SECONDS)
        end_seconds = min(
            example.active_end_seconds,
            example.anchor_seconds + detection_horizon_seconds,
        )
        start_sample = round(context_start * MODEL_SAMPLE_RATE)
        end_sample = round(end_seconds * MODEL_SAMPLE_RATE)
        observations: list[InteractionBaselineObservation] = []
        inference_duration_seconds = 0.0
        for window_start in range(start_sample, end_sample, SILERO_WINDOW_SAMPLES):
            window = example.waveform[window_start : window_start + SILERO_WINDOW_SAMPLES]
            if window.size < SILERO_WINDOW_SAMPLES:
                window = np.pad(window, (0, SILERO_WINDOW_SAMPLES - window.size))
            started_at = time.perf_counter()
            probability = float(
                model(torch.from_numpy(window.astype(np.float32, copy=False)), MODEL_SAMPLE_RATE)
                .reshape(-1)[0]
                .item()
            )
            inference_duration_seconds += time.perf_counter() - started_at
            window_end_seconds = (window_start + SILERO_WINDOW_SAMPLES) / MODEL_SAMPLE_RATE
            if window_end_seconds < example.anchor_seconds:
                continue
            observations.append(
                InteractionBaselineObservation(
                    elapsed_seconds=max(0.0, window_end_seconds - example.anchor_seconds),
                    action_probability=probability,
                )
            )
        predictions.append(
            InteractionBaselineEventPrediction(
                event_id=example.event_id,
                interaction_kind=example.interaction_kind,
                observations=tuple(observations),
                inference_duration_seconds=inference_duration_seconds,
            )
        )
    return tuple(predictions)


def _predict_smart_turn(
    examples: tuple[_InteractionExample, ...],
    model_cache_directory: Path | None,
) -> tuple[InteractionBaselineEventPrediction, ...]:
    inference = load_smart_turn_v3_inference(
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        model_filename=MODEL_FILENAME,
        cache_directory=model_cache_directory,
    )
    predictions: list[InteractionBaselineEventPrediction] = []
    for example in examples:
        evaluation_seconds = example.active_end_seconds + SMART_TURN_GATE_SECONDS
        end_sample = min(len(example.waveform), round(evaluation_seconds * MODEL_SAMPLE_RATE))
        start_sample = max(0, end_sample - round(MAX_MODEL_WINDOW_SECONDS * MODEL_SAMPLE_RATE))
        audio = example.waveform[start_sample:end_sample]
        started_at = time.perf_counter()
        probability = smart_turn_completion_probability(inference=inference, audio=audio)
        inference_duration_seconds = time.perf_counter() - started_at
        predictions.append(
            _single_observation_prediction(
                example, evaluation_seconds, probability, inference_duration_seconds
            )
        )
    return tuple(predictions)


def _predict_livekit_vad(
    examples: tuple[_InteractionExample, ...],
    detection_horizon_seconds: float,
) -> tuple[InteractionBaselineEventPrediction, ...]:
    inference = load_livekit_v1_mini_inference()
    predictions: list[InteractionBaselineEventPrediction] = []
    for example in examples:
        start_sample = round(example.anchor_seconds * MODEL_SAMPLE_RATE)
        end_seconds = min(
            example.active_end_seconds,
            example.anchor_seconds + detection_horizon_seconds,
        )
        end_sample = round(end_seconds * MODEL_SAMPLE_RATE)
        observations: list[InteractionBaselineObservation] = []
        inference_duration_seconds = 0.0
        for window_start in range(start_sample, end_sample, VAD_WINDOW_SAMPLES):
            float_audio = example.waveform[window_start : window_start + VAD_WINDOW_SAMPLES]
            if float_audio.size < VAD_WINDOW_SAMPLES:
                float_audio = np.pad(float_audio, (0, VAD_WINDOW_SAMPLES - float_audio.size))
            audio = np.clip(float_audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
            started_at = time.perf_counter()
            probability = inference.vad_model.predict(audio)
            inference_duration_seconds += time.perf_counter() - started_at
            window_end_seconds = (window_start + VAD_WINDOW_SAMPLES) / MODEL_SAMPLE_RATE
            observations.append(
                InteractionBaselineObservation(
                    elapsed_seconds=window_end_seconds - example.anchor_seconds,
                    action_probability=probability,
                )
            )
        predictions.append(
            InteractionBaselineEventPrediction(
                event_id=example.event_id,
                interaction_kind=example.interaction_kind,
                observations=tuple(observations),
                inference_duration_seconds=inference_duration_seconds,
            )
        )
    return tuple(predictions)


def _predict_livekit(
    examples: tuple[_InteractionExample, ...],
) -> tuple[InteractionBaselineEventPrediction, ...]:
    inference = load_livekit_v1_mini_inference()
    predictions: list[InteractionBaselineEventPrediction] = []
    for example in examples:
        evaluation_seconds = example.active_end_seconds + LIVEKIT_GATE_SECONDS
        end_sample = min(len(example.waveform), round(evaluation_seconds * MODEL_SAMPLE_RATE))
        start_sample = max(0, end_sample - EOT_MAX_SAMPLES)
        audio = np.clip(example.waveform[start_sample:end_sample] * 32767.0, -32768.0, 32767.0)
        started_at = time.perf_counter()
        probability = inference.end_of_turn_model.predict(audio.astype(np.int16))
        inference_duration_seconds = time.perf_counter() - started_at
        predictions.append(
            _single_observation_prediction(
                example, evaluation_seconds, probability, inference_duration_seconds
            )
        )
    return tuple(predictions)


def _single_observation_prediction(
    example: _InteractionExample,
    evaluation_seconds: float,
    probability: float,
    inference_duration_seconds: float,
) -> InteractionBaselineEventPrediction:
    return InteractionBaselineEventPrediction(
        event_id=example.event_id,
        interaction_kind=example.interaction_kind,
        observations=(
            InteractionBaselineObservation(
                elapsed_seconds=evaluation_seconds - example.anchor_seconds,
                action_probability=probability,
            ),
        ),
        inference_duration_seconds=inference_duration_seconds,
    )


def _first_action_latency(
    prediction: InteractionBaselineEventPrediction,
    threshold: float,
    minimum_action_seconds: float,
) -> float | None:
    run_start: float | None = None
    previous_seconds: float | None = None
    for observation in prediction.observations:
        if observation.action_probability < threshold:
            run_start = None
            previous_seconds = observation.elapsed_seconds
            continue
        if run_start is None:
            run_start = observation.elapsed_seconds
        previous_seconds = observation.elapsed_seconds
        if previous_seconds - run_start >= minimum_action_seconds - 1e-9:
            return run_start
    return None


def _latency_metrics(eligible_count: int, latencies: tuple[float, ...]) -> LatencyMetrics:
    ordered = tuple(sorted(latencies))
    return LatencyMetrics(
        eligible_count=eligible_count,
        observed_count=len(ordered),
        censored_count=eligible_count - len(ordered),
        median_seconds=_quantile(ordered, 0.5),
        p95_seconds=_quantile(ordered, 0.95),
    )


def _quantile(values: tuple[float, ...], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values), quantile))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
