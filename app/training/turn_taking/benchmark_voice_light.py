from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch
from torch import Tensor

from app.local.training_corpus.export import MaterializedTrainingSample
from app.training.turn_taking.backbone import FeatureBackbone
from app.training.turn_taking.benchmark_models import (
    CandidateInventory,
    CandidatePrediction,
    CompletionCandidatePrediction,
    CompletionPredictionArtifact,
    CompletionPredictionManifest,
    PredictionArtifact,
    PredictionManifest,
    TurnCompletionInventory,
    VoiceLightCompletionDetectorConfiguration,
    VoiceLightCompletionDetectorProvenance,
    VoiceLightDetectorConfiguration,
    VoiceLightDetectorProvenance,
    completion_prediction_rows_sha256,
    prediction_rows_sha256,
)
from app.training.turn_taking.config import TrainingConfig
from app.training.turn_taking.data import TrainingBatch
from app.training.turn_taking.model import TurnTakingAdapter

IMPLEMENTATION_VERSION = "voice-light-causal-adapters-v1"
HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class LoadedVoiceLightCheckpoint:
    path: Path
    sha256: str
    optimizer_step: int
    config: TrainingConfig
    adapter: TurnTakingAdapter


@dataclass(frozen=True)
class _TargetFrameKey:
    dataset_id: str
    conversation_id: str
    user_side: str
    user_audio_path: str
    frame_index: int


@dataclass(frozen=True)
class _TargetReference:
    candidate_id: str
    absolute_time_seconds: float
    silence_duration_seconds: float


@dataclass(frozen=True)
class _SelectedPrediction:
    context_frame_count: int
    window_id: str
    prediction: CandidatePrediction


@dataclass(frozen=True)
class _CompletionTargetReference:
    candidate_id: str
    absolute_time_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class _SelectedCompletionPrediction:
    context_frame_count: int
    window_id: str
    prediction: CompletionCandidatePrediction


def load_voice_light_checkpoint(path: Path) -> LoadedVoiceLightCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TrainingConfig.model_validate(payload["training_config"])
    adapter = TurnTakingAdapter(config.adapter)
    adapter.load_state_dict(payload["adapter_state"], strict=True)
    return LoadedVoiceLightCheckpoint(
        path=path,
        sha256=_file_sha256(path),
        optimizer_step=int(payload["optimizer_step"]),
        config=config,
        adapter=adapter,
    )


def predict_voice_light_checkpoints(
    backbone: FeatureBackbone,
    checkpoints: tuple[LoadedVoiceLightCheckpoint, ...],
    batches: Iterable[TrainingBatch],
    samples: tuple[MaterializedTrainingSample, ...],
    inventory: CandidateInventory,
    model_repository: str,
    model_revision: str,
    device: torch.device,
    total_batch_count: int | None = None,
    progress_output: TextIO | None = None,
) -> tuple[PredictionArtifact, ...]:
    if not checkpoints:
        raise ValueError("Voice Light prediction requires at least one checkpoint.")
    if total_batch_count is not None and total_batch_count <= 0:
        raise ValueError("Total batch count must be positive when provided.")
    _validate_checkpoint_configs(checkpoints)
    sample_by_window_id = {sample.window_id: sample for sample in samples}
    target_lookup = _target_lookup(inventory)
    selected_by_checkpoint: list[dict[_TargetFrameKey, _SelectedPrediction]] = [
        {} for _ in checkpoints
    ]
    for checkpoint in checkpoints:
        checkpoint.adapter.to(device).eval()
    progress_started_at = time.perf_counter()
    if progress_output is not None:
        _write_progress(
            output=progress_output,
            completed_batch_count=0,
            total_batch_count=total_batch_count,
            elapsed_seconds=0.0,
        )
    with torch.no_grad():
        for batch_index, batch in enumerate(batches, start=1):
            feature_started_at = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = backbone.extract(batch.waveforms, batch.waveform_lengths)
            feature_duration_per_sample = (time.perf_counter() - feature_started_at) / len(
                batch.sample_ids
            )
            taps = tuple(tap.to(device) for tap in features.taps)
            assistant_speaking = _align_assistant_speaking(
                batch.assistant_speaking.to(device),
                taps[0].shape[1],
            )
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                adapter_started_at = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = checkpoint.adapter(taps, assistant_speaking)
                adapter_duration_per_sample = (time.perf_counter() - adapter_started_at) / len(
                    batch.sample_ids
                )
                _collect_batch_predictions(
                    probabilities=output.yield_logits.sigmoid().cpu(),
                    frame_mask=features.frame_mask.cpu(),
                    batch=batch,
                    sample_by_window_id=sample_by_window_id,
                    target_lookup=target_lookup,
                    frame_seconds=inventory.manifest.frame_seconds,
                    inference_duration_seconds=(
                        feature_duration_per_sample + adapter_duration_per_sample
                    ),
                    selected=selected_by_checkpoint[checkpoint_index],
                )
            if progress_output is not None:
                _write_progress(
                    output=progress_output,
                    completed_batch_count=batch_index,
                    total_batch_count=total_batch_count,
                    elapsed_seconds=time.perf_counter() - progress_started_at,
                )
    return tuple(
        _prediction_artifact(
            checkpoint=checkpoint,
            inventory=inventory,
            model_repository=model_repository,
            model_revision=model_revision,
            selected=selected,
        )
        for checkpoint, selected in zip(checkpoints, selected_by_checkpoint, strict=True)
    )


def predict_voice_light_completion_checkpoints(
    backbone: FeatureBackbone,
    checkpoints: tuple[LoadedVoiceLightCheckpoint, ...],
    batches: Iterable[TrainingBatch],
    samples: tuple[MaterializedTrainingSample, ...],
    inventory: TurnCompletionInventory,
    model_repository: str,
    model_revision: str,
    device: torch.device,
    total_batch_count: int | None = None,
    progress_output: TextIO | None = None,
) -> tuple[CompletionPredictionArtifact, ...]:
    if inventory.manifest.split.value != "validation":
        raise ValueError("Turn-completion prediction is restricted to validation.")
    if not math.isclose(inventory.manifest.causal_horizon_seconds, 2.0, abs_tol=1e-6):
        raise ValueError("Voice Light completion inference requires full two-second candidates.")
    if not checkpoints:
        raise ValueError("Voice Light prediction requires at least one checkpoint.")
    if total_batch_count is not None and total_batch_count <= 0:
        raise ValueError("Total batch count must be positive when provided.")
    _validate_checkpoint_configs(checkpoints)
    sample_by_window_id = {sample.window_id: sample for sample in samples}
    target_lookup = _completion_target_lookup(inventory)
    selected_by_checkpoint: list[dict[_TargetFrameKey, _SelectedCompletionPrediction]] = [
        {} for _ in checkpoints
    ]
    for checkpoint in checkpoints:
        checkpoint.adapter.to(device).eval()
    progress_started_at = time.perf_counter()
    if progress_output is not None:
        _write_progress(progress_output, 0, total_batch_count, 0.0)
    with torch.no_grad():
        for batch_index, batch in enumerate(batches, start=1):
            feature_started_at = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = backbone.extract(batch.waveforms, batch.waveform_lengths)
            feature_duration_per_sample = (time.perf_counter() - feature_started_at) / len(
                batch.sample_ids
            )
            taps = tuple(tap.to(device) for tap in features.taps)
            assistant_speaking = _align_assistant_speaking(
                batch.assistant_speaking.to(device), taps[0].shape[1]
            )
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                adapter_started_at = time.perf_counter()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = checkpoint.adapter(taps, assistant_speaking)
                adapter_duration_per_sample = (time.perf_counter() - adapter_started_at) / len(
                    batch.sample_ids
                )
                _collect_completion_batch_predictions(
                    probabilities=output.event_logits[..., 0].sigmoid().cpu(),
                    frame_mask=features.frame_mask.cpu(),
                    batch=batch,
                    sample_by_window_id=sample_by_window_id,
                    target_lookup=target_lookup,
                    frame_seconds=inventory.manifest.frame_seconds,
                    inference_duration_seconds=(
                        feature_duration_per_sample + adapter_duration_per_sample
                    ),
                    selected=selected_by_checkpoint[checkpoint_index],
                )
            if progress_output is not None:
                _write_progress(
                    progress_output,
                    batch_index,
                    total_batch_count,
                    time.perf_counter() - progress_started_at,
                )
    return tuple(
        _completion_prediction_artifact(
            checkpoint=checkpoint,
            inventory=inventory,
            model_repository=model_repository,
            model_revision=model_revision,
            selected=selected,
        )
        for checkpoint, selected in zip(checkpoints, selected_by_checkpoint, strict=True)
    )


def _write_progress(
    output: TextIO,
    completed_batch_count: int,
    total_batch_count: int | None,
    elapsed_seconds: float,
) -> None:
    average_seconds = elapsed_seconds / completed_batch_count if completed_batch_count > 0 else None
    if total_batch_count is None:
        total = "?"
        percent = "?"
        eta = "?"
    else:
        total = str(total_batch_count)
        percent = f"{completed_batch_count / total_batch_count:.1%}"
        remaining_batches = total_batch_count - completed_batch_count
        eta = (
            _format_duration(remaining_batches * average_seconds)
            if average_seconds is not None
            else "?"
        )
    average = f"{average_seconds:.1f}s/batch" if average_seconds is not None else "?"
    print(
        "Voice Light inference: "
        f"{completed_batch_count}/{total} batches ({percent}), "
        f"elapsed {_format_duration(elapsed_seconds)}, ETA {eta}, average {average}",
        file=output,
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    rounded_seconds = max(0, round(seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def _collect_batch_predictions(
    probabilities: Tensor,
    frame_mask: Tensor,
    batch: TrainingBatch,
    sample_by_window_id: dict[str, MaterializedTrainingSample],
    target_lookup: dict[_TargetFrameKey, _TargetReference],
    frame_seconds: float,
    inference_duration_seconds: float,
    selected: dict[_TargetFrameKey, _SelectedPrediction],
) -> None:
    output_frame_count = probabilities.shape[1]
    target_frame_count = batch.targets.yield_probability.shape[1]
    target_indices = _aligned_target_indices(output_frame_count, target_frame_count)
    for batch_index, window_id in enumerate(batch.sample_ids):
        sample = sample_by_window_id[window_id]
        sample_start_frame = round(sample.start_seconds / frame_seconds)
        for output_index, target_index in enumerate(target_indices):
            if not bool(frame_mask[batch_index, output_index]):
                continue
            key = _TargetFrameKey(
                dataset_id=str(sample.dataset_id),
                conversation_id=str(sample.sample_id),
                user_side=sample.user_side.value,
                user_audio_path=sample.user_audio_path,
                frame_index=sample_start_frame + target_index,
            )
            target = target_lookup.get(key)
            if target is None:
                continue
            candidate_prediction = CandidatePrediction(
                candidate_id=target.candidate_id,
                absolute_time_seconds=target.absolute_time_seconds,
                silence_duration_seconds=target.silence_duration_seconds,
                yield_probability=float(probabilities[batch_index, output_index].item()),
                inference_duration_seconds=inference_duration_seconds,
            )
            candidate_selection = _SelectedPrediction(
                context_frame_count=target_index + 1,
                window_id=window_id,
                prediction=candidate_prediction,
            )
            existing = selected.get(key)
            if existing is None or _prefer_candidate(candidate_selection, existing):
                selected[key] = candidate_selection


def _collect_completion_batch_predictions(
    probabilities: Tensor,
    frame_mask: Tensor,
    batch: TrainingBatch,
    sample_by_window_id: dict[str, MaterializedTrainingSample],
    target_lookup: dict[_TargetFrameKey, _CompletionTargetReference],
    frame_seconds: float,
    inference_duration_seconds: float,
    selected: dict[_TargetFrameKey, _SelectedCompletionPrediction],
) -> None:
    output_frame_count = probabilities.shape[1]
    target_frame_count = batch.targets.yield_probability.shape[1]
    for batch_index, window_id in enumerate(batch.sample_ids):
        sample = sample_by_window_id[window_id]
        sample_start_frame = round(sample.start_seconds / frame_seconds)
        for output_index, target_index in _earliest_aligned_output_indices(
            output_frame_count,
            target_frame_count,
        ):
            if not bool(frame_mask[batch_index, output_index]):
                continue
            key = _TargetFrameKey(
                dataset_id=str(sample.dataset_id),
                conversation_id=str(sample.sample_id),
                user_side=sample.user_side.value,
                user_audio_path=sample.user_audio_path,
                frame_index=sample_start_frame + target_index,
            )
            target = target_lookup.get(key)
            if target is None:
                continue
            selection = _SelectedCompletionPrediction(
                context_frame_count=target_index + 1,
                window_id=window_id,
                prediction=CompletionCandidatePrediction(
                    candidate_id=target.candidate_id,
                    absolute_time_seconds=target.absolute_time_seconds,
                    elapsed_seconds=target.elapsed_seconds,
                    completion_probability=float(probabilities[batch_index, output_index].item()),
                    inference_duration_seconds=inference_duration_seconds,
                ),
            )
            existing = selected.get(key)
            if existing is None or _prefer_completion_candidate(selection, existing):
                selected[key] = selection


def _target_lookup(inventory: CandidateInventory) -> dict[_TargetFrameKey, _TargetReference]:
    frame_seconds = inventory.manifest.frame_seconds
    lookup: dict[_TargetFrameKey, _TargetReference] = {}
    for candidate in inventory.candidates:
        for target_point in candidate.target_points:
            frame_index = round(target_point.absolute_time_seconds / frame_seconds) - 1
            if not math.isclose(
                (frame_index + 1) * frame_seconds,
                target_point.absolute_time_seconds,
                abs_tol=1e-6,
            ):
                raise ValueError("Candidate target is not aligned to the inventory frame grid.")
            key = _TargetFrameKey(
                dataset_id=candidate.dataset_id,
                conversation_id=candidate.conversation_id,
                user_side=candidate.user_side,
                user_audio_path=candidate.user_audio_path,
                frame_index=frame_index,
            )
            if key in lookup:
                raise ValueError("Candidate inventory contains duplicate target frame keys.")
            lookup[key] = _TargetReference(
                candidate_id=candidate.candidate_id,
                absolute_time_seconds=target_point.absolute_time_seconds,
                silence_duration_seconds=target_point.silence_duration_seconds,
            )
    return lookup


def _completion_target_lookup(
    inventory: TurnCompletionInventory,
) -> dict[_TargetFrameKey, _CompletionTargetReference]:
    frame_seconds = inventory.manifest.frame_seconds
    lookup: dict[_TargetFrameKey, _CompletionTargetReference] = {}
    for candidate in inventory.candidates:
        anchor_frame = round(candidate.anchor_seconds / frame_seconds)
        if not math.isclose(
            anchor_frame * frame_seconds,
            candidate.anchor_seconds,
            abs_tol=1e-6,
        ):
            raise ValueError("Completion candidate anchor is not aligned to the frame grid.")
        first_target = candidate.target_points[0]
        expected_time = (anchor_frame + 1) * frame_seconds
        if not math.isclose(
            first_target.absolute_time_seconds,
            expected_time,
            abs_tol=1e-6,
        ) or not math.isclose(
            first_target.elapsed_seconds,
            frame_seconds,
            abs_tol=1e-6,
        ):
            raise ValueError("Completion candidate first score must be one frame after its anchor.")
        key = _TargetFrameKey(
            dataset_id=candidate.dataset_id,
            conversation_id=candidate.conversation_id,
            user_side=candidate.user_side,
            user_audio_path=candidate.user_audio_path,
            frame_index=anchor_frame,
        )
        if key in lookup:
            raise ValueError("Completion inventory contains duplicate boundary frame keys.")
        lookup[key] = _CompletionTargetReference(
            candidate_id=candidate.candidate_id,
            absolute_time_seconds=first_target.absolute_time_seconds,
            elapsed_seconds=first_target.elapsed_seconds,
        )
    return lookup


def _prediction_artifact(
    checkpoint: LoadedVoiceLightCheckpoint,
    inventory: CandidateInventory,
    model_repository: str,
    model_revision: str,
    selected: dict[_TargetFrameKey, _SelectedPrediction],
) -> PredictionArtifact:
    predictions = tuple(
        sorted(
            (selection.prediction for selection in selected.values()),
            key=lambda prediction: (
                prediction.candidate_id,
                prediction.absolute_time_seconds,
            ),
        )
    )
    detector = VoiceLightDetectorProvenance(
        display_name=f"Voice Light step {checkpoint.optimizer_step:,}",
        implementation_version=IMPLEMENTATION_VERSION,
        model_repository=model_repository,
        model_revision=model_revision,
        checkpoint_path=str(checkpoint.path),
        checkpoint_sha256=checkpoint.sha256,
        configuration=VoiceLightDetectorConfiguration(
            model_identifier=checkpoint.config.model_identifier,
            lookahead_tokens=checkpoint.config.lookahead_tokens,
            encoder_frame_seconds=checkpoint.config.encoder_frame_seconds,
            optimizer_step=checkpoint.optimizer_step,
        ),
    )
    return PredictionArtifact(
        manifest=PredictionManifest(
            inventory_sha256=inventory.manifest.candidate_sha256,
            split=inventory.manifest.split,
            detector=detector,
            prediction_count=len(predictions),
            predictions_sha256=prediction_rows_sha256(predictions),
        ),
        predictions=predictions,
    )


def _completion_prediction_artifact(
    checkpoint: LoadedVoiceLightCheckpoint,
    inventory: TurnCompletionInventory,
    model_repository: str,
    model_revision: str,
    selected: dict[_TargetFrameKey, _SelectedCompletionPrediction],
) -> CompletionPredictionArtifact:
    predictions = tuple(
        sorted(
            (selection.prediction for selection in selected.values()),
            key=lambda prediction: (
                prediction.candidate_id,
                prediction.absolute_time_seconds,
            ),
        )
    )
    if len(predictions) != inventory.manifest.candidate_count:
        raise ValueError("Voice Light completion inference did not score every inventory boundary.")
    detector = VoiceLightCompletionDetectorProvenance(
        display_name=f"Voice Light completion head step {checkpoint.optimizer_step:,}",
        implementation_version="voice-light-turn-completion-adapter-v2",
        model_repository=model_repository,
        model_revision=model_revision,
        checkpoint_path=str(checkpoint.path),
        checkpoint_sha256=checkpoint.sha256,
        configuration=VoiceLightCompletionDetectorConfiguration(
            model_identifier=checkpoint.config.model_identifier,
            lookahead_tokens=checkpoint.config.lookahead_tokens,
            encoder_frame_seconds=checkpoint.config.encoder_frame_seconds,
            optimizer_step=checkpoint.optimizer_step,
        ),
    )
    return CompletionPredictionArtifact(
        manifest=CompletionPredictionManifest(
            inventory_sha256=inventory.manifest.candidate_sha256,
            split=inventory.manifest.split,
            detector=detector,
            prediction_count=len(predictions),
            predictions_sha256=completion_prediction_rows_sha256(predictions),
        ),
        predictions=predictions,
    )


def _prefer_candidate(candidate: _SelectedPrediction, existing: _SelectedPrediction) -> bool:
    if candidate.context_frame_count != existing.context_frame_count:
        return candidate.context_frame_count > existing.context_frame_count
    return candidate.window_id < existing.window_id


def _prefer_completion_candidate(
    candidate: _SelectedCompletionPrediction,
    existing: _SelectedCompletionPrediction,
) -> bool:
    if candidate.context_frame_count != existing.context_frame_count:
        return candidate.context_frame_count > existing.context_frame_count
    return candidate.window_id < existing.window_id


def _validate_checkpoint_configs(checkpoints: tuple[LoadedVoiceLightCheckpoint, ...]) -> None:
    reference = checkpoints[0].config
    for checkpoint in checkpoints[1:]:
        if checkpoint.config.model_identifier != reference.model_identifier:
            raise ValueError("Voice Light checkpoints use different backbone models.")
        if checkpoint.config.sample_rate_hz != reference.sample_rate_hz:
            raise ValueError("Voice Light checkpoints use different sample rates.")
        if checkpoint.config.lookahead_tokens != reference.lookahead_tokens:
            raise ValueError("Voice Light checkpoints use different lookahead settings.")
        if checkpoint.config.adapter != reference.adapter:
            raise ValueError("Voice Light checkpoints use different adapter configurations.")


def _align_assistant_speaking(values: Tensor, frame_count: int) -> Tensor:
    indices = (
        torch.linspace(0, values.shape[1] - 1, frame_count, device=values.device).round().long()
    )
    return values[:, indices]


def _aligned_target_indices(output_frame_count: int, target_frame_count: int) -> tuple[int, ...]:
    if output_frame_count <= 0 or target_frame_count <= 0:
        raise ValueError("Frame counts must be positive.")
    return tuple(
        torch.linspace(0, target_frame_count - 1, output_frame_count).round().long().tolist()
    )


def _earliest_aligned_output_indices(
    output_frame_count: int,
    target_frame_count: int,
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    observed_target_indices: set[int] = set()
    for output_index, target_index in enumerate(
        _aligned_target_indices(output_frame_count, target_frame_count)
    ):
        if target_index in observed_target_indices:
            continue
        observed_target_indices.add(target_index)
        pairs.append((output_index, target_index))
    return tuple(pairs)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
