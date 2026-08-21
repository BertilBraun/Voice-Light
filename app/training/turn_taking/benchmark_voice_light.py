from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from app.local.training_corpus.export import MaterializedTrainingSample
from app.training.turn_taking.backbone import FeatureBackbone
from app.training.turn_taking.benchmark_models import (
    CandidateInventory,
    CandidatePrediction,
    PredictionArtifact,
    PredictionManifest,
    VoiceLightDetectorConfiguration,
    VoiceLightDetectorProvenance,
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
) -> tuple[PredictionArtifact, ...]:
    if not checkpoints:
        raise ValueError("Voice Light prediction requires at least one checkpoint.")
    _validate_checkpoint_configs(checkpoints)
    sample_by_window_id = {sample.window_id: sample for sample in samples}
    target_lookup = _target_lookup(inventory)
    selected_by_checkpoint: list[dict[_TargetFrameKey, _SelectedPrediction]] = [
        {} for _ in checkpoints
    ]
    for checkpoint in checkpoints:
        checkpoint.adapter.to(device).eval()
    with torch.no_grad():
        for batch in batches:
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
    target_indices = (
        torch.linspace(0, target_frame_count - 1, output_frame_count).round().long().tolist()
    )
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


def _prefer_candidate(candidate: _SelectedPrediction, existing: _SelectedPrediction) -> bool:
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
