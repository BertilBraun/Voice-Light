from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import torch

from app.local.synthetic_generation.conversation_compiler import (
    BackchannelAnchor,
    ConversationCompilerConfig,
    InterruptionAnchor,
)
from app.local.training_corpus.splits import TrainingCorpusSplit
from app.training.turn_taking.backbone import FeatureBackbone
from app.training.turn_taking.benchmark_voice_light import LoadedVoiceLightCheckpoint
from app.training.turn_taking.config import SyntheticAnchorSamplingConfig
from app.training.turn_taking.data import collate_training_items
from app.training.turn_taking.interaction_evaluation import (
    InteractionEventPrediction,
    InteractionKind,
    InteractionObservation,
    InteractionPredictionArtifact,
    InteractionPredictionManifest,
    MissingInteractionEvent,
    MissingInteractionReason,
    aligned_source_indices,
    interaction_predictions_sha256,
    output_anchor_index,
)
from app.training.turn_taking.synthetic_dataset import (
    AnchoredSyntheticTrainingItem,
    AnchoredSyntheticTurnTakingDataset,
)


def predict_synthetic_interactions(
    backbone: FeatureBackbone,
    checkpoint: LoadedVoiceLightCheckpoint,
    source_roots: tuple[Path, ...],
    batch_size: int,
    detection_horizon_seconds: float,
    device: torch.device,
    maximum_events: int | None = None,
) -> InteractionPredictionArtifact:
    if not source_roots:
        raise ValueError("Interaction prediction requires at least one synthetic corpus.")
    if batch_size <= 0:
        raise ValueError("Interaction prediction batch size must be positive.")
    if detection_horizon_seconds <= 0.0:
        raise ValueError("Interaction detection horizon must be positive.")
    if maximum_events is not None and maximum_events <= 0:
        raise ValueError("Maximum events must be positive when provided.")
    examples = _interaction_examples(source_roots, checkpoint)
    if maximum_events is not None:
        examples = examples[:maximum_events]
    checkpoint.adapter.to(device).eval()
    predictions: list[InteractionEventPrediction] = []
    missing: list[MissingInteractionEvent] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            selected = examples[start : start + batch_size]
            batch = collate_training_items(tuple(example.item for example in selected))
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = backbone.extract(batch.waveforms, batch.waveform_lengths)
                taps = tuple(tap.to(device) for tap in features.taps)
                assistant_speaking = _align_assistant_speaking(
                    batch.assistant_speaking.to(device), taps[0].shape[1]
                )
                event_probabilities = (
                    checkpoint.adapter(taps, assistant_speaking)
                    .event_logits.sigmoid()
                    .float()
                    .cpu()
                )
            for batch_index, example in enumerate(selected):
                prediction, absent = _event_prediction(
                    example=example,
                    event_probabilities=event_probabilities[batch_index],
                    frame_mask=features.frame_mask[batch_index].cpu(),
                    detection_horizon_seconds=detection_horizon_seconds,
                    frame_seconds=checkpoint.config.encoder_frame_seconds,
                )
                if prediction is not None:
                    predictions.append(prediction)
                if absent is not None:
                    missing.append(absent)
    ordered_predictions = tuple(sorted(predictions, key=lambda prediction: prediction.event_id))
    ordered_missing = tuple(sorted(missing, key=lambda event: event.event_id))
    return InteractionPredictionArtifact(
        manifest=InteractionPredictionManifest(
            generated_at=datetime.now(UTC),
            checkpoint_sha256=checkpoint.sha256,
            optimizer_step=checkpoint.optimizer_step,
            source_roots=tuple(str(root.resolve()) for root in source_roots),
            split=TrainingCorpusSplit.VALIDATION.value,
            frame_seconds=checkpoint.config.encoder_frame_seconds,
            detection_horizon_seconds=detection_horizon_seconds,
            eligible_event_count=len(examples),
            prediction_count=len(ordered_predictions),
            missing_event_count=len(ordered_missing),
            predictions_sha256=interaction_predictions_sha256(ordered_predictions),
        ),
        predictions=ordered_predictions,
        missing_events=ordered_missing,
    )


def _interaction_examples(
    source_roots: tuple[Path, ...],
    checkpoint: LoadedVoiceLightCheckpoint,
) -> tuple[AnchoredSyntheticTrainingItem, ...]:
    selected: list[AnchoredSyntheticTrainingItem] = []
    for source_index, source_root in enumerate(source_roots):
        dataset = AnchoredSyntheticTurnTakingDataset(
            root=source_root,
            split=TrainingCorpusSplit.VALIDATION,
            compiler_config=ConversationCompilerConfig(
                sample_rate_hz=checkpoint.config.sample_rate_hz,
                crop_duration_seconds=checkpoint.config.context_seconds,
                crop_variant_count=1,
            ),
            sampling_config=SyntheticAnchorSamplingConfig(),
            augmenter=None,
            random_seed=checkpoint.config.random_seed + source_index,
            randomize=False,
        )
        selected.extend(
            dataset.anchored_item(index)
            for index, entry in enumerate(dataset.anchors)
            if isinstance(entry.anchor, BackchannelAnchor | InterruptionAnchor)
        )
    return tuple(selected)


def _event_prediction(
    example: AnchoredSyntheticTrainingItem,
    event_probabilities: torch.Tensor,
    frame_mask: torch.Tensor,
    detection_horizon_seconds: float,
    frame_seconds: float,
) -> tuple[InteractionEventPrediction | None, MissingInteractionEvent | None]:
    kind = _interaction_kind(example)
    source_frame_count = example.item.targets.event_targets.shape[0]
    output_frame_count = event_probabilities.shape[0]
    anchor_index = output_anchor_index(
        example.anchor_frame_index, source_frame_count, output_frame_count
    )
    if anchor_index is None:
        return None, MissingInteractionEvent(
            event_id=example.item.sample_id,
            interaction_kind=kind,
            reason=MissingInteractionReason.ANCHOR_NOT_EMITTED,
        )
    source_indices = aligned_source_indices(source_frame_count, output_frame_count)
    maximum_offset = round(detection_horizon_seconds / frame_seconds)
    observations: list[InteractionObservation] = []
    for output_index in range(
        anchor_index, min(output_frame_count, anchor_index + maximum_offset + 1)
    ):
        if not bool(frame_mask[output_index]):
            continue
        source_index = int(source_indices[output_index])
        observations.append(
            InteractionObservation(
                elapsed_seconds=(output_index - anchor_index) * frame_seconds,
                non_floor_feedback_probability=float(event_probabilities[output_index, 3]),
                floor_take_probability=float(event_probabilities[output_index, 4]),
                backchannel_active=(
                    bool(example.item.targets.event_mask[source_index, 3])
                    and float(example.item.targets.event_targets[source_index, 3]) >= 0.5
                ),
            )
        )
    if not observations:
        return None, MissingInteractionEvent(
            event_id=example.item.sample_id,
            interaction_kind=kind,
            reason=MissingInteractionReason.NO_VALID_OBSERVATION,
        )
    return (
        InteractionEventPrediction(
            event_id=example.item.sample_id,
            interaction_kind=kind,
            observations=tuple(observations),
        ),
        None,
    )


def _interaction_kind(example: AnchoredSyntheticTrainingItem) -> InteractionKind:
    match example.anchor:
        case BackchannelAnchor():
            return InteractionKind.BACKCHANNEL
        case InterruptionAnchor():
            return InteractionKind.INTERRUPTION
        case _:
            raise ValueError(
                "Interaction evaluation accepts only backchannel/interruption anchors."
            )


def _align_assistant_speaking(values: torch.Tensor, frame_count: int) -> torch.Tensor:
    indices = (
        torch.linspace(0, values.shape[1] - 1, frame_count, device=values.device).round().long()
    )
    return values[:, indices]
