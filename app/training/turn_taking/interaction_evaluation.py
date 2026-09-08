from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum

import torch
from pydantic import Field, model_validator
from torch import Tensor

from app.shared.base_model import FrozenBaseModel


class InteractionKind(StrEnum):
    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"


class InteractionDecision(StrEnum):
    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"
    UNRESOLVED = "unresolved"


class MissingInteractionReason(StrEnum):
    ANCHOR_NOT_EMITTED = "anchor_not_emitted"
    NO_VALID_OBSERVATION = "no_valid_observation"


class InteractionObservation(FrozenBaseModel):
    elapsed_seconds: float = Field(ge=0.0)
    non_floor_feedback_probability: float = Field(ge=0.0, le=1.0)
    floor_take_probability: float = Field(ge=0.0, le=1.0)
    backchannel_active: bool


class InteractionEventPrediction(FrozenBaseModel):
    event_id: str
    interaction_kind: InteractionKind
    observations: tuple[InteractionObservation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_observation_times(self) -> InteractionEventPrediction:
        times = tuple(observation.elapsed_seconds for observation in self.observations)
        if times != tuple(sorted(set(times))):
            raise ValueError("Interaction observations must have unique increasing times.")
        if self.interaction_kind is InteractionKind.BACKCHANNEL and not any(
            observation.backchannel_active for observation in self.observations
        ):
            raise ValueError("Backchannel predictions must cover an active backchannel frame.")
        return self


class MissingInteractionEvent(FrozenBaseModel):
    event_id: str
    interaction_kind: InteractionKind
    reason: MissingInteractionReason


class InteractionPredictionManifest(FrozenBaseModel):
    schema_version: str = "voice-light-interaction-predictions-v1"
    generated_at: datetime
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    optimizer_step: int = Field(gt=0)
    source_roots: tuple[str, ...] = Field(min_length=1)
    split: str
    frame_seconds: float = Field(gt=0.0)
    detection_horizon_seconds: float = Field(gt=0.0)
    event_offset: int = Field(default=0, ge=0)
    eligible_event_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    missing_event_count: int = Field(ge=0)
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InteractionPredictionArtifact(FrozenBaseModel):
    manifest: InteractionPredictionManifest
    predictions: tuple[InteractionEventPrediction, ...]
    missing_events: tuple[MissingInteractionEvent, ...]

    @model_validator(mode="after")
    def validate_counts_and_hash(self) -> InteractionPredictionArtifact:
        if len(self.predictions) != self.manifest.prediction_count:
            raise ValueError("Interaction prediction count does not match its manifest.")
        if len(self.missing_events) != self.manifest.missing_event_count:
            raise ValueError("Missing interaction count does not match its manifest.")
        if len(self.predictions) + len(self.missing_events) != self.manifest.eligible_event_count:
            raise ValueError("Interaction artifact does not account for every eligible event.")
        if interaction_predictions_sha256(self.predictions) != self.manifest.predictions_sha256:
            raise ValueError("Interaction predictions do not match their manifest hash.")
        return self


class ConfusionCount(FrozenBaseModel):
    actual: InteractionKind
    predicted: InteractionDecision
    count: int = Field(ge=0)


class LatencyMetrics(FrozenBaseModel):
    eligible_count: int = Field(ge=0)
    observed_count: int = Field(ge=0)
    censored_count: int = Field(ge=0)
    median_seconds: float | None = Field(default=None, ge=0.0)
    p95_seconds: float | None = Field(default=None, ge=0.0)


class MissingReasonCount(FrozenBaseModel):
    reason: MissingInteractionReason
    count: int = Field(ge=0)


class InteractionEvaluationReport(FrozenBaseModel):
    schema_version: str = "voice-light-interaction-evaluation-v1"
    threshold: float = Field(ge=0.0, le=1.0)
    eligible_event_count: int = Field(ge=0)
    evaluated_event_count: int = Field(ge=0)
    missing_event_count: int = Field(ge=0)
    missing_reasons: tuple[MissingReasonCount, ...]
    backchannel_count: int = Field(ge=0)
    backchannel_false_cancel_count: int = Field(ge=0)
    backchannel_false_cancel_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    backchannel_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    interruption_count: int = Field(ge=0)
    interruption_floor_take_count: int = Field(ge=0)
    interruption_floor_take_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    confusion_matrix: tuple[ConfusionCount, ...]
    backchannel_decision_latency: LatencyMetrics
    interruption_floor_take_latency: LatencyMetrics


def evaluate_interaction_predictions(
    artifact: InteractionPredictionArtifact,
    threshold: float,
) -> InteractionEvaluationReport:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Interaction threshold must be between zero and one.")
    decisions = tuple(
        (_decision(prediction, threshold), prediction) for prediction in artifact.predictions
    )
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
    false_cancel_count = sum(_false_cancelled(prediction, threshold) for prediction in backchannels)
    backchannel_latencies = tuple(
        latency
        for prediction in backchannels
        if (latency := _first_crossing(prediction, threshold, InteractionKind.BACKCHANNEL))
        is not None
    )
    interruption_latencies = tuple(
        latency
        for prediction in interruptions
        if (latency := _first_crossing(prediction, threshold, InteractionKind.INTERRUPTION))
        is not None
    )
    confusion = tuple(
        ConfusionCount(
            actual=actual,
            predicted=predicted,
            count=sum(
                prediction.interaction_kind is actual and decision is predicted
                for decision, prediction in decisions
            ),
        )
        for actual in InteractionKind
        for predicted in InteractionDecision
    )
    missing_reasons = tuple(
        MissingReasonCount(
            reason=reason,
            count=sum(event.reason is reason for event in artifact.missing_events),
        )
        for reason in MissingInteractionReason
    )
    backchannel_correct = sum(
        decision is InteractionDecision.BACKCHANNEL
        for decision, prediction in decisions
        if prediction.interaction_kind is InteractionKind.BACKCHANNEL
    )
    interruption_correct = len(interruption_latencies)
    return InteractionEvaluationReport(
        threshold=threshold,
        eligible_event_count=artifact.manifest.eligible_event_count,
        evaluated_event_count=len(artifact.predictions),
        missing_event_count=len(artifact.missing_events),
        missing_reasons=missing_reasons,
        backchannel_count=len(backchannels),
        backchannel_false_cancel_count=false_cancel_count,
        backchannel_false_cancel_rate=_rate(false_cancel_count, len(backchannels)),
        backchannel_recall=_rate(backchannel_correct, len(backchannels)),
        interruption_count=len(interruptions),
        interruption_floor_take_count=interruption_correct,
        interruption_floor_take_recall=_rate(interruption_correct, len(interruptions)),
        confusion_matrix=confusion,
        backchannel_decision_latency=_latency_metrics(len(backchannels), backchannel_latencies),
        interruption_floor_take_latency=_latency_metrics(
            len(interruptions), interruption_latencies
        ),
    )


def interaction_predictions_sha256(
    predictions: Sequence[InteractionEventPrediction],
) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        digest.update(prediction.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def merge_interaction_prediction_artifacts(
    artifacts: Sequence[InteractionPredictionArtifact],
) -> InteractionPredictionArtifact:
    if not artifacts:
        raise ValueError("Interaction prediction merge requires at least one artifact.")
    first = artifacts[0].manifest
    for artifact in artifacts[1:]:
        manifest = artifact.manifest
        if (
            manifest.checkpoint_sha256 != first.checkpoint_sha256
            or manifest.optimizer_step != first.optimizer_step
            or manifest.split != first.split
            or manifest.frame_seconds != first.frame_seconds
            or manifest.detection_horizon_seconds != first.detection_horizon_seconds
        ):
            raise ValueError("Interaction prediction artifacts use incompatible configurations.")
    predictions = tuple(
        sorted(
            (prediction for artifact in artifacts for prediction in artifact.predictions),
            key=lambda prediction: prediction.event_id,
        )
    )
    missing_events = tuple(
        sorted(
            (event for artifact in artifacts for event in artifact.missing_events),
            key=lambda event: event.event_id,
        )
    )
    event_ids = tuple(prediction.event_id for prediction in predictions) + tuple(
        event.event_id for event in missing_events
    )
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Interaction prediction artifacts contain duplicate event IDs.")
    source_roots = tuple(
        dict.fromkeys(
            source_root for artifact in artifacts for source_root in artifact.manifest.source_roots
        )
    )
    return InteractionPredictionArtifact(
        manifest=InteractionPredictionManifest(
            generated_at=datetime.now(UTC),
            checkpoint_sha256=first.checkpoint_sha256,
            optimizer_step=first.optimizer_step,
            source_roots=source_roots,
            split=first.split,
            frame_seconds=first.frame_seconds,
            detection_horizon_seconds=first.detection_horizon_seconds,
            event_offset=0,
            eligible_event_count=sum(
                artifact.manifest.eligible_event_count for artifact in artifacts
            ),
            prediction_count=len(predictions),
            missing_event_count=len(missing_events),
            predictions_sha256=interaction_predictions_sha256(predictions),
        ),
        predictions=predictions,
        missing_events=missing_events,
    )


def aligned_source_indices(source_frame_count: int, output_frame_count: int) -> Tensor:
    if source_frame_count <= 0 or output_frame_count <= 0:
        raise ValueError("Frame counts must be positive.")
    return torch.linspace(0, source_frame_count - 1, output_frame_count).round().long()


def output_anchor_index(
    source_anchor_index: int,
    source_frame_count: int,
    output_frame_count: int,
) -> int | None:
    if not 0 <= source_anchor_index < source_frame_count:
        raise ValueError("Source anchor index falls outside the target frames.")
    matches = torch.nonzero(
        aligned_source_indices(source_frame_count, output_frame_count) == source_anchor_index,
        as_tuple=False,
    ).flatten()
    return int(matches[0]) if matches.numel() else None


def _decision(
    prediction: InteractionEventPrediction,
    threshold: float,
) -> InteractionDecision:
    for observation in prediction.observations:
        backchannel = observation.non_floor_feedback_probability >= threshold
        interruption = observation.floor_take_probability >= threshold
        if backchannel and interruption:
            return (
                InteractionDecision.BACKCHANNEL
                if observation.non_floor_feedback_probability >= observation.floor_take_probability
                else InteractionDecision.INTERRUPTION
            )
        if backchannel:
            return InteractionDecision.BACKCHANNEL
        if interruption:
            return InteractionDecision.INTERRUPTION
    return InteractionDecision.UNRESOLVED


def _false_cancelled(prediction: InteractionEventPrediction, threshold: float) -> bool:
    return any(
        observation.backchannel_active and observation.floor_take_probability >= threshold
        for observation in prediction.observations
    )


def _first_crossing(
    prediction: InteractionEventPrediction,
    threshold: float,
    kind: InteractionKind,
) -> float | None:
    for observation in prediction.observations:
        probability = (
            observation.non_floor_feedback_probability
            if kind is InteractionKind.BACKCHANNEL
            else observation.floor_take_probability
        )
        if probability >= threshold:
            return observation.elapsed_seconds
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


def _quantile(values: tuple[float, ...], probability: float) -> float | None:
    if not values:
        return None
    index = math.ceil(probability * len(values)) - 1
    return values[max(0, index)]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
