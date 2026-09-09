from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from app.training.turn_taking.config import TrainingConfig
from app.training.turn_taking.model import IncrementalAdapterState, TurnTakingAdapter


@dataclass(frozen=True)
class TurnAdapterProbabilities:
    turn_completion: float
    continuation_pause: float
    non_floor_feedback: float
    floor_take: float
    user_yield: float
    future_activity: tuple[float, float, float, float]


@dataclass(frozen=True)
class LoadedStreamingTurnAdapter:
    checkpoint_sha256: str
    optimizer_step: int
    training_config: TrainingConfig
    adapter: TurnTakingAdapter


def load_streaming_turn_adapter(
    checkpoint_path: Path,
    expected_model_identifier: str,
    expected_model_revision: str,
    expected_lookahead_tokens: int,
    device: torch.device,
) -> LoadedStreamingTurnAdapter:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_config = TrainingConfig.model_validate(payload["training_config"])
    if training_config.model_identifier != expected_model_identifier:
        raise ValueError("Turn adapter checkpoint uses a different encoder model.")
    if training_config.model_revision != expected_model_revision:
        raise ValueError("Turn adapter checkpoint uses a different encoder revision.")
    if training_config.lookahead_tokens != expected_lookahead_tokens:
        raise ValueError("Turn adapter checkpoint uses a different lookahead configuration.")
    adapter = TurnTakingAdapter(training_config.adapter)
    adapter.load_state_dict(payload["adapter_state"], strict=True)
    adapter.to(device).eval()
    adapter.recurrent.flatten_parameters()
    return LoadedStreamingTurnAdapter(
        checkpoint_sha256=_file_sha256(checkpoint_path),
        optimizer_step=int(payload["optimizer_step"]),
        training_config=training_config,
        adapter=adapter,
    )


class StreamingTurnAdapter:
    def __init__(self, loaded: LoadedStreamingTurnAdapter) -> None:
        self.loaded = loaded
        self.state: IncrementalAdapterState | None = None

    @property
    def tap_layer_indices(self) -> tuple[int, ...]:
        return self.loaded.training_config.adapter.tap_layer_indices

    @property
    def checkpoint_sha256(self) -> str:
        return self.loaded.checkpoint_sha256

    def reset(self) -> None:
        self.state = None

    def predict(
        self,
        feature_taps: tuple[Tensor, ...],
        assistant_speaking: bool,
    ) -> TurnAdapterProbabilities:
        parameter = next(self.loaded.adapter.parameters())
        aligned_feature_taps = tuple(
            features.to(device=parameter.device, dtype=parameter.dtype) for features in feature_taps
        )
        frame_count = aligned_feature_taps[0].shape[1]
        assistant_condition = aligned_feature_taps[0].new_full(
            (feature_taps[0].shape[0], frame_count),
            float(assistant_speaking),
        )
        with torch.no_grad():
            output, self.state = self.loaded.adapter.forward_incremental(
                aligned_feature_taps,
                assistant_condition,
                self.state,
            )
        event_probabilities = output.event_logits[0, -1].sigmoid().float().cpu()
        future_probabilities = output.future_activity_logits[0, -1].sigmoid().float().cpu()
        return TurnAdapterProbabilities(
            turn_completion=float(event_probabilities[0]),
            continuation_pause=float(event_probabilities[1]),
            non_floor_feedback=float(event_probabilities[3]),
            floor_take=float(event_probabilities[4]),
            user_yield=float(output.yield_logits[0, -1].sigmoid().float().cpu()),
            future_activity=(
                float(future_probabilities[0]),
                float(future_probabilities[1]),
                float(future_probabilities[2]),
                float(future_probabilities[3]),
            ),
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
