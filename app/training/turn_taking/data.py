from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from app.training.turn_taking.schema import EventType, PolicyClass, TurnEvent, TurnTakingSample

POLICY_INDEX = {policy: index for index, policy in enumerate(PolicyClass)}
EVENT_TARGETS = (
    EventType.BACKCHANNEL,
    EventType.INTERRUPTION,
    EventType.COOPERATIVE_OVERLAP,
    EventType.COMPETITIVE_OVERLAP,
    EventType.LAUGHTER,
    EventType.NONSPEECH_VOCALIZATION,
)
FUTURE_WINDOWS_SECONDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0))
END_HORIZONS_SECONDS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class WaveformAugmenter:
    gain_probability: float = 0.8
    noise_probability: float = 0.3
    dropout_probability: float = 0.1

    def __call__(self, waveform: Tensor, generator: random.Random) -> Tensor:
        augmented = waveform.clone()
        if generator.random() < self.gain_probability:
            gain_db = generator.uniform(-12.0, 6.0)
            augmented *= 10.0 ** (gain_db / 20.0)
        if generator.random() < self.noise_probability:
            signal_power = augmented.square().mean().clamp_min(1e-8)
            signal_to_noise_db = generator.uniform(5.0, 30.0)
            noise_power = signal_power / (10.0 ** (signal_to_noise_db / 10.0))
            augmented += torch.randn_like(augmented) * noise_power.sqrt()
        if augmented.numel() and generator.random() < self.dropout_probability:
            width = max(1, int(0.04 * augmented.numel()))
            start = generator.randrange(max(1, augmented.numel() - width + 1))
            augmented[start : start + width] = 0.0
        return augmented.clamp(-1.0, 1.0)


@dataclass(frozen=True)
class FrameTargets:
    policy: Tensor
    future_activity: Tensor
    end_horizons: Tensor
    time_to_shift_bucket: Tensor
    events: Tensor
    confidence: Tensor
    loss_mask: Tensor


@dataclass(frozen=True)
class TrainingItem:
    sample_id: str
    waveform: Tensor
    targets: FrameTargets


@dataclass(frozen=True)
class TrainingBatch:
    sample_ids: tuple[str, ...]
    waveforms: Tensor
    waveform_lengths: Tensor
    targets: FrameTargets


class TurnTakingDataset(Dataset[TrainingItem]):
    def __init__(
        self,
        samples: Sequence[TurnTakingSample],
        frame_seconds: float,
        burn_in_seconds: float,
        augmenter: Callable[[Tensor, random.Random], Tensor] | None,
        random_seed: int,
    ) -> None:
        if frame_seconds <= 0.0:
            raise ValueError("frame_seconds must be positive.")
        self.samples = tuple(samples)
        self.frame_seconds = frame_seconds
        self.burn_in_seconds = burn_in_seconds
        self.augmenter = augmenter
        self.generator = random.Random(random_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingItem:
        sample = self.samples[index]
        waveform = load_audio_window(
            path=sample.target_audio_path,
            sample_rate_hz=sample.sample_rate_hz,
            start_seconds=sample.context_start_seconds,
            end_seconds=sample.decision_end_seconds,
        )
        if self.augmenter is not None:
            waveform = self.augmenter(waveform, self.generator)
        targets = build_frame_targets(
            sample=sample,
            frame_seconds=self.frame_seconds,
            burn_in_seconds=self.burn_in_seconds,
        )
        return TrainingItem(sample_id=sample.sample_id, waveform=waveform, targets=targets)


def load_audio_window(
    path: Path, sample_rate_hz: int, start_seconds: float, end_seconds: float
) -> Tensor:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"Audio stream not found: {path}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate_hz)
        parts = [
            frame.to_ndarray().reshape(-1)
            for decoded in container.decode(stream)
            for frame in resampler.resample(decoded)
        ]
        parts.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(None))
    audio = (
        np.concatenate(parts).astype(np.float32, copy=False) if parts else np.empty(0, np.float32)
    )
    start_sample = round(start_seconds * sample_rate_hz)
    end_sample = round(end_seconds * sample_rate_hz)
    return torch.from_numpy(audio[start_sample:end_sample].copy())


def build_frame_targets(
    sample: TurnTakingSample, frame_seconds: float, burn_in_seconds: float
) -> FrameTargets:
    duration = sample.decision_end_seconds - sample.context_start_seconds
    frame_count = max(1, int(np.ceil(duration / frame_seconds)))
    times = (
        sample.context_start_seconds
        + torch.arange(frame_count, dtype=torch.float32) * frame_seconds
    )
    policy = torch.full((frame_count,), POLICY_INDEX[PolicyClass.WAIT], dtype=torch.long)
    confidence = torch.ones(frame_count, dtype=torch.float32)
    for event in sample.events:
        event_mask = (times >= event.start_seconds) & (times < event.end_seconds)
        policy_class = _policy_for_event(event.event_type)
        if policy_class is not None:
            policy[event_mask] = POLICY_INDEX[policy_class]
            confidence[event_mask] = event.confidence
    future_activity = torch.stack(
        [
            _future_activity(times, sample.events, event_type, window_start, window_end)
            for event_type in (EventType.TARGET_SPEECH, EventType.PARTNER_SPEECH)
            for window_start, window_end in FUTURE_WINDOWS_SECONDS
        ],
        dim=-1,
    )
    shifts = tuple(event for event in sample.events if event.event_type is EventType.TURN_SHIFT)
    time_to_shift = _time_to_next_shift(times, shifts)
    end_horizons = torch.stack(
        [(time_to_shift >= 0.0) & (time_to_shift <= horizon) for horizon in END_HORIZONS_SECONDS],
        dim=-1,
    ).float()
    time_bucket = torch.bucketize(time_to_shift.clamp_min(0.0), torch.tensor([0.25, 0.5, 1.0, 2.0]))
    event_targets = torch.stack(
        [_active_at(times, sample.events, event_type) for event_type in EVENT_TARGETS], dim=-1
    )
    burn_in_end = sample.context_start_seconds + burn_in_seconds
    loss_mask = (times >= max(burn_in_end, sample.decision_start_seconds)) & (
        times < sample.decision_end_seconds
    )
    return FrameTargets(
        policy=policy,
        future_activity=future_activity,
        end_horizons=end_horizons,
        time_to_shift_bucket=time_bucket,
        events=event_targets,
        confidence=confidence,
        loss_mask=loss_mask,
    )


def collate_training_items(items: Sequence[TrainingItem]) -> TrainingBatch:
    waveforms = pad_sequence([item.waveform for item in items], batch_first=True)
    waveform_lengths = torch.tensor([item.waveform.numel() for item in items], dtype=torch.long)
    return TrainingBatch(
        sample_ids=tuple(item.sample_id for item in items),
        waveforms=waveforms,
        waveform_lengths=waveform_lengths,
        targets=FrameTargets(
            policy=_pad_targets(items, "policy", POLICY_INDEX[PolicyClass.WAIT]),
            future_activity=_pad_targets(items, "future_activity", 0.0),
            end_horizons=_pad_targets(items, "end_horizons", 0.0),
            time_to_shift_bucket=_pad_targets(items, "time_to_shift_bucket", 4),
            events=_pad_targets(items, "events", 0.0),
            confidence=_pad_targets(items, "confidence", 0.0),
            loss_mask=_pad_targets(items, "loss_mask", False),
        ),
    )


def _pad_targets(
    items: Sequence[TrainingItem], field_name: str, padding_value: float | int | bool
) -> Tensor:
    tensors: list[Tensor] = []
    for item in items:
        match field_name:
            case "policy":
                tensor = item.targets.policy
            case "future_activity":
                tensor = item.targets.future_activity
            case "end_horizons":
                tensor = item.targets.end_horizons
            case "time_to_shift_bucket":
                tensor = item.targets.time_to_shift_bucket
            case "events":
                tensor = item.targets.events
            case "confidence":
                tensor = item.targets.confidence
            case "loss_mask":
                tensor = item.targets.loss_mask
            case _:
                raise ValueError(f"Unknown target field: {field_name}")
        tensors.append(tensor)
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value)


def _policy_for_event(event_type: EventType) -> PolicyClass | None:
    match event_type:
        case EventType.TURN_SHIFT:
            return PolicyClass.TAKE_TURN
        case EventType.HOLD:
            return PolicyClass.WAIT
        case EventType.BACKCHANNEL:
            return PolicyClass.MAY_BACKCHANNEL
        case EventType.INTERRUPTION:
            return PolicyClass.YIELD
        case _:
            return None


def _active_at(times: Tensor, events: Sequence[TurnEvent], event_type: EventType) -> Tensor:
    active = torch.zeros_like(times)
    for event in events:
        if event.event_type is event_type:
            active = torch.maximum(
                active, ((times >= event.start_seconds) & (times < event.end_seconds)).float()
            )
    return active


def _future_activity(
    times: Tensor,
    events: Sequence[TurnEvent],
    event_type: EventType,
    window_start: float,
    window_end: float,
) -> Tensor:
    midpoint = times + (window_start + window_end) / 2.0
    return _active_at(midpoint, events, event_type)


def _time_to_next_shift(times: Tensor, shifts: Sequence[TurnEvent]) -> Tensor:
    result = torch.full_like(times, 3.0)
    for shift in shifts:
        delta = shift.start_seconds - times
        result = torch.where((delta >= 0.0) & (delta < result), delta, result)
    return result
