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

from app.training.turn_taking.schema import TurnTakingSample

EVENT_CLASS_COUNT = 5
FUTURE_ACTIVITY_BIN_COUNT = 4


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
    yield_probability: Tensor
    primary_weight: Tensor
    primary_mask: Tensor
    event_distribution: Tensor
    event_weight: Tensor
    event_mask: Tensor
    future_activity: Tensor
    future_activity_mask: Tensor


@dataclass(frozen=True)
class TrainingItem:
    sample_id: str
    waveform: Tensor
    assistant_speaking: Tensor
    targets: FrameTargets


@dataclass(frozen=True)
class TrainingBatch:
    sample_ids: tuple[str, ...]
    waveforms: Tensor
    waveform_lengths: Tensor
    assistant_speaking: Tensor
    targets: FrameTargets


class TurnTakingDataset(Dataset[TrainingItem]):
    def __init__(
        self,
        samples: Sequence[TurnTakingSample],
        frame_seconds: float,
        burn_in_seconds: float,
        unmeasured_reliability_weight: float,
        augmenter: Callable[[Tensor, random.Random], Tensor] | None,
        random_seed: int,
    ) -> None:
        if frame_seconds <= 0.0:
            raise ValueError("frame_seconds must be positive.")
        if not 0.0 <= unmeasured_reliability_weight <= 1.0:
            raise ValueError("unmeasured_reliability_weight must be between zero and one.")
        self.samples = tuple(samples)
        self.frame_seconds = frame_seconds
        self.burn_in_seconds = burn_in_seconds
        self.unmeasured_reliability_weight = unmeasured_reliability_weight
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
            unmeasured_reliability_weight=self.unmeasured_reliability_weight,
        )
        assistant_speaking = build_assistant_speaking_input(
            sample=sample,
            frame_seconds=self.frame_seconds,
        )
        return TrainingItem(
            sample_id=sample.sample_id,
            waveform=waveform,
            assistant_speaking=assistant_speaking,
            targets=targets,
        )


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
    sample: TurnTakingSample,
    frame_seconds: float,
    burn_in_seconds: float,
    unmeasured_reliability_weight: float,
) -> FrameTargets:
    if not 0.0 <= unmeasured_reliability_weight <= 1.0:
        raise ValueError("unmeasured_reliability_weight must be between zero and one.")
    duration = sample.decision_end_seconds - sample.context_start_seconds
    frame_count = max(1, int(np.ceil(duration / frame_seconds)))
    yield_probability = torch.zeros(frame_count, dtype=torch.float32)
    primary_weight = torch.zeros(frame_count, dtype=torch.float32)
    primary_mask = torch.zeros(frame_count, dtype=torch.bool)
    event_distribution = torch.zeros((frame_count, EVENT_CLASS_COUNT), dtype=torch.float32)
    event_weight = torch.zeros(frame_count, dtype=torch.float32)
    event_mask = torch.zeros(frame_count, dtype=torch.bool)
    future_activity = torch.zeros((frame_count, FUTURE_ACTIVITY_BIN_COUNT), dtype=torch.float32)
    future_activity_mask = torch.zeros((frame_count, FUTURE_ACTIVITY_BIN_COUNT), dtype=torch.bool)
    burn_in_end_seconds = sample.context_start_seconds + burn_in_seconds
    for decision in sample.decisions:
        frame_index = min(
            frame_count - 1,
            round((decision.time_seconds - sample.context_start_seconds) / frame_seconds),
        )
        after_burn_in = decision.time_seconds >= burn_in_end_seconds
        if decision.yield_probability is not None and after_burn_in:
            yield_probability[frame_index] = decision.yield_probability
            primary_weight[frame_index] = _reliability_weight(
                decision.primary_reliability, unmeasured_reliability_weight
            )
            primary_mask[frame_index] = True
        if decision.event_distribution is not None and after_burn_in:
            event_distribution[frame_index] = torch.tensor(
                decision.event_distribution.as_tuple(), dtype=torch.float32
            )
            event_weight[frame_index] = _reliability_weight(
                decision.event_reliability, unmeasured_reliability_weight
            )
            event_mask[frame_index] = True
        if after_burn_in:
            for bin_index, active in enumerate(decision.future_user_activity):
                if active is not None:
                    future_activity[frame_index, bin_index] = float(active)
                    future_activity_mask[frame_index, bin_index] = True
    return FrameTargets(
        yield_probability=yield_probability,
        primary_weight=primary_weight,
        primary_mask=primary_mask,
        event_distribution=event_distribution,
        event_weight=event_weight,
        event_mask=event_mask,
        future_activity=future_activity,
        future_activity_mask=future_activity_mask,
    )


def build_assistant_speaking_input(
    sample: TurnTakingSample,
    frame_seconds: float,
) -> Tensor:
    if frame_seconds <= 0.0:
        raise ValueError("frame_seconds must be positive.")
    duration = sample.decision_end_seconds - sample.context_start_seconds
    frame_count = max(1, int(np.ceil(duration / frame_seconds)))
    times = (
        sample.context_start_seconds
        + torch.arange(frame_count, dtype=torch.float32) * frame_seconds
    )
    assistant_speaking = torch.zeros(frame_count, dtype=torch.bool)
    for span in sample.assistant_speech_spans:
        assistant_speaking |= (times >= span.start_seconds) & (times < span.end_seconds)
    return assistant_speaking


def collate_training_items(items: Sequence[TrainingItem]) -> TrainingBatch:
    return TrainingBatch(
        sample_ids=tuple(item.sample_id for item in items),
        waveforms=pad_sequence([item.waveform for item in items], batch_first=True),
        waveform_lengths=torch.tensor([item.waveform.numel() for item in items], dtype=torch.long),
        assistant_speaking=pad_sequence(
            [item.assistant_speaking for item in items],
            batch_first=True,
            padding_value=False,
        ),
        targets=FrameTargets(
            yield_probability=pad_sequence(
                [item.targets.yield_probability for item in items], batch_first=True
            ),
            primary_weight=pad_sequence(
                [item.targets.primary_weight for item in items], batch_first=True
            ),
            primary_mask=pad_sequence(
                [item.targets.primary_mask for item in items],
                batch_first=True,
                padding_value=False,
            ),
            event_distribution=pad_sequence(
                [item.targets.event_distribution for item in items], batch_first=True
            ),
            event_weight=pad_sequence(
                [item.targets.event_weight for item in items], batch_first=True
            ),
            event_mask=pad_sequence(
                [item.targets.event_mask for item in items],
                batch_first=True,
                padding_value=False,
            ),
            future_activity=pad_sequence(
                [item.targets.future_activity for item in items], batch_first=True
            ),
            future_activity_mask=pad_sequence(
                [item.targets.future_activity_mask for item in items],
                batch_first=True,
                padding_value=False,
            ),
        ),
    )


def _reliability_weight(reliability: float | None, unmeasured_weight: float) -> float:
    return reliability if reliability is not None else unmeasured_weight
