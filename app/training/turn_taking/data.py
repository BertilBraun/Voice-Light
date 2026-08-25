from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from app.training.turn_taking.config import WaveformAugmentationConfig
from app.training.turn_taking.schema import TurnTakingSample

EVENT_CLASS_COUNT = 5
FUTURE_ACTIVITY_BIN_COUNT = 4
SEEK_PREROLL_SECONDS = 1.0


@dataclass(frozen=True)
class WaveformAugmenter:
    config: WaveformAugmentationConfig
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")

    def __call__(self, waveform: Tensor, generator: random.Random) -> Tensor:
        augmented = waveform.clone()
        if generator.random() < self.config.gain_probability:
            gain_db = generator.uniform(
                self.config.minimum_gain_db,
                self.config.maximum_gain_db,
            )
            augmented *= 10.0 ** (gain_db / 20.0)
        if generator.random() < self.config.noise_probability:
            signal_power = augmented.square().mean().clamp_min(1e-8)
            signal_to_noise_db = generator.uniform(
                self.config.minimum_signal_to_noise_db,
                self.config.maximum_signal_to_noise_db,
            )
            noise_power = signal_power / (10.0 ** (signal_to_noise_db / 10.0))
            noise_generator = torch.Generator().manual_seed(generator.randrange(2**63))
            noise = torch.randn(
                augmented.shape,
                dtype=augmented.dtype,
                device=augmented.device,
                generator=noise_generator,
            )
            augmented += noise * noise_power.sqrt()
        if augmented.numel() and generator.random() < self.config.reverberation_probability:
            augmented = _add_synthetic_reverberation(
                augmented,
                self.sample_rate_hz,
                generator,
            )
        if augmented.numel() and generator.random() < self.config.bandwidth_probability:
            augmented = _limit_bandwidth(augmented, self.sample_rate_hz, generator)
        if augmented.numel() and generator.random() < self.config.clipping_probability:
            threshold = generator.uniform(0.35, 0.9)
            augmented = augmented.clamp(-threshold, threshold) / threshold
        if augmented.numel() and generator.random() < self.config.packet_loss_probability:
            duration_seconds = generator.uniform(
                self.config.minimum_packet_loss_seconds,
                self.config.maximum_packet_loss_seconds,
            )
            width = min(augmented.numel(), max(1, round(duration_seconds * self.sample_rate_hz)))
            start = generator.randrange(augmented.numel() - width + 1)
            augmented[start : start + width] = 0.0
        return augmented.clamp(-1.0, 1.0)


def _add_synthetic_reverberation(
    waveform: Tensor,
    sample_rate_hz: int,
    generator: random.Random,
) -> Tensor:
    wet = generator.uniform(0.1, 0.35)
    reverberant = waveform.clone()
    for tap_index in range(generator.randint(2, 4)):
        delay_seconds = generator.uniform(0.015, 0.08)
        delay_samples = min(waveform.numel() - 1, round(delay_seconds * sample_rate_hz))
        decay = wet * generator.uniform(0.45, 0.85) ** tap_index
        reverberant[delay_samples:] += decay * waveform[: waveform.numel() - delay_samples]
    return reverberant / (1.0 + wet)


def _limit_bandwidth(
    waveform: Tensor,
    sample_rate_hz: int,
    generator: random.Random,
) -> Tensor:
    target_sample_rate_hz = generator.choice((8_000, 12_000))
    if target_sample_rate_hz >= sample_rate_hz:
        return waveform
    reduced_length = max(1, round(waveform.numel() * target_sample_rate_hz / sample_rate_hz))
    batched = waveform.reshape(1, 1, -1)
    reduced = functional.interpolate(
        batched,
        size=reduced_length,
        mode="linear",
        align_corners=False,
    )
    return functional.interpolate(
        reduced,
        size=waveform.numel(),
        mode="linear",
        align_corners=False,
    ).reshape(-1)


@dataclass(frozen=True)
class FrameTargets:
    yield_probability: Tensor
    primary_weight: Tensor
    primary_mask: Tensor
    event_targets: Tensor
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
    path: Path,
    sample_rate_hz: int,
    start_seconds: float,
    end_seconds: float,
    pad_missing_suffix: bool = False,
) -> Tensor:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if start_seconds < 0.0 or end_seconds <= start_seconds:
        raise ValueError("Audio window must satisfy 0 <= start < end.")
    start_sample = round(start_seconds * sample_rate_hz)
    end_sample = round(end_seconds * sample_rate_hz)
    output = np.zeros(end_sample - start_sample, dtype=np.float32)
    covered = np.zeros(output.size, dtype=np.bool_)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"Audio stream not found: {path}")
        stream = container.streams.audio[0]
        if stream.time_base is None:
            raise ValueError(f"Audio stream has no time base: {path}")
        stream_start_pts = stream.start_time or 0
        seek_seconds = max(0.0, start_seconds - SEEK_PREROLL_SECONDS)
        seek_pts = stream_start_pts + int(seek_seconds / float(stream.time_base))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate_hz)
        reached_window_end = False
        for decoded in container.decode(stream):
            for frame in resampler.resample(decoded):
                reached_window_end = _copy_resampled_frame(
                    frame=frame,
                    stream_start_pts=stream_start_pts,
                    stream_time_base=stream.time_base,
                    sample_rate_hz=sample_rate_hz,
                    window_start_sample=start_sample,
                    window_end_sample=end_sample,
                    output=output,
                    covered=covered,
                )
                if reached_window_end:
                    break
            if reached_window_end:
                break
        if not reached_window_end:
            for frame in resampler.resample(None):
                _copy_resampled_frame(
                    frame=frame,
                    stream_start_pts=stream_start_pts,
                    stream_time_base=stream.time_base,
                    sample_rate_hz=sample_rate_hz,
                    window_start_sample=start_sample,
                    window_end_sample=end_sample,
                    output=output,
                    covered=covered,
                )
    if not covered.all():
        missing_count = int((~covered).sum())
        first_missing_index = int(np.flatnonzero(~covered)[0])
        missing_is_suffix = bool(
            covered[:first_missing_index].all() and (~covered[first_missing_index:]).all()
        )
        if pad_missing_suffix and covered.any() and missing_is_suffix:
            return torch.from_numpy(output)
        raise ValueError(f"Audio window {path} is missing {missing_count} decoded samples.")
    return torch.from_numpy(output)


def _copy_resampled_frame(
    frame: av.AudioFrame,
    stream_start_pts: int,
    stream_time_base: Fraction,
    sample_rate_hz: int,
    window_start_sample: int,
    window_end_sample: int,
    output: np.ndarray,
    covered: np.ndarray,
) -> bool:
    if frame.pts is None or frame.time_base is None:
        raise ValueError("Resampled audio frame has no timestamp.")
    stream_start_seconds = float(stream_start_pts * stream_time_base)
    frame_start_seconds = float(frame.pts * frame.time_base) - stream_start_seconds
    frame_start_sample = round(frame_start_seconds * sample_rate_hz)
    samples = frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
    frame_end_sample = frame_start_sample + samples.size
    copy_start_sample = max(frame_start_sample, window_start_sample)
    copy_end_sample = min(frame_end_sample, window_end_sample)
    if copy_start_sample < copy_end_sample:
        source_start = copy_start_sample - frame_start_sample
        source_end = copy_end_sample - frame_start_sample
        output_start = copy_start_sample - window_start_sample
        output_end = copy_end_sample - window_start_sample
        output[output_start:output_end] = samples[source_start:source_end]
        covered[output_start:output_end] = True
    return frame_start_sample >= window_end_sample


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
    times = (
        sample.context_start_seconds
        + torch.arange(frame_count, dtype=torch.float32) * frame_seconds
    )
    supervision_start_seconds = max(
        sample.context_start_seconds + burn_in_seconds,
        sample.decision_start_seconds,
    )
    supervised_frames = times >= supervision_start_seconds
    yield_probability = torch.zeros(frame_count, dtype=torch.float32)
    primary_weight = torch.zeros(frame_count, dtype=torch.float32)
    primary_mask = torch.zeros(frame_count, dtype=torch.bool)
    event_targets = torch.zeros((frame_count, EVENT_CLASS_COUNT), dtype=torch.float32)
    event_mask = torch.zeros((frame_count, EVENT_CLASS_COUNT), dtype=torch.bool)
    future_activity = torch.zeros((frame_count, FUTURE_ACTIVITY_BIN_COUNT), dtype=torch.float32)
    future_activity_mask = torch.zeros((frame_count, FUTURE_ACTIVITY_BIN_COUNT), dtype=torch.bool)
    burn_in_end_seconds = sample.context_start_seconds + burn_in_seconds
    for decision in sample.decisions:
        frame_index = min(
            frame_count - 1,
            round((decision.time_seconds - sample.context_start_seconds) / frame_seconds),
        )
        after_burn_in = decision.time_seconds >= burn_in_end_seconds
        if after_burn_in:
            yield_probability[frame_index] = decision.yield_probability
            primary_weight[frame_index] = _reliability_weight(
                decision.primary_reliability, unmeasured_reliability_weight
            )
            primary_mask[frame_index] = True
        if decision.event_distribution is not None and after_burn_in:
            event_targets[frame_index] = torch.tensor(
                decision.event_distribution.as_tuple(), dtype=torch.float32
            )
            event_mask[frame_index] = True
        if after_burn_in:
            for bin_index, active in enumerate(decision.future_user_activity):
                if active is not None:
                    future_activity[frame_index, bin_index] = float(active)
                    future_activity_mask[frame_index, bin_index] = True
    missing_primary_frames = supervised_frames & ~primary_mask
    if missing_primary_frames.any():
        missing_count = int(missing_primary_frames.sum().item())
        raise ValueError(
            f"Sample {sample.sample_id} is missing dense HOLD/YIELD labels for "
            f"{missing_count} supervised frames."
        )
    return FrameTargets(
        yield_probability=yield_probability,
        primary_weight=primary_weight,
        primary_mask=primary_mask,
        event_targets=event_targets,
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
            event_targets=pad_sequence(
                [item.targets.event_targets for item in items], batch_first=True
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
