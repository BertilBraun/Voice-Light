from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.asr.transcript import Word
from app.audio.wav import read_mono_wave_audio
from app.quality.vad import frame_rms


@dataclass(frozen=True)
class CrosstalkFilterConfig:
    frame_duration_seconds: float = 0.03
    rolling_radius_seconds: float = 15.0
    quiet_below_baseline_db: float = 12.0
    other_channel_dominance_db: float = 6.0


@dataclass(frozen=True)
class CrosstalkFramePower:
    target: NDArray[np.float32]
    other: NDArray[np.float32]
    frame_duration_seconds: float


@dataclass(frozen=True)
class FrameRange:
    start_index: int
    end_index: int


def load_crosstalk_frame_power(
    target_audio_path: Path,
    other_audio_path: Path,
    config: CrosstalkFilterConfig,
) -> CrosstalkFramePower:
    target_audio = read_mono_wave_audio(target_audio_path)
    other_audio = read_mono_wave_audio(other_audio_path)
    if target_audio.sample_rate != other_audio.sample_rate:
        raise ValueError("Crosstalk filtering requires matching speaker-track sample rates.")
    frame_size = max(1, round(config.frame_duration_seconds * target_audio.sample_rate))
    target_power = frame_power(
        samples=normalized_samples(
            samples=target_audio.samples,
            sample_width=target_audio.sample_width,
        ),
        frame_size=frame_size,
    )
    other_power = frame_power(
        samples=normalized_samples(
            samples=other_audio.samples,
            sample_width=other_audio.sample_width,
        ),
        frame_size=frame_size,
    )
    shared_frame_count = min(len(target_power), len(other_power))
    return CrosstalkFramePower(
        target=target_power[:shared_frame_count],
        other=other_power[:shared_frame_count],
        frame_duration_seconds=frame_size / target_audio.sample_rate,
    )


def normalized_samples(samples: NDArray[np.float64], sample_width: int) -> NDArray[np.float32]:
    maximum_amplitude = float((1 << (sample_width * 8 - 1)) - 1)
    return np.clip(samples / maximum_amplitude, -1.0, 1.0).astype(np.float32)


def frame_power(samples: NDArray[np.float32], frame_size: int) -> NDArray[np.float32]:
    return np.square(frame_rms(samples=samples, frame_size=frame_size)).astype(np.float32)


def filter_crosstalk_words(
    words: tuple[Word, ...],
    frame_power_pair: CrosstalkFramePower,
    config: CrosstalkFilterConfig,
) -> tuple[Word, ...]:
    active_frames = active_frame_mask(
        words=words,
        frame_count=len(frame_power_pair.target),
        frame_duration_seconds=frame_power_pair.frame_duration_seconds,
    )
    return tuple(
        word
        for word in words
        if not should_remove_word(
            word=word,
            active_frames=active_frames,
            frame_power_pair=frame_power_pair,
            config=config,
        )
    )


def active_frame_mask(
    words: tuple[Word, ...],
    frame_count: int,
    frame_duration_seconds: float,
) -> NDArray[np.bool_]:
    active_frames = np.zeros(frame_count, dtype=bool)
    for word in words:
        frame_range = word_frame_range(
            word=word,
            frame_count=frame_count,
            frame_duration_seconds=frame_duration_seconds,
        )
        if frame_range is not None:
            active_frames[frame_range.start_index : frame_range.end_index] = True
    return active_frames


def should_remove_word(
    word: Word,
    active_frames: NDArray[np.bool_],
    frame_power_pair: CrosstalkFramePower,
    config: CrosstalkFilterConfig,
) -> bool:
    frame_range = word_frame_range(
        word=word,
        frame_count=len(frame_power_pair.target),
        frame_duration_seconds=frame_power_pair.frame_duration_seconds,
    )
    if frame_range is None:
        return False
    local_range = seconds_frame_range(
        start_seconds=max(0.0, word.start_seconds - config.rolling_radius_seconds),
        end_seconds=word.end_seconds + config.rolling_radius_seconds,
        frame_count=len(frame_power_pair.target),
        frame_duration_seconds=frame_power_pair.frame_duration_seconds,
    )
    local_active_frames = active_frames[local_range.start_index : local_range.end_index]
    local_target_power = frame_power_pair.target[local_range.start_index : local_range.end_index][
        local_active_frames
    ]
    if len(local_target_power) == 0:
        return False
    target_word_db = power_db(
        frame_power_pair.target[frame_range.start_index : frame_range.end_index]
    )
    other_word_db = power_db(
        frame_power_pair.other[frame_range.start_index : frame_range.end_index]
    )
    local_baseline_db = power_db(local_target_power)
    is_quiet_outlier = target_word_db <= local_baseline_db - config.quiet_below_baseline_db
    is_other_channel_crosstalk = (
        target_word_db < local_baseline_db
        and other_word_db >= target_word_db + config.other_channel_dominance_db
    )
    return is_quiet_outlier or is_other_channel_crosstalk


def word_frame_range(
    word: Word,
    frame_count: int,
    frame_duration_seconds: float,
) -> FrameRange | None:
    if word.start_seconds is None or word.end_seconds is None:
        return None
    audio_duration_seconds = frame_count * frame_duration_seconds
    if word.end_seconds <= 0.0 or word.start_seconds >= audio_duration_seconds:
        return None
    return seconds_frame_range(
        start_seconds=word.start_seconds,
        end_seconds=word.end_seconds,
        frame_count=frame_count,
        frame_duration_seconds=frame_duration_seconds,
    )


def seconds_frame_range(
    start_seconds: float,
    end_seconds: float,
    frame_count: int,
    frame_duration_seconds: float,
) -> FrameRange:
    start_index = min(frame_count, max(0, int(start_seconds / frame_duration_seconds)))
    end_index = min(
        frame_count,
        max(start_index + 1, int(np.ceil(end_seconds / frame_duration_seconds))),
    )
    return FrameRange(start_index=start_index, end_index=end_index)


def power_db(power: NDArray[np.float32]) -> float:
    return float(10.0 * np.log10(max(float(np.mean(power)), 1e-12)))
