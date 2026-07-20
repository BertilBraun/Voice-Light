from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.shared.asr import TimestampedWord
from app.shared.audio.loading import AudioTrack
from app.shared.audio.metrics import frame_rms


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


def crosstalk_frame_power(
    target_audio: AudioTrack,
    other_audio: AudioTrack,
    config: CrosstalkFilterConfig,
) -> CrosstalkFramePower:
    if target_audio.metadata.sample_rate != other_audio.metadata.sample_rate:
        raise ValueError("Crosstalk filtering requires matching speaker-track sample rates.")
    frame_size = max(1, round(config.frame_duration_seconds * target_audio.metadata.sample_rate))
    target_power = frame_power(samples=target_audio.samples[:, 0], frame_size=frame_size)
    other_power = frame_power(samples=other_audio.samples[:, 0], frame_size=frame_size)
    shared_frame_count = min(len(target_power), len(other_power))
    return CrosstalkFramePower(
        target=target_power[:shared_frame_count],
        other=other_power[:shared_frame_count],
        frame_duration_seconds=frame_size / target_audio.metadata.sample_rate,
    )


def frame_power(samples: NDArray[np.float32], frame_size: int) -> NDArray[np.float32]:
    return np.square(frame_rms(samples=samples, frame_size=frame_size)).astype(np.float32)


def filter_crosstalk_words(
    words: tuple[TimestampedWord, ...],
    frame_power_pair: CrosstalkFramePower,
    config: CrosstalkFilterConfig,
) -> tuple[TimestampedWord, ...]:
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
    words: tuple[TimestampedWord, ...],
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
    word: TimestampedWord,
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
    assert word.start_seconds is not None
    assert word.end_seconds is not None
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
    word: TimestampedWord,
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


def parakeet_canary_union_words(
    parakeet_words: tuple[TimestampedWord, ...],
    canary_words: tuple[TimestampedWord, ...],
) -> tuple[TimestampedWord, ...]:
    canary_only_words = tuple(
        canary_word
        for canary_word in canary_words
        if is_timestamped(canary_word)
        and not any(
            words_overlap(first=canary_word, second=parakeet_word)
            for parakeet_word in parakeet_words
        )
    )
    return tuple(
        sorted(
            (*parakeet_words, *canary_only_words),
            key=lambda word: word.start_seconds if word.start_seconds is not None else float("inf"),
        )
    )


def is_timestamped(word: TimestampedWord) -> bool:
    return word.start_seconds is not None and word.end_seconds is not None


def words_overlap(first: TimestampedWord, second: TimestampedWord) -> bool:
    if not is_timestamped(first) or not is_timestamped(second):
        return False
    assert first.start_seconds is not None
    assert first.end_seconds is not None
    assert second.start_seconds is not None
    assert second.end_seconds is not None
    return first.start_seconds <= second.end_seconds and first.end_seconds >= second.start_seconds
