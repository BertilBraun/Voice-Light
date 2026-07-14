from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.shared.audio.metrics import frame_rms
from app.shared.quality import SpeakerSide, SpeechSegment, TrackVadResult
from app.shared.quality_analysis.utils import median, safe_ratio


@dataclass(frozen=True)
class VadConfig:
    frame_duration_seconds: float = 0.03
    energy_floor_percentile: float = 20.0
    energy_threshold_db_above_floor: float = 12.0
    minimum_speech_dbfs: float = -50.0
    minimum_speech_duration_seconds: float = 0.12
    merge_gap_seconds: float = 0.25
    relative_dominance_db: float = 6.0
    maximum_channel_baseline_correction_db: float = 3.0


def detect_speech_segments_pair(
    speaker1_samples: NDArray[np.float32],
    speaker2_samples: NDArray[np.float32],
    sample_rate: int,
    config: VadConfig,
) -> tuple[TrackVadResult, TrackVadResult]:
    speaker1_mask = speech_mask_for_samples(speaker1_samples, sample_rate, config)
    speaker2_mask = speech_mask_for_samples(speaker2_samples, sample_rate, config)
    speaker1_energy_db = frame_energy_db(speaker1_samples, sample_rate, config)
    speaker2_energy_db = frame_energy_db(speaker2_samples, sample_rate, config)
    speaker1_baseline_db = active_energy_baseline_db(speaker1_mask, speaker1_energy_db)
    speaker2_baseline_db = active_energy_baseline_db(speaker2_mask, speaker2_energy_db)
    speaker1_mask, speaker2_mask = suppress_non_dominant_shared_activity(
        speaker1_mask=speaker1_mask,
        speaker2_mask=speaker2_mask,
        speaker1_energy_db=speaker1_energy_db,
        speaker2_energy_db=speaker2_energy_db,
        speaker1_baseline_db=speaker1_baseline_db,
        speaker2_baseline_db=speaker2_baseline_db,
        relative_dominance_db=config.relative_dominance_db,
        maximum_channel_baseline_correction_db=config.maximum_channel_baseline_correction_db,
    )
    speaker1_raw_segments = speech_segments_for_mask(speaker1_mask, config)
    speaker2_raw_segments = speech_segments_for_mask(speaker2_mask, config)
    speaker1_segments = merge_close_segments_blocked_by_other_activity(
        speaker1_raw_segments,
        speaker2_raw_segments,
        config.merge_gap_seconds,
    )
    speaker2_segments = merge_close_segments_blocked_by_other_activity(
        speaker2_raw_segments,
        speaker1_raw_segments,
        config.merge_gap_seconds,
    )
    return (
        vad_result_from_segments(
            speaker1_samples, sample_rate, SpeakerSide.SPEAKER1, tuple(speaker1_segments)
        ),
        vad_result_from_segments(
            speaker2_samples, sample_rate, SpeakerSide.SPEAKER2, tuple(speaker2_segments)
        ),
    )


def speech_mask_for_samples(
    samples: NDArray[np.float32],
    sample_rate: int,
    config: VadConfig,
) -> NDArray[np.bool_]:
    rms_db = frame_energy_db(samples, sample_rate, config)
    if len(rms_db) == 0:
        return np.array([], dtype=bool)
    floor_db = float(np.percentile(rms_db, config.energy_floor_percentile))
    speech_threshold_db = max(
        floor_db + config.energy_threshold_db_above_floor, config.minimum_speech_dbfs
    )
    return rms_db >= speech_threshold_db


def frame_energy_db(
    samples: NDArray[np.float32],
    sample_rate: int,
    config: VadConfig,
) -> NDArray[np.float32]:
    frame_size = max(1, round(config.frame_duration_seconds * sample_rate))
    rms_values = frame_rms(samples, frame_size)
    if len(rms_values) == 0:
        return np.array([], dtype=np.float32)
    return (20.0 * np.log10(np.maximum(rms_values, 1e-8))).astype(np.float32)


def suppress_non_dominant_shared_activity(
    speaker1_mask: NDArray[np.bool_],
    speaker2_mask: NDArray[np.bool_],
    speaker1_energy_db: NDArray[np.float32],
    speaker2_energy_db: NDArray[np.float32],
    speaker1_baseline_db: float,
    speaker2_baseline_db: float,
    relative_dominance_db: float,
    maximum_channel_baseline_correction_db: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    filtered_speaker1_mask = speaker1_mask.copy()
    filtered_speaker2_mask = speaker2_mask.copy()
    shared_frame_count = min(
        len(filtered_speaker1_mask),
        len(filtered_speaker2_mask),
        len(speaker1_energy_db),
        len(speaker2_energy_db),
    )
    if shared_frame_count == 0:
        return filtered_speaker1_mask, filtered_speaker2_mask
    shared_activity = (
        filtered_speaker1_mask[:shared_frame_count] & filtered_speaker2_mask[:shared_frame_count]
    )
    channel_baseline_delta_db = float(
        np.clip(
            speaker1_baseline_db - speaker2_baseline_db,
            -maximum_channel_baseline_correction_db,
            maximum_channel_baseline_correction_db,
        )
    )
    dominance_delta_db = (
        speaker1_energy_db[:shared_frame_count]
        - speaker2_energy_db[:shared_frame_count]
        - channel_baseline_delta_db
    )
    filtered_speaker1_mask[:shared_frame_count][
        shared_activity & (dominance_delta_db <= -relative_dominance_db)
    ] = False
    filtered_speaker2_mask[:shared_frame_count][
        shared_activity & (dominance_delta_db >= relative_dominance_db)
    ] = False
    return filtered_speaker1_mask, filtered_speaker2_mask


def active_energy_baseline_db(
    speech_mask: NDArray[np.bool_], energy_db: NDArray[np.float32]
) -> float:
    shared_frame_count = min(len(speech_mask), len(energy_db))
    if shared_frame_count == 0:
        return 0.0
    active_energy_db = energy_db[:shared_frame_count][speech_mask[:shared_frame_count]]
    if len(active_energy_db) > 0:
        return float(np.median(active_energy_db))
    return float(np.median(energy_db[:shared_frame_count]))


def speech_segments_for_mask(
    speech_mask: NDArray[np.bool_],
    config: VadConfig,
) -> tuple[SpeechSegment, ...]:
    raw_segments = segments_from_mask(speech_mask, config.frame_duration_seconds)
    return tuple(
        segment
        for segment in raw_segments
        if segment.duration_seconds >= config.minimum_speech_duration_seconds
    )


def vad_result_from_segments(
    samples: NDArray[np.float32],
    sample_rate: int,
    side: SpeakerSide,
    speech_segments: tuple[SpeechSegment, ...],
) -> TrackVadResult:
    duration_seconds = len(samples) / sample_rate
    speech_time_seconds = sum(segment.duration_seconds for segment in speech_segments)
    durations = [segment.duration_seconds for segment in speech_segments]
    tiny_fragment_count = sum(1 for duration in durations if duration < 0.2)
    long_segment_count = sum(1 for duration in durations if duration > 30.0)
    return TrackVadResult(
        side=side,
        speech_segments=speech_segments,
        speech_time_seconds=speech_time_seconds,
        speech_ratio=safe_ratio(speech_time_seconds, duration_seconds),
        median_segment_duration_seconds=median(durations),
        tiny_fragment_ratio=safe_ratio(tiny_fragment_count, len(durations)),
        long_segment_ratio=safe_ratio(long_segment_count, len(durations)),
    )


def segments_from_mask(
    mask: NDArray[np.bool_], frame_duration_seconds: float
) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    active_start_index: int | None = None
    for frame_index, is_active in enumerate(mask):
        if is_active and active_start_index is None:
            active_start_index = frame_index
        if not is_active and active_start_index is not None:
            segments.append(
                SpeechSegment(
                    start_seconds=active_start_index * frame_duration_seconds,
                    end_seconds=frame_index * frame_duration_seconds,
                )
            )
            active_start_index = None
    if active_start_index is not None:
        segments.append(
            SpeechSegment(
                start_seconds=active_start_index * frame_duration_seconds,
                end_seconds=len(mask) * frame_duration_seconds,
            )
        )
    return segments


def merge_close_segments_blocked_by_other_activity(
    own_segments: Sequence[SpeechSegment],
    other_segments: Sequence[SpeechSegment],
    merge_gap_seconds: float,
) -> list[SpeechSegment]:
    if not own_segments:
        return []
    merged_segments = [own_segments[0]]
    for segment in own_segments[1:]:
        previous_segment = merged_segments[-1]
        gap_seconds = segment.start_seconds - previous_segment.end_seconds
        if gap_seconds <= merge_gap_seconds and not has_segment_overlap(
            other_segments,
            previous_segment.end_seconds,
            segment.start_seconds,
        ):
            merged_segments[-1] = SpeechSegment(
                start_seconds=previous_segment.start_seconds,
                end_seconds=segment.end_seconds,
            )
            continue
        merged_segments.append(segment)
    return merged_segments


def has_segment_overlap(
    segments: Sequence[SpeechSegment], start_seconds: float, end_seconds: float
) -> bool:
    for segment in segments:
        if min(segment.end_seconds, end_seconds) > max(segment.start_seconds, start_seconds):
            return True
    return False
