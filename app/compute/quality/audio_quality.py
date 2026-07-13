from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.compute.quality.utils import binary_entropy, clamp, penalty_above, safe_ratio
from app.shared.audio.metrics import frame_rms
from app.shared.quality import SpeakerSide, TrackAudioQuality, TrackVadResult


def track_audio_quality(
    samples: NDArray[np.float32],
    sample_rate: int,
    channels: int,
    vad_result: TrackVadResult,
    side: SpeakerSide,
) -> TrackAudioQuality:
    duration_seconds = len(samples) / sample_rate
    absolute_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) > 0 else 0.0
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-8))
    peak_amplitude = float(np.max(absolute_samples)) if len(samples) > 0 else 0.0
    clipping_ratio = float(np.mean(absolute_samples >= 0.999)) if len(samples) > 0 else 0.0
    near_zero_ratio = float(np.mean(absolute_samples <= 1e-4)) if len(samples) > 0 else 1.0
    silence_ratio = clamp(1.0 - vad_result.speech_ratio, 0.0, 1.0)
    speech_silence_entropy = binary_entropy(vad_result.speech_ratio)

    flags: list[str] = []
    if vad_result.speech_ratio < 0.05:
        flags.append("too_little_speech")
    if silence_ratio > 0.9:
        flags.append("too_much_silence")
    if clipping_ratio > 0.01:
        flags.append("clipping")
    if near_zero_ratio > 0.95:
        flags.append("near_zero_signal")
    if peak_amplitude < 0.005:
        flags.append("very_low_peak")
    if rms_dbfs < -60.0:
        flags.append("very_low_rms")
    elif rms_dbfs < -55.0:
        flags.append("low_rms")

    speech_score = quality_window_score(vad_result.speech_ratio, 0.05, 0.2, 0.85, 0.95)
    rms_score = quality_window_score(rms_dbfs, -70.0, -50.0, -18.0, -6.0)
    clipping_score = penalty_above(clipping_ratio, 0.001, 0.05)
    near_zero_score = penalty_above(near_zero_ratio, 0.6, 0.98)
    peak_score = quality_window_score(peak_amplitude, 0.005, 0.03, 0.98, 1.0)
    quality_score = clamp(
        0.25 * speech_score
        + 0.2 * rms_score
        + 0.2 * clipping_score
        + 0.15 * near_zero_score
        + 0.15 * peak_score
        + 0.05 * speech_silence_entropy,
        0.0,
        1.0,
    )
    if rms_dbfs < -60.0 or peak_amplitude < 0.01:
        quality_score = min(quality_score, 0.25)
    elif rms_dbfs < -55.0:
        quality_score = min(quality_score, 0.5)
    return TrackAudioQuality(
        side=side,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        rms_dbfs=rms_dbfs,
        peak_amplitude=peak_amplitude,
        clipping_ratio=clipping_ratio,
        near_zero_ratio=near_zero_ratio,
        silence_ratio=silence_ratio,
        speech_ratio=vad_result.speech_ratio,
        speech_silence_entropy=speech_silence_entropy,
        low_information=len(flags) > 0 and vad_result.speech_ratio < 0.05,
        quality_score=quality_score,
        flags=tuple(flags),
    )


def quality_window_score(
    value: float, low: float, ideal_low: float, ideal_high: float, high: float
) -> float:
    if low >= ideal_low or ideal_low > ideal_high or ideal_high >= high:
        raise ValueError("quality window bounds must satisfy low < ideal_low <= ideal_high < high")
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return clamp(safe_ratio(value - low, ideal_low - low), 0.0, 1.0)
    return clamp(safe_ratio(high - value, high - ideal_high), 0.0, 1.0)


def track_correlation(samples1: NDArray[np.float32], samples2: NDArray[np.float32]) -> float | None:
    sample_count = min(len(samples1), len(samples2))
    if sample_count < 2:
        return None
    trimmed1 = samples1[:sample_count]
    trimmed2 = samples2[:sample_count]
    if float(np.std(trimmed1)) <= 1e-8 or float(np.std(trimmed2)) <= 1e-8:
        return None
    correlation_matrix = np.corrcoef(trimmed1, trimmed2)
    return float(correlation_matrix[0, 1])


def energy_envelope_correlation(
    samples1: NDArray[np.float32],
    samples2: NDArray[np.float32],
    sample_rate: int,
    frame_duration_seconds: float = 0.1,
) -> float | None:
    sample_count = min(len(samples1), len(samples2))
    if sample_count < 2:
        return None
    frame_size = max(1, round(frame_duration_seconds * sample_rate))
    envelope1 = np.log10(np.maximum(frame_rms(samples1[:sample_count], frame_size), 1e-8))
    envelope2 = np.log10(np.maximum(frame_rms(samples2[:sample_count], frame_size), 1e-8))
    if float(np.std(envelope1)) <= 1e-8 or float(np.std(envelope2)) <= 1e-8:
        return None
    correlation_matrix = np.corrcoef(envelope1, envelope2)
    return float(correlation_matrix[0, 1])


def inactive_track_leakage_db(
    samples: NDArray[np.float32],
    own_vad: TrackVadResult,
    other_vad: TrackVadResult,
    sample_rate: int,
    frame_duration_seconds: float = 0.1,
) -> float | None:
    frame_size = max(1, round(frame_duration_seconds * sample_rate))
    frame_energy_db = 20.0 * np.log10(np.maximum(frame_rms(samples, frame_size), 1e-8))
    frame_count = len(frame_energy_db)
    own_mask = speech_mask_for_segments(own_vad, frame_count, frame_duration_seconds)
    other_mask = speech_mask_for_segments(other_vad, frame_count, frame_duration_seconds)
    own_only_energy = frame_energy_db[own_mask & ~other_mask]
    leaked_energy = frame_energy_db[other_mask & ~own_mask]
    if len(own_only_energy) == 0 or len(leaked_energy) == 0:
        return None
    return float(np.median(leaked_energy) - np.median(own_only_energy))


def speech_mask_for_segments(
    vad_result: TrackVadResult,
    frame_count: int,
    frame_duration_seconds: float,
) -> NDArray[np.bool_]:
    mask = np.zeros(frame_count, dtype=bool)
    for segment in vad_result.speech_segments:
        start_index = max(0, int(segment.start_seconds / frame_duration_seconds))
        end_index = min(frame_count, int(np.ceil(segment.end_seconds / frame_duration_seconds)))
        mask[start_index:end_index] = True
    return mask
