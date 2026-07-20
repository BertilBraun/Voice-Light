from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path

import numpy as np

from app.shared.audio.wav import read_mono_wave_audio
from app.shared.base_model import FrozenBaseModel
from app.shared.quality import AnnotationSpan, TrackAudioQuality

TARGET_ACTIVE_RMS_DBFS = -20.0
MINIMUM_DEFAULT_GAIN = 0.1
MAXIMUM_DEFAULT_GAIN = 12.0


class GainMeasurementBasis(StrEnum):
    WHOLE_TRACK_ESTIMATE = "whole_track_estimate"
    ANNOTATED_SPEECH = "annotated_speech"


class TrackGainNormalization(FrozenBaseModel):
    measurement_basis: GainMeasurementBasis
    measured_rms_dbfs: float | None
    estimated_active_rms_dbfs: float | None
    measured_speech_duration_seconds: float | None
    target_active_rms_dbfs: float
    default_gain: float


def track_gain_normalization(
    track_quality: TrackAudioQuality | None,
) -> TrackGainNormalization:
    if track_quality is None:
        return TrackGainNormalization(
            measurement_basis=GainMeasurementBasis.WHOLE_TRACK_ESTIMATE,
            measured_rms_dbfs=None,
            estimated_active_rms_dbfs=None,
            measured_speech_duration_seconds=None,
            target_active_rms_dbfs=TARGET_ACTIVE_RMS_DBFS,
            default_gain=1.0,
        )
    active_rms_dbfs = estimated_active_rms_dbfs(
        rms_dbfs=track_quality.rms_dbfs,
        speech_ratio=track_quality.speech_ratio,
    )
    return TrackGainNormalization(
        measurement_basis=GainMeasurementBasis.WHOLE_TRACK_ESTIMATE,
        measured_rms_dbfs=track_quality.rms_dbfs,
        estimated_active_rms_dbfs=active_rms_dbfs,
        measured_speech_duration_seconds=None,
        target_active_rms_dbfs=TARGET_ACTIVE_RMS_DBFS,
        default_gain=default_gain_for_active_rms(active_rms_dbfs=active_rms_dbfs),
    )


def estimated_active_rms_dbfs(rms_dbfs: float, speech_ratio: float) -> float:
    bounded_speech_ratio = min(1.0, max(0.01, speech_ratio))
    return rms_dbfs - 10.0 * math.log10(bounded_speech_ratio)


def default_gain_for_active_rms(active_rms_dbfs: float) -> float:
    gain = 10.0 ** ((TARGET_ACTIVE_RMS_DBFS - active_rms_dbfs) / 20.0)
    return min(MAXIMUM_DEFAULT_GAIN, max(MINIMUM_DEFAULT_GAIN, gain))


def speech_only_gain_normalization(
    wave_path: Path,
    speech_segments: tuple[AnnotationSpan, ...],
) -> TrackGainNormalization:
    audio = read_mono_wave_audio(wave_path=wave_path)
    maximum_amplitude = float((1 << (audio.sample_width * 8 - 1)) - 1)
    squared_amplitude_sum = 0.0
    speech_sample_count = 0
    for span in merged_speech_spans(spans=speech_segments):
        start_index = max(0, round(span.start_seconds * audio.sample_rate))
        end_index = min(audio.frame_count, round(span.end_seconds * audio.sample_rate))
        if end_index <= start_index:
            continue
        normalized_samples = audio.samples[start_index:end_index] / maximum_amplitude
        squared_amplitude_sum += float(np.sum(np.square(normalized_samples)))
        speech_sample_count += len(normalized_samples)
    if speech_sample_count == 0:
        return TrackGainNormalization(
            measurement_basis=GainMeasurementBasis.ANNOTATED_SPEECH,
            measured_rms_dbfs=None,
            estimated_active_rms_dbfs=None,
            measured_speech_duration_seconds=0.0,
            target_active_rms_dbfs=TARGET_ACTIVE_RMS_DBFS,
            default_gain=1.0,
        )
    rms = math.sqrt(squared_amplitude_sum / speech_sample_count)
    speech_rms_dbfs = 20.0 * math.log10(max(rms, 1e-8))
    return TrackGainNormalization(
        measurement_basis=GainMeasurementBasis.ANNOTATED_SPEECH,
        measured_rms_dbfs=speech_rms_dbfs,
        estimated_active_rms_dbfs=speech_rms_dbfs,
        measured_speech_duration_seconds=speech_sample_count / audio.sample_rate,
        target_active_rms_dbfs=TARGET_ACTIVE_RMS_DBFS,
        default_gain=default_gain_for_active_rms(active_rms_dbfs=speech_rms_dbfs),
    )


def merged_speech_spans(
    spans: tuple[AnnotationSpan, ...],
) -> tuple[AnnotationSpan, ...]:
    ordered_spans = sorted(
        (span for span in spans if span.end_seconds > span.start_seconds),
        key=lambda span: span.start_seconds,
    )
    if not ordered_spans:
        return ()
    merged: list[AnnotationSpan] = [ordered_spans[0]]
    for span in ordered_spans[1:]:
        previous = merged[-1]
        if span.start_seconds <= previous.end_seconds:
            merged[-1] = AnnotationSpan(
                start_seconds=previous.start_seconds,
                end_seconds=max(previous.end_seconds, span.end_seconds),
                text=None,
            )
        else:
            merged.append(span)
    return tuple(merged)
