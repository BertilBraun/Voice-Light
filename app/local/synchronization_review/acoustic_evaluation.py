from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.local.data.sessions import SpeakerName, session_audio_path
from app.local.synchronization_review.evaluation import (
    OffsetErrorMetrics,
    OffsetLabel,
    ReviewProvenance,
    offset_error_metrics,
)
from app.shared.audio.wav import read_mono_wave_audio
from app.shared.base_model import FrozenBaseModel

ACOUSTIC_FRAME_DURATION_SECONDS = 0.05
ACOUSTIC_MAXIMUM_LAG_SECONDS = 12.0
PEAK_EXCLUSION_SECONDS = 0.25


class AcousticOffsetPrediction(FrozenBaseModel):
    external_id: str
    provenance: ReviewProvenance
    reviewed_shift_seconds: float
    predicted_shift_seconds: float
    absolute_error_seconds: float
    peak_correlation: float
    peak_margin: float


class AcousticOffsetEvaluationReport(FrozenBaseModel):
    method: str
    frame_duration_seconds: float
    analyzed_duration_limit_seconds: float
    combined_reviewed_metrics: OffsetErrorMetrics
    database_review_metrics: OffsetErrorMetrics
    predictions: tuple[AcousticOffsetPrediction, ...]


def evaluate_acoustic_offsets(
    labels: tuple[OffsetLabel, ...],
) -> AcousticOffsetEvaluationReport:
    predictions = tuple(_prediction(label=label) for label in labels)
    database_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.provenance is ReviewProvenance.DATABASE
    )
    return AcousticOffsetEvaluationReport(
        method="normalized_log_energy_onset_correlation",
        frame_duration_seconds=ACOUSTIC_FRAME_DURATION_SECONDS,
        analyzed_duration_limit_seconds=180.0,
        combined_reviewed_metrics=offset_error_metrics(
            errors=tuple(prediction.absolute_error_seconds for prediction in predictions)
        ),
        database_review_metrics=offset_error_metrics(
            errors=tuple(prediction.absolute_error_seconds for prediction in database_predictions)
        ),
        predictions=predictions,
    )


def best_acoustic_offset(
    speaker1_envelope: NDArray[np.float64],
    speaker2_envelope: NDArray[np.float64],
    frame_duration_seconds: float = ACOUSTIC_FRAME_DURATION_SECONDS,
    maximum_lag_seconds: float = ACOUSTIC_MAXIMUM_LAG_SECONDS,
) -> tuple[float, float, float]:
    if frame_duration_seconds <= 0.0:
        raise ValueError("frame_duration_seconds must be positive")
    if maximum_lag_seconds <= 0.0:
        raise ValueError("maximum_lag_seconds must be positive")
    frame_count = min(len(speaker1_envelope), len(speaker2_envelope))
    maximum_lag_frames = round(maximum_lag_seconds / frame_duration_seconds)
    if frame_count <= 2 * maximum_lag_frames:
        raise ValueError("Acoustic envelopes are too short for the requested lag range")
    speaker1 = _standardized(speaker1_envelope[:frame_count])
    speaker2 = _standardized(speaker2_envelope[:frame_count])
    lag_correlations = tuple(
        (
            lag_frames,
            _lagged_correlation(
                speaker1=speaker1,
                speaker2=speaker2,
                lag_frames=lag_frames,
            ),
        )
        for lag_frames in range(-maximum_lag_frames, maximum_lag_frames + 1)
    )
    best_lag_frames, peak_correlation = max(lag_correlations, key=lambda item: item[1])
    exclusion_frames = max(1, round(PEAK_EXCLUSION_SECONDS / frame_duration_seconds))
    competing_peak = max(
        correlation
        for lag_frames, correlation in lag_correlations
        if abs(lag_frames - best_lag_frames) > exclusion_frames
    )
    return (
        best_lag_frames * frame_duration_seconds,
        peak_correlation,
        peak_correlation - competing_peak,
    )


def log_energy_onset_envelope(wave_path: Path) -> NDArray[np.float64]:
    audio = read_mono_wave_audio(wave_path=wave_path)
    frame_sample_count = round(audio.sample_rate * ACOUSTIC_FRAME_DURATION_SECONDS)
    framed_sample_count = len(audio.samples) - len(audio.samples) % frame_sample_count
    if framed_sample_count == 0:
        raise ValueError(f"Audio is too short for acoustic offset analysis: {wave_path}")
    frames = audio.samples[:framed_sample_count].reshape(-1, frame_sample_count)
    mean_square = np.mean(np.square(frames), axis=1)
    log_energy = 10.0 * np.log10(np.maximum(mean_square, 1e-12))
    lower, upper = np.percentile(log_energy, (5.0, 95.0))
    clipped_energy = np.clip(log_energy, lower, upper)
    return np.abs(np.diff(clipped_energy, prepend=clipped_energy[0])).astype(np.float64)


def _prediction(label: OffsetLabel) -> AcousticOffsetPrediction:
    speaker1_path = session_audio_path(
        identifier=label.external_id,
        speaker_name=SpeakerName.SPEAKER1,
    )
    speaker2_path = session_audio_path(
        identifier=label.external_id,
        speaker_name=SpeakerName.SPEAKER2,
    )
    predicted_shift, peak_correlation, peak_margin = best_acoustic_offset(
        speaker1_envelope=log_energy_onset_envelope(wave_path=speaker1_path),
        speaker2_envelope=log_energy_onset_envelope(wave_path=speaker2_path),
    )
    quantized_prediction = round(predicted_shift, 2)
    return AcousticOffsetPrediction(
        external_id=label.external_id,
        provenance=label.provenance,
        reviewed_shift_seconds=label.speaker2_shift_seconds,
        predicted_shift_seconds=quantized_prediction,
        absolute_error_seconds=abs(quantized_prediction - label.speaker2_shift_seconds),
        peak_correlation=peak_correlation,
        peak_margin=peak_margin,
    )


def _standardized(values: NDArray[np.float64]) -> NDArray[np.float64]:
    standard_deviation = float(np.std(values))
    if math.isclose(standard_deviation, 0.0):
        raise ValueError("Acoustic onset envelope has no variation")
    return (values - float(np.mean(values))) / standard_deviation


def _lagged_correlation(
    speaker1: NDArray[np.float64],
    speaker2: NDArray[np.float64],
    lag_frames: int,
) -> float:
    if lag_frames > 0:
        return float(np.mean(speaker1[lag_frames:] * speaker2[:-lag_frames]))
    if lag_frames < 0:
        return float(np.mean(speaker1[:lag_frames] * speaker2[-lag_frames:]))
    return float(np.mean(speaker1 * speaker2))
