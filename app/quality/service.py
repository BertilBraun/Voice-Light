from __future__ import annotations

from app.audio import AudioTrack
from app.quality.audio_quality import (
    energy_envelope_correlation,
    inactive_track_leakage_db,
    track_audio_quality,
    track_correlation,
)
from app.quality.events import extract_event_candidates
from app.quality.models import ProcessingStatus, QualityResult, RunConfig, SpeakerSide
from app.quality.scoring import (
    calibrated_quality_score,
    calibration_flags,
    combine_audio_quality_metrics,
    compute_interaction_density_metrics,
    compute_timing_reliability_metrics,
    total_quality_score,
)
from app.quality.vad import VadConfig, detect_speech_segments_pair


def score_two_track_sample(
    sample_id: str,
    speaker1_audio: AudioTrack,
    speaker2_audio: AudioTrack,
    speaker1_uri: str,
    speaker2_uri: str,
    config: RunConfig | None = None,
) -> QualityResult:
    effective_config = config if config is not None else RunConfig()
    try:
        sample_rate = min(speaker1_audio.metadata.sample_rate, speaker2_audio.metadata.sample_rate)
        if speaker1_audio.metadata.sample_rate != speaker2_audio.metadata.sample_rate:
            raise ValueError("speaker tracks must have matching sample rates")
        speaker1_vad, speaker2_vad = detect_speech_segments_pair(
            speaker1_audio.samples,
            speaker2_audio.samples,
            sample_rate,
            VadConfig(),
        )
        duration_seconds = min(
            speaker1_audio.metadata.duration_seconds,
            speaker2_audio.metadata.duration_seconds,
        )
        event_candidates = extract_event_candidates(speaker1_vad, speaker2_vad, duration_seconds)
        stored_event_candidates = event_candidates[: effective_config.max_events_per_sample]
        speaker1_quality = track_audio_quality(
            speaker1_audio.samples,
            speaker1_audio.metadata.sample_rate,
            speaker1_audio.metadata.channels,
            speaker1_vad,
            SpeakerSide.SPEAKER1,
        )
        speaker2_quality = track_audio_quality(
            speaker2_audio.samples,
            speaker2_audio.metadata.sample_rate,
            speaker2_audio.metadata.channels,
            speaker2_vad,
            SpeakerSide.SPEAKER2,
        )
        max_duration_seconds = max(
            speaker1_audio.metadata.duration_seconds,
            speaker2_audio.metadata.duration_seconds,
        )
        duration_gap_seconds = abs(
            speaker1_audio.metadata.duration_seconds - speaker2_audio.metadata.duration_seconds
        )
        audio_metrics = combine_audio_quality_metrics(
            speaker1=speaker1_quality,
            speaker2=speaker2_quality,
            duration_gap_seconds=duration_gap_seconds,
            duration_gap_ratio=duration_gap_seconds / max_duration_seconds
            if max_duration_seconds > 0.0
            else 0.0,
            track_correlation=track_correlation(speaker1_audio.samples, speaker2_audio.samples),
            energy_envelope_correlation=energy_envelope_correlation(
                speaker1_audio.samples,
                speaker2_audio.samples,
                speaker1_audio.metadata.sample_rate,
            ),
            speaker1_leakage_db=inactive_track_leakage_db(
                speaker1_audio.samples,
                speaker1_vad,
                speaker2_vad,
                speaker1_audio.metadata.sample_rate,
            ),
            speaker2_leakage_db=inactive_track_leakage_db(
                speaker2_audio.samples,
                speaker2_vad,
                speaker1_vad,
                speaker2_audio.metadata.sample_rate,
            ),
        )
        density_metrics = compute_interaction_density_metrics(
            event_candidates,
            speaker1_vad,
            speaker2_vad,
            duration_seconds,
        )
        timing_metrics = compute_timing_reliability_metrics(
            event_candidates,
            speaker1_vad,
            speaker2_vad,
            duration_seconds,
        )
        raw_quality_score = total_quality_score(
            interaction_density=density_metrics,
            timing_reliability=timing_metrics,
            audio_quality=audio_metrics,
            weights=effective_config.weights,
        )
        calibrated_score = calibrated_quality_score(
            raw_quality_score,
            density_metrics,
            timing_metrics,
            audio_metrics,
        )
        return QualityResult(
            metric_version=effective_config.metric_version,
            sample_id=sample_id,
            status=ProcessingStatus.COMPLETED,
            speaker1_uri=speaker1_uri,
            speaker2_uri=speaker2_uri,
            duration_seconds=duration_seconds,
            interaction_density=density_metrics,
            timing_reliability=timing_metrics,
            audio_quality=audio_metrics,
            event_candidates=stored_event_candidates,
            raw_quality_score=raw_quality_score,
            calibrated_quality_score=calibrated_score,
            calibration_flags=calibration_flags(density_metrics, timing_metrics, audio_metrics),
            total_quality_score=calibrated_score,
            error=None,
        )
    except Exception as error:
        return QualityResult(
            metric_version=effective_config.metric_version,
            sample_id=sample_id,
            status=ProcessingStatus.FAILED,
            speaker1_uri=speaker1_uri,
            speaker2_uri=speaker2_uri,
            duration_seconds=None,
            interaction_density=None,
            timing_reliability=None,
            audio_quality=None,
            event_candidates=(),
            raw_quality_score=None,
            calibrated_quality_score=None,
            calibration_flags=(),
            total_quality_score=None,
            error=f"{type(error).__name__}: {error}",
        )
