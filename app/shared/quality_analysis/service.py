from __future__ import annotations

from app.shared.audio import AudioTrack
from app.shared.quality import (
    ConversationAnnotation,
    ProcessingStatus,
    QualityResult,
    RunConfig,
    SpeakerSide,
)
from app.shared.quality_analysis.audio_quality import (
    energy_envelope_correlation,
    inactive_track_leakage_db,
    track_audio_quality,
    track_correlation,
)
from app.shared.quality_analysis.events import extract_event_candidates
from app.shared.quality_analysis.preprocessing import prepare_audio_track
from app.shared.quality_analysis.scoring import (
    calibrated_quality_score,
    calibration_flags,
    combine_audio_quality_metrics,
    compute_interaction_density_metrics,
    compute_timing_reliability_metrics,
    total_quality_score,
)
from app.shared.quality_analysis.vad import VadConfig, detect_speech_segments_pair


def score_two_track_sample(
    sample_id: str,
    speaker1_audio: AudioTrack,
    speaker2_audio: AudioTrack,
    speaker1_uri: str,
    speaker2_uri: str,
    conversation_annotation: ConversationAnnotation | None = None,
    config: RunConfig | None = None,
) -> QualityResult:
    effective_config = config if config is not None else RunConfig()
    try:
        prepared_speaker1_audio = prepare_audio_track(speaker1_audio)
        prepared_speaker2_audio = prepare_audio_track(speaker2_audio)
        sample_rate = prepared_speaker1_audio.metadata.sample_rate
        speaker1_vad, speaker2_vad = detect_speech_segments_pair(
            prepared_speaker1_audio.samples,
            prepared_speaker2_audio.samples,
            sample_rate,
            VadConfig(),
        )
        duration_seconds = min(
            prepared_speaker1_audio.metadata.duration_seconds,
            prepared_speaker2_audio.metadata.duration_seconds,
        )
        event_candidates = extract_event_candidates(speaker1_vad, speaker2_vad, duration_seconds)
        stored_event_candidates = event_candidates[: effective_config.max_events_per_sample]
        speaker1_quality = track_audio_quality(
            prepared_speaker1_audio.samples,
            prepared_speaker1_audio.metadata.sample_rate,
            prepared_speaker1_audio.metadata.channels,
            speaker1_vad,
            SpeakerSide.SPEAKER1,
        )
        speaker2_quality = track_audio_quality(
            prepared_speaker2_audio.samples,
            prepared_speaker2_audio.metadata.sample_rate,
            prepared_speaker2_audio.metadata.channels,
            speaker2_vad,
            SpeakerSide.SPEAKER2,
        )
        max_duration_seconds = max(
            prepared_speaker1_audio.metadata.duration_seconds,
            prepared_speaker2_audio.metadata.duration_seconds,
        )
        duration_gap_seconds = abs(
            prepared_speaker1_audio.metadata.duration_seconds
            - prepared_speaker2_audio.metadata.duration_seconds
        )
        audio_metrics = combine_audio_quality_metrics(
            speaker1=speaker1_quality,
            speaker2=speaker2_quality,
            duration_gap_seconds=duration_gap_seconds,
            duration_gap_ratio=duration_gap_seconds / max_duration_seconds
            if max_duration_seconds > 0.0
            else 0.0,
            track_correlation=track_correlation(
                prepared_speaker1_audio.samples,
                prepared_speaker2_audio.samples,
            ),
            energy_envelope_correlation=energy_envelope_correlation(
                prepared_speaker1_audio.samples,
                prepared_speaker2_audio.samples,
                prepared_speaker1_audio.metadata.sample_rate,
            ),
            speaker1_leakage_db=inactive_track_leakage_db(
                prepared_speaker1_audio.samples,
                speaker1_vad,
                speaker2_vad,
                prepared_speaker1_audio.metadata.sample_rate,
            ),
            speaker2_leakage_db=inactive_track_leakage_db(
                prepared_speaker2_audio.samples,
                speaker2_vad,
                speaker1_vad,
                prepared_speaker2_audio.metadata.sample_rate,
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
            conversation_annotation=conversation_annotation,
            weights=effective_config.weights,
        )
        calibrated_score = calibrated_quality_score(
            raw_quality_score,
            density_metrics,
            timing_metrics,
            audio_metrics,
            conversation_annotation,
        )
        result_flags = calibration_flags(
            density_metrics,
            timing_metrics,
            audio_metrics,
            conversation_annotation,
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
            conversation_annotation=conversation_annotation,
            event_candidates=stored_event_candidates,
            raw_quality_score=raw_quality_score,
            calibrated_quality_score=calibrated_score,
            calibration_flags=result_flags,
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
            conversation_annotation=None,
            event_candidates=(),
            raw_quality_score=None,
            calibrated_quality_score=None,
            calibration_flags=(),
            total_quality_score=None,
            error=f"{type(error).__name__}: {error}",
        )
