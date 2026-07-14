from __future__ import annotations

from app.shared.audio import AudioTrack
from app.shared.quality import (
    ConversationAnnotation,
    ConversationCountEstimate,
    ConversationEventCounts,
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
    combine_audio_quality_metrics,
    compute_interaction_density_metrics,
    compute_timing_reliability_metrics,
    quality_flags,
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
        result_flags = quality_flags(
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
            conversation_count_estimate=conversation_count_estimate(
                annotation=conversation_annotation,
                represented_duration_seconds=duration_seconds,
            ),
            event_candidates=stored_event_candidates,
            raw_quality_score=raw_quality_score,
            quality_flags=result_flags,
            total_quality_score=raw_quality_score,
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
            conversation_count_estimate=None,
            event_candidates=(),
            raw_quality_score=None,
            quality_flags=(),
            total_quality_score=None,
            error=f"{type(error).__name__}: {error}",
        )


def conversation_count_estimate(
    annotation: ConversationAnnotation | None,
    represented_duration_seconds: float,
) -> ConversationCountEstimate | None:
    if annotation is None:
        return None
    if annotation.analyzed_duration_seconds <= 0.0:
        raise ValueError("Conversation annotation duration must be positive.")
    if represented_duration_seconds <= 0.0:
        raise ValueError("Represented conversation duration must be positive.")
    scale_factor = max(
        1.0,
        represented_duration_seconds / annotation.analyzed_duration_seconds,
    )
    observed = conversation_event_counts(annotation)
    return ConversationCountEstimate(
        annotation_duration_seconds=annotation.analyzed_duration_seconds,
        represented_duration_seconds=represented_duration_seconds,
        scale_factor=scale_factor,
        observed=observed,
        estimated=ConversationEventCounts(
            speech_segment_count=round(observed.speech_segment_count * scale_factor),
            interaction_count=round(observed.interaction_count * scale_factor),
            turn_count=round(observed.turn_count * scale_factor),
            turn_taking_count=round(observed.turn_taking_count * scale_factor),
            pause_count=round(observed.pause_count * scale_factor),
            backchannel_count=round(observed.backchannel_count * scale_factor),
            interruption_count=round(observed.interruption_count * scale_factor),
            usable_event_count=round(observed.usable_event_count * scale_factor),
        ),
    )


def conversation_event_counts(annotation: ConversationAnnotation) -> ConversationEventCounts:
    return ConversationEventCounts(
        speech_segment_count=annotation.speech_segment_count,
        interaction_count=annotation.interaction_count,
        turn_count=annotation.turn_count,
        turn_taking_count=annotation.turn_taking_count,
        pause_count=annotation.pause_count,
        backchannel_count=annotation.backchannel_count,
        interruption_count=annotation.interruption_count,
        usable_event_count=annotation.usable_event_count,
    )
