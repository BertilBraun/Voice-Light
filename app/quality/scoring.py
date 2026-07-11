from __future__ import annotations

from dataclasses import dataclass

from app.quality.models import (
    AudioQualityMetrics,
    EventCandidate,
    EventType,
    InteractionDensityMetrics,
    QualityWeights,
    SpeakerSide,
    TimingReliabilityMetrics,
    TrackAudioQuality,
    TrackVadResult,
)
from app.quality.utils import clamp, median, penalty_above, per_hour, safe_ratio, score_between


@dataclass(frozen=True)
class DensityScoringConfig:
    minimum_speech_ratio: float = 0.12
    target_speech_ratio: float = 0.45
    target_turn_completions_per_hour: float = 120.0
    target_candidate_windows_per_hour: float = 180.0
    maximum_overlap_ratio: float = 0.18


@dataclass(frozen=True)
class TimingScoringConfig:
    minimum_plausible_median_segment_seconds: float = 0.25
    target_plausible_median_segment_seconds: float = 1.20
    maximum_tiny_fragment_ratio: float = 0.15
    rejected_tiny_fragment_ratio: float = 0.50
    maximum_long_segment_ratio: float = 0.02
    rejected_long_segment_ratio: float = 0.15


@dataclass(frozen=True)
class EventTypeCounts:
    turn_completions: int
    pauses: int
    start_responses: int
    interruptions: int
    backchannels: int
    overlaps: int


@dataclass(frozen=True)
class SpeakerEventCounts:
    speaker1: int
    speaker2: int


@dataclass(frozen=True)
class TimeInterval:
    start_seconds: float
    end_seconds: float


DEFAULT_DENSITY_SCORING_CONFIG = DensityScoringConfig()
DEFAULT_TIMING_SCORING_CONFIG = TimingScoringConfig()


def compute_interaction_density_metrics(
    event_candidates: tuple[EventCandidate, ...],
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
    total_duration_seconds: float,
    config: DensityScoringConfig = DEFAULT_DENSITY_SCORING_CONFIG,
) -> InteractionDensityMetrics:
    if total_duration_seconds <= 0.0:
        raise ValueError("total_duration_seconds must be positive")
    event_counts = _count_event_types(event_candidates)
    speech_ratio = _union_speech_ratio(speaker1_vad, speaker2_vad, total_duration_seconds)
    overlap_ratio = _overlap_ratio(speaker1_vad, speaker2_vad, total_duration_seconds)
    silence_ratio = clamp(1.0 - speech_ratio, 0.0, 1.0)
    usable_candidate_count = (
        event_counts.turn_completions
        + event_counts.pauses
        + event_counts.start_responses
        + event_counts.interruptions
        + event_counts.backchannels
    )
    turn_completions_per_hour = per_hour(event_counts.turn_completions, total_duration_seconds)
    usable_candidate_windows_per_hour = per_hour(usable_candidate_count, total_duration_seconds)
    speech_score = score_between(
        speech_ratio, config.minimum_speech_ratio, config.target_speech_ratio
    )
    turn_score = score_between(
        turn_completions_per_hour, 0.0, config.target_turn_completions_per_hour
    )
    candidate_score = score_between(
        usable_candidate_windows_per_hour, 0.0, config.target_candidate_windows_per_hour
    )
    overlap_score = penalty_above(
        overlap_ratio, config.maximum_overlap_ratio, config.maximum_overlap_ratio * 2.0
    )
    return InteractionDensityMetrics(
        speech_ratio=speech_ratio,
        silence_ratio=silence_ratio,
        overlap_ratio=overlap_ratio,
        turn_completions_per_hour=turn_completions_per_hour,
        pause_events_per_hour=per_hour(event_counts.pauses, total_duration_seconds),
        start_responses_per_hour=per_hour(event_counts.start_responses, total_duration_seconds),
        interruptions_per_hour=per_hour(event_counts.interruptions, total_duration_seconds),
        backchannels_per_hour=per_hour(event_counts.backchannels, total_duration_seconds),
        overlaps_per_hour=per_hour(event_counts.overlaps, total_duration_seconds),
        usable_candidate_windows_per_hour=usable_candidate_windows_per_hour,
        quality_score=clamp(
            (speech_score + turn_score + candidate_score + overlap_score) / 4.0, 0.0, 1.0
        ),
    )


def compute_timing_reliability_metrics(
    event_candidates: tuple[EventCandidate, ...],
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
    total_duration_seconds: float,
    config: TimingScoringConfig = DEFAULT_TIMING_SCORING_CONFIG,
) -> TimingReliabilityMetrics:
    if total_duration_seconds <= 0.0:
        raise ValueError("total_duration_seconds must be positive")
    segment_durations = [
        segment.duration_seconds
        for segment in speaker1_vad.speech_segments + speaker2_vad.speech_segments
    ]
    tiny_fragment_ratio = _weighted_ratio(
        speaker1_vad.tiny_fragment_ratio,
        len(speaker1_vad.speech_segments),
        speaker2_vad.tiny_fragment_ratio,
        len(speaker2_vad.speech_segments),
    )
    long_segment_ratio = _weighted_ratio(
        speaker1_vad.long_segment_ratio,
        len(speaker1_vad.speech_segments),
        speaker2_vad.long_segment_ratio,
        len(speaker2_vad.speech_segments),
    )
    median_segment_duration_seconds = median(segment_durations)
    plausible_segment_duration_score = _plausible_segment_duration_score(
        median_segment_duration_seconds,
        tiny_fragment_ratio,
        long_segment_ratio,
        config,
    )
    event_density_stability_score = _event_density_stability_score(event_candidates)
    return TimingReliabilityMetrics(
        median_segment_duration_seconds=median_segment_duration_seconds,
        tiny_fragment_ratio=tiny_fragment_ratio,
        long_segment_ratio=long_segment_ratio,
        median_pause_duration_seconds=median(_event_gaps(event_candidates, EventType.PAUSE)),
        median_turn_gap_seconds=median(_turn_gap_seconds(event_candidates)),
        median_overlap_duration_seconds=median(_event_overlaps(event_candidates)),
        plausible_segment_duration_score=plausible_segment_duration_score,
        event_density_stability_score=event_density_stability_score,
        quality_score=clamp(
            (plausible_segment_duration_score + event_density_stability_score) / 2.0, 0.0, 1.0
        ),
    )


def combine_audio_quality_metrics(
    speaker1: TrackAudioQuality,
    speaker2: TrackAudioQuality,
    duration_gap_seconds: float,
    duration_gap_ratio: float,
    track_correlation: float | None,
    energy_envelope_correlation: float | None,
    speaker1_leakage_db: float | None,
    speaker2_leakage_db: float | None,
) -> AudioQualityMetrics:
    if speaker1.side != SpeakerSide.SPEAKER1:
        raise ValueError("speaker1 must contain speaker1 quality")
    if speaker2.side != SpeakerSide.SPEAKER2:
        raise ValueError("speaker2 must contain speaker2 quality")
    duration_score = penalty_above(duration_gap_ratio, 0.01, 0.10)
    correlation_score = (
        1.0 if track_correlation is None else penalty_above(abs(track_correlation), 0.15, 0.50)
    )
    envelope_score = (
        1.0
        if energy_envelope_correlation is None
        else penalty_above(max(0.0, energy_envelope_correlation), 0.20, 0.70)
    )
    leakage_score = min(
        leakage_db_score(speaker1_leakage_db), leakage_db_score(speaker2_leakage_db)
    )
    track_leakage_risk = (
        (track_correlation is not None and abs(track_correlation) >= 0.35)
        or (energy_envelope_correlation is not None and energy_envelope_correlation >= 0.35)
        or leakage_db_risk(speaker1_leakage_db)
        or leakage_db_risk(speaker2_leakage_db)
    )
    flags = _audio_quality_flags(speaker1, speaker2, duration_gap_ratio, track_leakage_risk)
    return AudioQualityMetrics(
        speaker1=speaker1,
        speaker2=speaker2,
        duration_gap_seconds=duration_gap_seconds,
        duration_gap_ratio=duration_gap_ratio,
        track_correlation=track_correlation,
        energy_envelope_correlation=energy_envelope_correlation,
        speaker1_leakage_db=speaker1_leakage_db,
        speaker2_leakage_db=speaker2_leakage_db,
        track_leakage_risk=track_leakage_risk,
        quality_score=clamp(
            (
                speaker1.quality_score
                + speaker2.quality_score
                + duration_score
                + correlation_score
                + envelope_score
                + leakage_score
            )
            / 6.0,
            0.0,
            1.0,
        ),
        flags=flags,
    )


def leakage_db_score(leakage_db: float | None) -> float:
    if leakage_db is None:
        return 1.0
    if leakage_db <= -30.0:
        return 1.0
    if leakage_db >= -12.0:
        return 0.0
    return clamp((-12.0 - leakage_db) / 18.0, 0.0, 1.0)


def leakage_db_risk(leakage_db: float | None) -> bool:
    return leakage_db is not None and leakage_db >= -20.0


def total_quality_score(
    interaction_density: InteractionDensityMetrics,
    timing_reliability: TimingReliabilityMetrics,
    audio_quality: AudioQualityMetrics,
    weights: QualityWeights,
) -> float:
    total_weight = weights.interaction_density + weights.timing_reliability + weights.audio_quality
    if total_weight <= 0.0:
        raise ValueError("quality weights must sum to a positive value")
    return clamp(
        (
            interaction_density.quality_score * weights.interaction_density
            + timing_reliability.quality_score * weights.timing_reliability
            + audio_quality.quality_score * weights.audio_quality
        )
        / total_weight,
        0.0,
        1.0,
    )


def calibrated_quality_score(
    raw_quality_score: float,
    interaction_density: InteractionDensityMetrics,
    timing_reliability: TimingReliabilityMetrics,
    audio_quality: AudioQualityMetrics,
) -> float:
    flags = calibration_flags(interaction_density, timing_reliability, audio_quality)
    return clamp(
        min(score_between(raw_quality_score, 0.70, 0.98), calibration_cap(flags)), 0.0, 1.0
    )


def calibration_flags(
    interaction_density: InteractionDensityMetrics,
    timing_reliability: TimingReliabilityMetrics,
    audio_quality: AudioQualityMetrics,
) -> tuple[str, ...]:
    flags: list[str] = []
    if (
        interaction_density.speech_ratio < 0.08
        or interaction_density.usable_candidate_windows_per_hour < 25.0
    ):
        flags.append("empty_or_no_interaction")
    if audio_quality.speaker1.low_information and audio_quality.speaker2.low_information:
        flags.append("both_tracks_low_information")
    if any("very_low_rms" in flag for flag in audio_quality.flags):
        flags.append("very_low_rms")
    if (
        audio_quality.energy_envelope_correlation is not None
        and audio_quality.energy_envelope_correlation >= 0.75
    ):
        flags.append("shared_energy_reverb_or_bleed")
    if leakage_db_risk(audio_quality.speaker1_leakage_db) or leakage_db_risk(
        audio_quality.speaker2_leakage_db
    ):
        flags.append("inactive_track_leakage")
    if interaction_density.overlap_ratio >= 0.45:
        flags.append("severe_overlap")
    elif interaction_density.overlap_ratio >= 0.30:
        flags.append("high_overlap")
    elif interaction_density.overlap_ratio >= 0.18 and interaction_density.speech_ratio >= 0.70:
        flags.append("dense_overlap")
    if audio_quality.duration_gap_ratio > 0.01 and interaction_density.speech_ratio < 0.35:
        flags.append("low_interaction_duration_mismatch")
    if timing_reliability.quality_score < 0.35:
        flags.append("poor_timing_plausibility")
    return tuple(flags)


def calibration_cap(flags: tuple[str, ...]) -> float:
    cap = 1.0
    if "empty_or_no_interaction" in flags or "both_tracks_low_information" in flags:
        cap = min(cap, 0.0)
    if "very_low_rms" in flags:
        cap = min(cap, 0.15)
    if "shared_energy_reverb_or_bleed" in flags:
        cap = min(cap, 0.15)
    if "severe_overlap" in flags:
        cap = min(cap, 0.45)
    if "high_overlap" in flags:
        cap = min(cap, 0.70)
    if "low_interaction_duration_mismatch" in flags:
        cap = min(cap, 0.30)
    if "poor_timing_plausibility" in flags:
        cap = min(cap, 0.25)
    return cap


def _count_event_types(event_candidates: tuple[EventCandidate, ...]) -> EventTypeCounts:
    return EventTypeCounts(
        turn_completions=_count_events(event_candidates, EventType.TURN_COMPLETION),
        pauses=_count_events(event_candidates, EventType.PAUSE),
        start_responses=_count_events(event_candidates, EventType.START_RESPONSE),
        interruptions=_count_events(event_candidates, EventType.INTERRUPTION),
        backchannels=_count_events(event_candidates, EventType.BACKCHANNEL),
        overlaps=_count_events(event_candidates, EventType.OVERLAP),
    )


def _count_events(event_candidates: tuple[EventCandidate, ...], event_type: EventType) -> int:
    return sum(
        1 for event_candidate in event_candidates if event_candidate.event_type == event_type
    )


def _union_speech_ratio(
    speaker1_vad: TrackVadResult, speaker2_vad: TrackVadResult, duration: float
) -> float:
    intervals = [
        TimeInterval(segment.start_seconds, segment.end_seconds)
        for segment in speaker1_vad.speech_segments + speaker2_vad.speech_segments
    ]
    return safe_ratio(_merged_interval_duration(intervals), duration)


def _overlap_ratio(
    speaker1_vad: TrackVadResult, speaker2_vad: TrackVadResult, duration: float
) -> float:
    intervals: list[TimeInterval] = []
    for speaker1_segment in speaker1_vad.speech_segments:
        for speaker2_segment in speaker2_vad.speech_segments:
            overlap_start_seconds = max(
                speaker1_segment.start_seconds, speaker2_segment.start_seconds
            )
            overlap_end_seconds = min(speaker1_segment.end_seconds, speaker2_segment.end_seconds)
            if overlap_end_seconds > overlap_start_seconds:
                intervals.append(TimeInterval(overlap_start_seconds, overlap_end_seconds))
    return safe_ratio(_merged_interval_duration(intervals), duration)


def _merged_interval_duration(intervals: list[TimeInterval]) -> float:
    if not intervals:
        return 0.0
    sorted_intervals = sorted(
        intervals, key=lambda interval: (interval.start_seconds, interval.end_seconds)
    )
    merged_intervals: list[TimeInterval] = [sorted_intervals[0]]
    for interval in sorted_intervals[1:]:
        previous_interval = merged_intervals[-1]
        if interval.start_seconds <= previous_interval.end_seconds:
            merged_intervals[-1] = TimeInterval(
                previous_interval.start_seconds,
                max(previous_interval.end_seconds, interval.end_seconds),
            )
            continue
        merged_intervals.append(interval)
    return sum(interval.end_seconds - interval.start_seconds for interval in merged_intervals)


def _weighted_ratio(
    first_ratio: float, first_count: int, second_ratio: float, second_count: int
) -> float:
    total_count = first_count + second_count
    if total_count == 0:
        return 0.0
    return ((first_ratio * first_count) + (second_ratio * second_count)) / total_count


def _plausible_segment_duration_score(
    median_segment_duration_seconds: float | None,
    tiny_fragment_ratio: float,
    long_segment_ratio: float,
    config: TimingScoringConfig,
) -> float:
    if median_segment_duration_seconds is None:
        return 0.0
    median_score = score_between(
        median_segment_duration_seconds,
        config.minimum_plausible_median_segment_seconds,
        config.target_plausible_median_segment_seconds,
    )
    tiny_score = penalty_above(
        tiny_fragment_ratio, config.maximum_tiny_fragment_ratio, config.rejected_tiny_fragment_ratio
    )
    long_score = penalty_above(
        long_segment_ratio, config.maximum_long_segment_ratio, config.rejected_long_segment_ratio
    )
    return clamp((median_score + tiny_score + long_score) / 3.0, 0.0, 1.0)


def _event_density_stability_score(event_candidates: tuple[EventCandidate, ...]) -> float:
    counts = _speaker_event_counts(event_candidates)
    total_count = counts.speaker1 + counts.speaker2
    if total_count == 0:
        return 0.0
    return clamp(1.0 - abs(counts.speaker1 - counts.speaker2) / total_count, 0.0, 1.0)


def _speaker_event_counts(event_candidates: tuple[EventCandidate, ...]) -> SpeakerEventCounts:
    return SpeakerEventCounts(
        speaker1=sum(
            1
            for event_candidate in event_candidates
            if event_candidate.primary_speaker == SpeakerSide.SPEAKER1
        ),
        speaker2=sum(
            1
            for event_candidate in event_candidates
            if event_candidate.primary_speaker == SpeakerSide.SPEAKER2
        ),
    )


def _event_gaps(event_candidates: tuple[EventCandidate, ...], event_type: EventType) -> list[float]:
    return [
        event_candidate.gap_seconds
        for event_candidate in event_candidates
        if event_candidate.event_type == event_type and event_candidate.gap_seconds is not None
    ]


def _turn_gap_seconds(event_candidates: tuple[EventCandidate, ...]) -> list[float]:
    return [
        event_candidate.gap_seconds
        for event_candidate in event_candidates
        if event_candidate.event_type in {EventType.TURN_COMPLETION, EventType.START_RESPONSE}
        and event_candidate.gap_seconds is not None
    ]


def _event_overlaps(event_candidates: tuple[EventCandidate, ...]) -> list[float]:
    return [
        event_candidate.overlap_seconds
        for event_candidate in event_candidates
        if event_candidate.overlap_seconds is not None
    ]


def _audio_quality_flags(
    speaker1: TrackAudioQuality,
    speaker2: TrackAudioQuality,
    duration_gap_ratio: float,
    track_leakage_risk: bool,
) -> tuple[str, ...]:
    flags: list[str] = []
    flags.extend(f"speaker1:{flag}" for flag in speaker1.flags)
    flags.extend(f"speaker2:{flag}" for flag in speaker2.flags)
    if duration_gap_ratio > 0.01:
        flags.append("duration_mismatch")
    if track_leakage_risk:
        flags.append("track_leakage_risk")
    return tuple(flags)
