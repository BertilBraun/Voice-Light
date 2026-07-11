from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from app.quality.models import EventCandidate, EventType, SpeakerSide, SpeechSegment, TrackVadResult


@dataclass(frozen=True)
class EventExtractionConfig:
    minimum_turn_gap_seconds: float = 0.05
    maximum_turn_gap_seconds: float = 2.0
    minimum_pause_seconds: float = 0.75
    maximum_start_response_seconds: float = 0.70
    minimum_overlap_seconds: float = 0.12
    minimum_interruption_overlap_seconds: float = 0.20
    maximum_backchannel_duration_seconds: float = 0.85
    minimum_primary_segment_seconds: float = 1.00


@dataclass(frozen=True)
class SpeakerSegment:
    side: SpeakerSide
    segment: SpeechSegment


@dataclass(frozen=True)
class SpeakerTransition:
    previous: SpeakerSegment
    next: SpeakerSegment
    gap_seconds: float


DEFAULT_EVENT_EXTRACTION_CONFIG = EventExtractionConfig()


def extract_event_candidates(
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
    total_duration_seconds: float,
    config: EventExtractionConfig = DEFAULT_EVENT_EXTRACTION_CONFIG,
) -> tuple[EventCandidate, ...]:
    if total_duration_seconds <= 0.0:
        raise ValueError("total_duration_seconds must be positive")
    if speaker1_vad.side == speaker2_vad.side:
        raise ValueError("VAD results must be for different speaker sides")

    events: list[EventCandidate] = []
    events.extend(_extract_transition_events(speaker1_vad, speaker2_vad, config))
    events.extend(_extract_pause_events(speaker1_vad, speaker2_vad, config))
    events.extend(_extract_pause_events(speaker2_vad, speaker1_vad, config))
    events.extend(_extract_overlap_events(speaker1_vad, speaker2_vad, config))
    return tuple(sorted(events, key=_event_sort_key))


def _extract_transition_events(
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
    config: EventExtractionConfig,
) -> list[EventCandidate]:
    events: list[EventCandidate] = []
    for transition in _cross_speaker_transitions(speaker1_vad, speaker2_vad):
        if transition.gap_seconds < config.minimum_turn_gap_seconds:
            continue
        if transition.gap_seconds <= config.maximum_turn_gap_seconds:
            events.append(
                EventCandidate(
                    event_type=EventType.TURN_COMPLETION,
                    primary_speaker=transition.previous.side,
                    secondary_speaker=transition.next.side,
                    start_seconds=transition.previous.segment.end_seconds,
                    end_seconds=transition.next.segment.start_seconds,
                    gap_seconds=transition.gap_seconds,
                    overlap_seconds=None,
                )
            )
        if transition.gap_seconds <= config.maximum_start_response_seconds:
            events.append(
                EventCandidate(
                    event_type=EventType.START_RESPONSE,
                    primary_speaker=transition.next.side,
                    secondary_speaker=transition.previous.side,
                    start_seconds=transition.next.segment.start_seconds,
                    end_seconds=transition.next.segment.end_seconds,
                    gap_seconds=transition.gap_seconds,
                    overlap_seconds=None,
                )
            )
    return events


def _extract_pause_events(
    primary_vad: TrackVadResult,
    secondary_vad: TrackVadResult,
    config: EventExtractionConfig,
) -> list[EventCandidate]:
    events: list[EventCandidate] = []
    for previous_segment, next_segment in pairwise(primary_vad.speech_segments):
        gap_seconds = next_segment.start_seconds - previous_segment.end_seconds
        if gap_seconds < config.minimum_pause_seconds:
            continue
        if _has_secondary_activity(
            secondary_vad, previous_segment.end_seconds, next_segment.start_seconds
        ):
            continue
        events.append(
            EventCandidate(
                event_type=EventType.PAUSE,
                primary_speaker=primary_vad.side,
                secondary_speaker=None,
                start_seconds=previous_segment.end_seconds,
                end_seconds=next_segment.start_seconds,
                gap_seconds=gap_seconds,
                overlap_seconds=None,
            )
        )
    return events


def _has_secondary_activity(
    secondary_vad: TrackVadResult, start_seconds: float, end_seconds: float
) -> bool:
    for segment in secondary_vad.speech_segments:
        if min(segment.end_seconds, end_seconds) > max(segment.start_seconds, start_seconds):
            return True
    return False


def _extract_overlap_events(
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
    config: EventExtractionConfig,
) -> list[EventCandidate]:
    events: list[EventCandidate] = []
    for speaker1_segment in speaker1_vad.speech_segments:
        for speaker2_segment in speaker2_vad.speech_segments:
            overlap_seconds = _segment_overlap_seconds(speaker1_segment, speaker2_segment)
            if overlap_seconds < config.minimum_overlap_seconds:
                continue
            overlap_start_seconds = max(
                speaker1_segment.start_seconds, speaker2_segment.start_seconds
            )
            overlap_end_seconds = min(speaker1_segment.end_seconds, speaker2_segment.end_seconds)
            events.append(
                EventCandidate(
                    event_type=EventType.OVERLAP,
                    primary_speaker=speaker1_vad.side,
                    secondary_speaker=speaker2_vad.side,
                    start_seconds=overlap_start_seconds,
                    end_seconds=overlap_end_seconds,
                    gap_seconds=None,
                    overlap_seconds=overlap_seconds,
                )
            )
            events.extend(
                _classify_overlap(
                    first=SpeakerSegment(speaker1_vad.side, speaker1_segment),
                    second=SpeakerSegment(speaker2_vad.side, speaker2_segment),
                    overlap_seconds=overlap_seconds,
                    overlap_end_seconds=overlap_end_seconds,
                    config=config,
                )
            )
    return events


def _classify_overlap(
    first: SpeakerSegment,
    second: SpeakerSegment,
    overlap_seconds: float,
    overlap_end_seconds: float,
    config: EventExtractionConfig,
) -> list[EventCandidate]:
    short_segment = _short_backchannel_segment(first, second, config)
    if short_segment is not None:
        secondary_segment = _other_segment(first, second, short_segment)
        return [
            EventCandidate(
                event_type=EventType.BACKCHANNEL,
                primary_speaker=short_segment.side,
                secondary_speaker=secondary_segment.side,
                start_seconds=short_segment.segment.start_seconds,
                end_seconds=short_segment.segment.end_seconds,
                gap_seconds=None,
                overlap_seconds=overlap_seconds,
            )
        ]
    interrupter = _interrupter_segment(first, second, overlap_seconds, config)
    if interrupter is None:
        return []
    interrupted = _other_segment(first, second, interrupter)
    return [
        EventCandidate(
            event_type=EventType.INTERRUPTION,
            primary_speaker=interrupter.side,
            secondary_speaker=interrupted.side,
            start_seconds=interrupter.segment.start_seconds,
            end_seconds=overlap_end_seconds,
            gap_seconds=None,
            overlap_seconds=overlap_seconds,
        )
    ]


def _cross_speaker_transitions(
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
) -> list[SpeakerTransition]:
    segments = _sorted_segments(speaker1_vad, speaker2_vad)
    transitions: list[SpeakerTransition] = []
    for previous_segment, next_segment in pairwise(segments):
        if previous_segment.side == next_segment.side:
            continue
        gap_seconds = next_segment.segment.start_seconds - previous_segment.segment.end_seconds
        if gap_seconds >= 0.0:
            transitions.append(SpeakerTransition(previous_segment, next_segment, gap_seconds))
    return transitions


def _sorted_segments(
    speaker1_vad: TrackVadResult, speaker2_vad: TrackVadResult
) -> list[SpeakerSegment]:
    segments = [
        SpeakerSegment(speaker1_vad.side, segment) for segment in speaker1_vad.speech_segments
    ] + [SpeakerSegment(speaker2_vad.side, segment) for segment in speaker2_vad.speech_segments]
    return sorted(segments, key=_segment_sort_key)


def _segment_sort_key(speaker_segment: SpeakerSegment) -> tuple[float, float, str]:
    return (
        speaker_segment.segment.start_seconds,
        speaker_segment.segment.end_seconds,
        speaker_segment.side.value,
    )


def _event_sort_key(event_candidate: EventCandidate) -> tuple[float, float, str, str]:
    return (
        event_candidate.start_seconds,
        event_candidate.end_seconds,
        event_candidate.event_type.value,
        event_candidate.primary_speaker.value,
    )


def _segment_overlap_seconds(first: SpeechSegment, second: SpeechSegment) -> float:
    return max(
        0.0,
        min(first.end_seconds, second.end_seconds) - max(first.start_seconds, second.start_seconds),
    )


def _short_backchannel_segment(
    first: SpeakerSegment,
    second: SpeakerSegment,
    config: EventExtractionConfig,
) -> SpeakerSegment | None:
    if _is_backchannel_inside(first, second, config):
        return first
    if _is_backchannel_inside(second, first, config):
        return second
    return None


def _is_backchannel_inside(
    primary: SpeakerSegment,
    secondary: SpeakerSegment,
    config: EventExtractionConfig,
) -> bool:
    return (
        primary.segment.duration_seconds <= config.maximum_backchannel_duration_seconds
        and secondary.segment.duration_seconds >= config.minimum_primary_segment_seconds
        and primary.segment.start_seconds >= secondary.segment.start_seconds
        and primary.segment.end_seconds <= secondary.segment.end_seconds
    )


def _interrupter_segment(
    first: SpeakerSegment,
    second: SpeakerSegment,
    overlap_seconds: float,
    config: EventExtractionConfig,
) -> SpeakerSegment | None:
    if overlap_seconds < config.minimum_interruption_overlap_seconds:
        return None
    if first.segment.start_seconds < second.segment.start_seconds < first.segment.end_seconds:
        return second
    if second.segment.start_seconds < first.segment.start_seconds < second.segment.end_seconds:
        return first
    return None


def _other_segment(
    first: SpeakerSegment,
    second: SpeakerSegment,
    selected: SpeakerSegment,
) -> SpeakerSegment:
    if selected == first:
        return second
    return first
