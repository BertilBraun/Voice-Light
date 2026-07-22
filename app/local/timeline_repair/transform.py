from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from app.local.asr.transcript import SpeakerTrack, Word
from app.shared.quality import SpeechSegment


@dataclass(frozen=True)
class GlobalTimelineRepair:
    shift_seconds: float


@dataclass(frozen=True)
class PiecewiseTimelineRepair:
    first_shift_seconds: float
    second_shift_seconds: float
    canonical_transition_seconds: float
    exclusion_start_seconds: float
    exclusion_end_seconds: float

    def __post_init__(self) -> None:
        if self.canonical_transition_seconds < 0.0:
            raise ValueError("Transition time cannot be negative.")
        if self.exclusion_start_seconds < 0.0:
            raise ValueError("Exclusion start cannot be negative.")
        if self.exclusion_end_seconds < self.exclusion_start_seconds:
            raise ValueError("Exclusion end cannot precede its start.")
        if (
            not self.exclusion_start_seconds
            <= self.canonical_transition_seconds
            <= self.exclusion_end_seconds
        ):
            raise ValueError("The transition must lie inside its exclusion window.")


TimelineRepair: TypeAlias = GlobalTimelineRepair | PiecewiseTimelineRepair


@dataclass(frozen=True)
class AudioIntervalMapping:
    canonical_start_seconds: float
    canonical_end_seconds: float
    source_start_seconds: float
    source_end_seconds: float
    shift_seconds: float


@dataclass(frozen=True)
class CanonicalInterval:
    start_seconds: float
    end_seconds: float


def transform_words(
    words: tuple[Word, ...],
    track: SpeakerTrack,
    repair: TimelineRepair,
) -> tuple[Word, ...]:
    if track is SpeakerTrack.SPEAKER1:
        return words
    transformed: list[Word] = []
    for word in words:
        if word.start_seconds is None or word.end_seconds is None:
            raise ValueError(f"Cannot repair an untimed ASR word: {word.text}")
        if word.end_seconds < word.start_seconds:
            raise ValueError(f"ASR word end precedes start: {word.text}")
        shift_seconds = _word_shift(word=word, repair=repair)
        if shift_seconds is None:
            continue
        transformed.append(
            word.model_copy(
                update={
                    "start_seconds": word.start_seconds + shift_seconds,
                    "end_seconds": word.end_seconds + shift_seconds,
                }
            )
        )
    return tuple(transformed)


def transform_speech_segments(
    segments: tuple[SpeechSegment, ...],
    track: SpeakerTrack,
    repair: TimelineRepair,
) -> tuple[SpeechSegment, ...]:
    if track is SpeakerTrack.SPEAKER1:
        return segments
    transformed = [
        shifted_segment
        for segment in segments
        for shifted_segment in _shifted_segments(segment=segment, repair=repair)
    ]
    return tuple(
        sorted(transformed, key=lambda segment: (segment.start_seconds, segment.end_seconds))
    )


def interval_intersects_exclusion(
    start_seconds: float,
    end_seconds: float,
    repair: TimelineRepair,
) -> bool:
    _validate_interval(start_seconds=start_seconds, end_seconds=end_seconds)
    match repair:
        case GlobalTimelineRepair():
            return False
        case PiecewiseTimelineRepair():
            return (
                start_seconds < repair.exclusion_end_seconds
                and end_seconds > repair.exclusion_start_seconds
            )


def map_canonical_audio_interval(
    track: SpeakerTrack,
    canonical_start_seconds: float,
    canonical_end_seconds: float,
    source_duration_seconds: float,
    repair: TimelineRepair,
) -> AudioIntervalMapping:
    _validate_interval(
        start_seconds=canonical_start_seconds,
        end_seconds=canonical_end_seconds,
    )
    if source_duration_seconds < 0.0:
        raise ValueError("Source duration cannot be negative.")
    shift_seconds = _audio_interval_shift(
        track=track,
        canonical_start_seconds=canonical_start_seconds,
        canonical_end_seconds=canonical_end_seconds,
        repair=repair,
    )
    source_start_seconds = canonical_start_seconds - shift_seconds
    source_end_seconds = canonical_end_seconds - shift_seconds
    if source_start_seconds < 0.0 or source_end_seconds > source_duration_seconds:
        raise ValueError("Canonical interval maps outside the source audio.")
    return AudioIntervalMapping(
        canonical_start_seconds=canonical_start_seconds,
        canonical_end_seconds=canonical_end_seconds,
        source_start_seconds=source_start_seconds,
        source_end_seconds=source_end_seconds,
        shift_seconds=shift_seconds,
    )


def canonical_available_intervals(
    track: SpeakerTrack,
    source_duration_seconds: float,
    repair: TimelineRepair,
) -> tuple[CanonicalInterval, ...]:
    if source_duration_seconds < 0.0:
        raise ValueError("Source duration cannot be negative.")
    match repair:
        case GlobalTimelineRepair():
            shift_seconds = 0.0 if track is SpeakerTrack.SPEAKER1 else repair.shift_seconds
            return _nonempty_canonical_intervals(
                (max(0.0, shift_seconds), source_duration_seconds + shift_seconds)
            )
        case PiecewiseTimelineRepair():
            if track is SpeakerTrack.SPEAKER1:
                first_shift_seconds = 0.0
                second_shift_seconds = 0.0
            else:
                first_shift_seconds = repair.first_shift_seconds
                second_shift_seconds = repair.second_shift_seconds
            return _nonempty_canonical_intervals(
                (
                    max(0.0, first_shift_seconds),
                    min(
                        repair.exclusion_start_seconds,
                        source_duration_seconds + first_shift_seconds,
                    ),
                ),
                (
                    max(0.0, repair.exclusion_end_seconds, second_shift_seconds),
                    source_duration_seconds + second_shift_seconds,
                ),
            )


def _word_shift(word: Word, repair: TimelineRepair) -> float | None:
    match repair:
        case GlobalTimelineRepair():
            return repair.shift_seconds
        case PiecewiseTimelineRepair():
            assert word.start_seconds is not None
            assert word.end_seconds is not None
            if word.end_seconds + repair.first_shift_seconds <= repair.canonical_transition_seconds:
                return repair.first_shift_seconds
            if (
                word.start_seconds + repair.second_shift_seconds
                >= repair.canonical_transition_seconds
            ):
                return repair.second_shift_seconds
            return None


def _shifted_segments(
    segment: SpeechSegment,
    repair: TimelineRepair,
) -> tuple[SpeechSegment, ...]:
    if segment.end_seconds < segment.start_seconds:
        raise ValueError("Speech segment end cannot precede its start.")
    match repair:
        case GlobalTimelineRepair():
            return (_shifted_segment(segment=segment, shift_seconds=repair.shift_seconds),)
        case PiecewiseTimelineRepair():
            source_boundary_seconds = (
                repair.canonical_transition_seconds
                - (repair.first_shift_seconds + repair.second_shift_seconds) / 2.0
            )
            if segment.end_seconds <= source_boundary_seconds:
                return (_shifted_segment(segment, repair.first_shift_seconds),)
            if segment.start_seconds >= source_boundary_seconds:
                return (_shifted_segment(segment, repair.second_shift_seconds),)
            return (
                _shifted_segment(
                    SpeechSegment(
                        start_seconds=segment.start_seconds,
                        end_seconds=source_boundary_seconds,
                    ),
                    repair.first_shift_seconds,
                ),
                _shifted_segment(
                    SpeechSegment(
                        start_seconds=source_boundary_seconds,
                        end_seconds=segment.end_seconds,
                    ),
                    repair.second_shift_seconds,
                ),
            )


def _shifted_segment(segment: SpeechSegment, shift_seconds: float) -> SpeechSegment:
    return SpeechSegment(
        start_seconds=segment.start_seconds + shift_seconds,
        end_seconds=segment.end_seconds + shift_seconds,
    )


def _audio_interval_shift(
    track: SpeakerTrack,
    canonical_start_seconds: float,
    canonical_end_seconds: float,
    repair: TimelineRepair,
) -> float:
    if track is SpeakerTrack.SPEAKER1:
        return 0.0
    match repair:
        case GlobalTimelineRepair():
            return repair.shift_seconds
        case PiecewiseTimelineRepair():
            if interval_intersects_exclusion(
                start_seconds=canonical_start_seconds,
                end_seconds=canonical_end_seconds,
                repair=repair,
            ):
                raise ValueError("Canonical interval intersects the repair exclusion window.")
            if canonical_end_seconds <= repair.exclusion_start_seconds:
                return repair.first_shift_seconds
            return repair.second_shift_seconds


def _validate_interval(start_seconds: float, end_seconds: float) -> None:
    if start_seconds < 0.0:
        raise ValueError("Interval start cannot be negative.")
    if end_seconds <= start_seconds:
        raise ValueError("Interval end must follow its start.")


def _nonempty_canonical_intervals(
    *bounds: tuple[float, float],
) -> tuple[CanonicalInterval, ...]:
    return tuple(
        CanonicalInterval(start_seconds=start_seconds, end_seconds=end_seconds)
        for start_seconds, end_seconds in bounds
        if end_seconds > start_seconds
    )
