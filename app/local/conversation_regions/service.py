from __future__ import annotations

from dataclasses import dataclass

from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionAnalysis,
    ConversationRegionConfig,
    ConversationRegionReason,
    UnusableConversationRegion,
)
from app.shared.quality import (
    ConversationAnnotation,
    SpeakerSide,
    SpeechSegment,
    TrackVadResult,
)


@dataclass(frozen=True)
class TimeRange:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class ReasonedTimeRange:
    start_seconds: float
    end_seconds: float
    reason: ConversationRegionReason


@dataclass(frozen=True)
class TaggedSpeechSpan:
    side: SpeakerSide
    start_seconds: float
    end_seconds: float


def analyze_conversation_regions(
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
    annotation: ConversationAnnotation,
    config: ConversationRegionConfig,
) -> ConversationRegionAnalysis:
    if speaker1_vad.side is not SpeakerSide.SPEAKER1:
        raise ValueError("speaker1_vad must describe speaker1.")
    if speaker2_vad.side is not SpeakerSide.SPEAKER2:
        raise ValueError("speaker2_vad must describe speaker2.")
    duration_seconds = annotation.analyzed_duration_seconds
    candidate_ranges = [
        *_dual_silence_ranges(
            speaker1_vad=speaker1_vad,
            speaker2_vad=speaker2_vad,
            duration_seconds=duration_seconds,
            minimum_duration_seconds=config.minimum_dual_silence_seconds,
        ),
        *_one_sided_activity_ranges(
            absent_segments=speaker1_vad.speech_segments,
            present_segments=speaker2_vad.speech_segments,
            duration_seconds=duration_seconds,
            config=config,
        ),
        *_one_sided_activity_ranges(
            absent_segments=speaker2_vad.speech_segments,
            present_segments=speaker1_vad.speech_segments,
            duration_seconds=duration_seconds,
            config=config,
        ),
        *_slow_turn_exchange_ranges(annotation=annotation, config=config),
    ]
    unusable_regions = _merge_reasoned_ranges(
        ranges=candidate_ranges,
        duration_seconds=duration_seconds,
    )
    unusable_duration_seconds = sum(
        region.end_seconds - region.start_seconds for region in unusable_regions
    )
    usable_duration_seconds = max(0.0, duration_seconds - unusable_duration_seconds)
    return ConversationRegionAnalysis(
        analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        annotation_version=annotation.annotation_version,
        config=config,
        duration_seconds=duration_seconds,
        usable_duration_seconds=usable_duration_seconds,
        unusable_duration_seconds=unusable_duration_seconds,
        usable_ratio=usable_duration_seconds / duration_seconds,
        unusable_regions=unusable_regions,
    )


def _dual_silence_ranges(
    speaker1_vad: TrackVadResult,
    speaker2_vad: TrackVadResult,
    duration_seconds: float,
    minimum_duration_seconds: float,
) -> tuple[ReasonedTimeRange, ...]:
    active_ranges = _merge_time_ranges(
        ranges=tuple(
            TimeRange(segment.start_seconds, segment.end_seconds)
            for segment in (*speaker1_vad.speech_segments, *speaker2_vad.speech_segments)
        ),
        duration_seconds=duration_seconds,
    )
    silence_ranges = _gaps(active_ranges, duration_seconds)
    return tuple(
        ReasonedTimeRange(
            start_seconds=gap.start_seconds,
            end_seconds=gap.end_seconds,
            reason=ConversationRegionReason.DUAL_SILENCE,
        )
        for gap in silence_ranges
        if gap.end_seconds - gap.start_seconds >= minimum_duration_seconds
    )


def _one_sided_activity_ranges(
    absent_segments: tuple[SpeechSegment, ...],
    present_segments: tuple[SpeechSegment, ...],
    duration_seconds: float,
    config: ConversationRegionConfig,
) -> tuple[ReasonedTimeRange, ...]:
    absent_ranges = _merge_time_ranges(
        ranges=tuple(
            TimeRange(segment.start_seconds, segment.end_seconds) for segment in absent_segments
        ),
        duration_seconds=duration_seconds,
    )
    present_ranges = _merge_time_ranges(
        ranges=tuple(
            TimeRange(segment.start_seconds, segment.end_seconds) for segment in present_segments
        ),
        duration_seconds=duration_seconds,
    )
    candidates: list[ReasonedTimeRange] = []
    for absent_gap in _gaps(absent_ranges, duration_seconds):
        if (
            absent_gap.end_seconds - absent_gap.start_seconds
            < config.maximum_one_sided_activity_seconds
        ):
            continue
        overlapping_present = tuple(
            intersection
            for present_range in present_ranges
            if (intersection := _intersection(absent_gap, present_range)) is not None
        )
        present_speech_seconds = sum(
            overlap.end_seconds - overlap.start_seconds for overlap in overlapping_present
        )
        if present_speech_seconds < config.minimum_present_speech_seconds:
            continue
        candidates.append(
            ReasonedTimeRange(
                start_seconds=overlapping_present[0].start_seconds,
                end_seconds=overlapping_present[-1].end_seconds,
                reason=ConversationRegionReason.ONE_SIDED_ACTIVITY,
            )
        )
    return tuple(candidates)


def _slow_turn_exchange_ranges(
    annotation: ConversationAnnotation,
    config: ConversationRegionConfig,
) -> tuple[ReasonedTimeRange, ...]:
    speech_spans = sorted(
        (
            *(
                TaggedSpeechSpan(
                    side=SpeakerSide.SPEAKER1,
                    start_seconds=span.start_seconds,
                    end_seconds=span.end_seconds,
                )
                for span in annotation.speaker1.speech_segments
            ),
            *(
                TaggedSpeechSpan(
                    side=SpeakerSide.SPEAKER2,
                    start_seconds=span.start_seconds,
                    end_seconds=span.end_seconds,
                )
                for span in annotation.speaker2.speech_segments
            ),
        ),
        key=lambda span: (span.start_seconds, span.end_seconds, span.side.value),
    )
    candidates: list[ReasonedTimeRange] = []
    for earlier, later in zip(speech_spans, speech_spans[1:], strict=False):
        if earlier.side is later.side:
            continue
        gap_seconds = later.start_seconds - earlier.end_seconds
        if gap_seconds < config.maximum_turn_exchange_gap_seconds:
            continue
        candidates.append(
            ReasonedTimeRange(
                start_seconds=earlier.end_seconds,
                end_seconds=later.start_seconds,
                reason=ConversationRegionReason.SLOW_TURN_EXCHANGE,
            )
        )
    return tuple(candidates)


def _merge_time_ranges(
    ranges: tuple[TimeRange, ...],
    duration_seconds: float,
) -> tuple[TimeRange, ...]:
    clipped = sorted(
        (
            TimeRange(
                start_seconds=max(0.0, time_range.start_seconds),
                end_seconds=min(duration_seconds, time_range.end_seconds),
            )
            for time_range in ranges
            if min(duration_seconds, time_range.end_seconds) > max(0.0, time_range.start_seconds)
        ),
        key=lambda time_range: (time_range.start_seconds, time_range.end_seconds),
    )
    merged: list[TimeRange] = []
    for time_range in clipped:
        if not merged or time_range.start_seconds > merged[-1].end_seconds:
            merged.append(time_range)
            continue
        previous = merged[-1]
        merged[-1] = TimeRange(
            start_seconds=previous.start_seconds,
            end_seconds=max(previous.end_seconds, time_range.end_seconds),
        )
    return tuple(merged)


def _gaps(ranges: tuple[TimeRange, ...], duration_seconds: float) -> tuple[TimeRange, ...]:
    gaps: list[TimeRange] = []
    cursor_seconds = 0.0
    for time_range in ranges:
        if time_range.start_seconds > cursor_seconds:
            gaps.append(TimeRange(cursor_seconds, time_range.start_seconds))
        cursor_seconds = max(cursor_seconds, time_range.end_seconds)
    if cursor_seconds < duration_seconds:
        gaps.append(TimeRange(cursor_seconds, duration_seconds))
    return tuple(gaps)


def _intersection(first: TimeRange, second: TimeRange) -> TimeRange | None:
    start_seconds = max(first.start_seconds, second.start_seconds)
    end_seconds = min(first.end_seconds, second.end_seconds)
    return (
        TimeRange(start_seconds=start_seconds, end_seconds=end_seconds)
        if end_seconds > start_seconds
        else None
    )


def _merge_reasoned_ranges(
    ranges: list[ReasonedTimeRange],
    duration_seconds: float,
) -> tuple[UnusableConversationRegion, ...]:
    boundaries = sorted(
        {
            max(0.0, min(duration_seconds, boundary))
            for time_range in ranges
            for boundary in (time_range.start_seconds, time_range.end_seconds)
        }
    )
    regions: list[UnusableConversationRegion] = []
    for start_seconds, end_seconds in zip(boundaries, boundaries[1:], strict=False):
        if end_seconds <= start_seconds:
            continue
        reasons = tuple(
            sorted(
                {
                    time_range.reason
                    for time_range in ranges
                    if time_range.start_seconds < end_seconds
                    and time_range.end_seconds > start_seconds
                },
                key=lambda reason: reason.value,
            )
        )
        if not reasons:
            continue
        if regions and regions[-1].end_seconds == start_seconds and regions[-1].reasons == reasons:
            previous = regions[-1]
            regions[-1] = UnusableConversationRegion(
                start_seconds=previous.start_seconds,
                end_seconds=end_seconds,
                reasons=reasons,
            )
            continue
        regions.append(
            UnusableConversationRegion(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                reasons=reasons,
            )
        )
    return tuple(regions)
