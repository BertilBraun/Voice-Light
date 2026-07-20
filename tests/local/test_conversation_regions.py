from __future__ import annotations

import pytest

from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionAnalysis,
    ConversationRegionConfig,
    ConversationRegionReason,
)
from app.local.conversation_regions.service import analyze_conversation_regions
from app.shared.quality import (
    AnnotationSpan,
    ConversationAnnotation,
    SpeakerConversationAnnotation,
    SpeakerSide,
    SpeechSegment,
    TrackVadResult,
)


def test_conversation_regions_use_permissive_versioned_defaults() -> None:
    config = ConversationRegionConfig()

    assert config.minimum_dual_silence_seconds == 6.0
    assert config.maximum_one_sided_activity_seconds == 45.0
    assert config.maximum_turn_exchange_gap_seconds == 3.0

    analysis = analyze_conversation_regions(
        speaker1_vad=_vad(SpeakerSide.SPEAKER1, ((0.0, 10.0), (20.0, 30.0))),
        speaker2_vad=_vad(SpeakerSide.SPEAKER2, ((12.0, 18.0), (70.0, 75.0))),
        annotation=_annotation(
            duration_seconds=120.0,
            speaker1_spans=((0.0, 10.0), (20.0, 30.0)),
            speaker2_spans=((14.0, 18.0), (70.0, 75.0)),
        ),
        config=config,
    )

    assert analysis.analysis_version == CONVERSATION_REGION_ANALYSIS_VERSION
    assert _region_reasons_at(analysis, 11.0) == (ConversationRegionReason.SLOW_TURN_EXCHANGE,)
    assert ConversationRegionReason.DUAL_SILENCE in _region_reasons_at(analysis, 40.0)
    assert ConversationRegionReason.ONE_SIDED_ACTIVITY in _region_reasons_at(analysis, 72.0)
    assert analysis.usable_duration_seconds + analysis.unusable_duration_seconds == pytest.approx(
        120.0
    )


def test_short_silence_and_normal_handoffs_remain_usable() -> None:
    analysis = analyze_conversation_regions(
        speaker1_vad=_vad(SpeakerSide.SPEAKER1, ((0.0, 8.0), (18.0, 28.0))),
        speaker2_vad=_vad(SpeakerSide.SPEAKER2, ((9.0, 17.0), (29.0, 40.0))),
        annotation=_annotation(
            duration_seconds=40.0,
            speaker1_spans=((0.0, 8.0), (18.0, 28.0)),
            speaker2_spans=((9.0, 17.0), (29.0, 40.0)),
        ),
        config=ConversationRegionConfig(),
    )

    assert analysis.unusable_regions == ()
    assert analysis.usable_ratio == pytest.approx(1.0)


def _region_reasons_at(
    analysis: ConversationRegionAnalysis,
    time_seconds: float,
) -> tuple[ConversationRegionReason, ...]:
    matching = [
        region
        for region in analysis.unusable_regions
        if region.start_seconds <= time_seconds < region.end_seconds
    ]
    return matching[0].reasons if matching else ()


def _vad(
    side: SpeakerSide,
    spans: tuple[tuple[float, float], ...],
) -> TrackVadResult:
    speech_segments = tuple(
        SpeechSegment(start_seconds=start_seconds, end_seconds=end_seconds)
        for start_seconds, end_seconds in spans
    )
    speech_time_seconds = sum(
        segment.end_seconds - segment.start_seconds for segment in speech_segments
    )
    return TrackVadResult(
        side=side,
        speech_segments=speech_segments,
        speech_time_seconds=speech_time_seconds,
        speech_ratio=speech_time_seconds / 120.0,
        median_segment_duration_seconds=None,
        tiny_fragment_ratio=0.0,
        long_segment_ratio=0.0,
    )


def _annotation(
    duration_seconds: float,
    speaker1_spans: tuple[tuple[float, float], ...],
    speaker2_spans: tuple[tuple[float, float], ...],
) -> ConversationAnnotation:
    speaker1 = _speaker_annotation(SpeakerSide.SPEAKER1, speaker1_spans)
    speaker2 = _speaker_annotation(SpeakerSide.SPEAKER2, speaker2_spans)
    return ConversationAnnotation(
        annotation_version="test-annotation",
        analyzed_duration_seconds=duration_seconds,
        speaker1=speaker1,
        speaker2=speaker2,
        speech_segment_count=len(speaker1_spans) + len(speaker2_spans),
        turn_count=0,
        turn_taking_count=0,
        interaction_count=0,
        pause_count=0,
        backchannel_count=0,
        interruption_count=0,
        usable_event_count=0,
        events_per_hour=0.0,
        speaker_balance_score=1.0,
        quality_score=1.0,
    )


def _speaker_annotation(
    side: SpeakerSide,
    spans: tuple[tuple[float, float], ...],
) -> SpeakerConversationAnnotation:
    speech_segments = tuple(
        AnnotationSpan(start_seconds=start_seconds, end_seconds=end_seconds, text=None)
        for start_seconds, end_seconds in spans
    )
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=speech_segments,
        pauses=(),
        backchannels=(),
        turns=(),
        interruptions=(),
        segment_targets=(),
        connection_targets=(),
        speech_duration_seconds=sum(
            span.end_seconds - span.start_seconds for span in speech_segments
        ),
        pause_duration_seconds=0.0,
        backchannel_duration_seconds=0.0,
    )
