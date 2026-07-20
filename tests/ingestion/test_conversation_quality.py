from __future__ import annotations

import pytest

from app.local.analyses.end_of_turn.detectors.two_speaker_annotation import (
    OTHER_SPEAKER,
    TARGET_SPEAKER,
    TranscriptTurn,
)
from app.local.db.models import JobStatus, SampleLanguageStatus, SampleListFilter
from app.local.db.repository import dashboard_filter_sql
from app.local.ingestion.conversation import conversation_annotation
from app.local.ingestion.service import (
    ingestion_summary,
)
from app.shared.quality import (
    AnnotationPoint,
    AnnotationSpan,
    SpeakerConversationAnnotation,
    SpeakerSide,
)


def test_dashboard_summary_filter_reuses_sample_filters() -> None:
    filter_sql = dashboard_filter_sql(
        SampleListFilter(
            quality_min=0.9,
            overlap_ratio_max=0.1,
            flag="track_leakage_risk",
            language_status=SampleLanguageStatus.NON_ENGLISH,
        )
    )

    assert "latest_quality.total_quality_score >= %s" in filter_sql.where_clause
    assert "latest_quality.overlap_ratio <= %s" in filter_sql.where_clause
    assert "%s = ANY(samples.quality_flags)" in filter_sql.where_clause
    assert "language_summary.language_status = %s" in filter_sql.where_clause
    assert filter_sql.parameters == (0.9, "track_leakage_risk", 0.1, "non_english")


@pytest.mark.parametrize(
    ("processed_samples", "failed_samples", "sample_errors", "expected_message"),
    [
        (0, 1, ("sample: RuntimeError: failed",), "Ingestion failed for 1 of 1 samples"),
        (2, 1, ("sample: RuntimeError: failed",), "Ingestion failed for 1 of 3 samples"),
    ],
)
def test_ingestion_summary_reports_sample_failures(
    processed_samples: int,
    failed_samples: int,
    sample_errors: tuple[str, ...],
    expected_message: str,
) -> None:
    summary = ingestion_summary(
        processed_samples=processed_samples,
        analyzed_samples=processed_samples,
        language_excluded_samples=0,
        failed_samples=failed_samples,
        sample_errors=sample_errors,
    )

    assert summary.status is JobStatus.FAILED
    assert summary.message.startswith(expected_message)
    assert summary.error == sample_errors[0]


def test_ingestion_summary_completes_without_failures() -> None:
    summary = ingestion_summary(
        processed_samples=2,
        analyzed_samples=1,
        language_excluded_samples=1,
        failed_samples=0,
        sample_errors=(),
    )

    assert summary.status is JobStatus.COMPLETED
    assert summary.message == "Ingestion completed; analyzed 1; language-excluded 1"
    assert summary.error is None


def test_conversation_annotation_counts_speaker_transitions_and_events() -> None:
    speaker1 = speaker_annotation(
        side=SpeakerSide.SPEAKER1,
        speech_duration_seconds=30.0,
        turn_count=2,
        pause_count=1,
        backchannel_count=1,
        interruption_count=0,
    )
    speaker2 = speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_duration_seconds=20.0,
        turn_count=1,
        pause_count=0,
        backchannel_count=0,
        interruption_count=1,
    )
    turns = [
        transcript_turn(TARGET_SPEAKER, 0.0, 10.0),
        transcript_turn(OTHER_SPEAKER, 10.0, 20.0),
        transcript_turn(OTHER_SPEAKER, 20.0, 30.0),
        transcript_turn(TARGET_SPEAKER, 30.0, 40.0),
    ]

    annotation = conversation_annotation(
        turns=turns,
        duration_seconds=60.0,
        speaker1=speaker1,
        speaker2=speaker2,
    )

    assert annotation.turn_count == 3
    assert annotation.turn_taking_count == 2
    assert annotation.interaction_count == 4
    assert annotation.pause_count == 1
    assert annotation.backchannel_count == 1
    assert annotation.interruption_count == 1
    assert annotation.usable_event_count == 6
    assert annotation.events_per_hour == 360.0
    assert annotation.speaker_balance_score == 0.8


def speaker_annotation(
    side: SpeakerSide,
    speech_duration_seconds: float,
    turn_count: int,
    pause_count: int,
    backchannel_count: int,
    interruption_count: int,
) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=(),
        pauses=tuple(
            AnnotationSpan(start_seconds=0.0, end_seconds=1.0, text=None)
            for _ in range(pause_count)
        ),
        backchannels=tuple(
            AnnotationSpan(start_seconds=0.0, end_seconds=1.0, text="yes")
            for _ in range(backchannel_count)
        ),
        turns=tuple(
            AnnotationPoint(time_seconds=1.0, confidence=None, text=None) for _ in range(turn_count)
        ),
        interruptions=tuple(
            AnnotationPoint(time_seconds=1.0, confidence=0.9, text="interrupt")
            for _ in range(interruption_count)
        ),
        segment_targets=(),
        connection_targets=(),
        speech_duration_seconds=speech_duration_seconds,
        pause_duration_seconds=float(pause_count),
        backchannel_duration_seconds=float(backchannel_count),
    )


def transcript_turn(speaker: str, start_seconds: float, end_seconds: float) -> TranscriptTurn:
    return TranscriptTurn(
        speaker=speaker,
        text="turn",
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        words=[],
    )
