from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.local.backchannel_review import router as backchannel_review_router
from app.local.backchannel_review.models import BackchannelReviewCandidate
from app.local.backchannel_review.router import _candidate_page
from app.local.backchannel_review.service import find_ambiguous_backchannel_candidates
from app.local.db.models import DashboardSample, SampleListFilter
from app.local.main import app
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    ConversationAnnotation,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)

SAMPLE_ID = UUID("5b41194e-c7ba-4bd2-a0c4-a572e972004c")


class FakeBackchannelRepository:
    def __init__(self, samples: list[DashboardSample]) -> None:
        self.samples = samples
        self.requested_offsets: list[int] = []

    def list_dashboard_samples(
        self,
        sample_filter: SampleListFilter,
    ) -> list[DashboardSample]:
        self.requested_offsets.append(sample_filter.offset)
        return self.samples[sample_filter.offset : sample_filter.offset + sample_filter.limit]


@pytest.fixture(scope="module")
def backchannel_review_script() -> Iterator[str]:
    with TestClient(app) as client:
        page_response = client.get("/analyses/backchannel-review")
        script_response = client.get("/pages/backchannel-review/app.js")
    assert page_response.status_code == 200
    assert "Ambiguous backchannel review" in page_response.text
    assert script_response.status_code == 200
    yield script_response.text


@pytest.mark.parametrize(
    "visible_control",
    (
        "previous",
        "next",
        "Play both",
        "floor_holder_connection",
        "possible_backchannel",
        "timelineInspector",
        "inspectTimelineAt",
        "Same turn",
        "New turn",
        "choice-track",
    ),
)
def test_backchannel_review_exposes_required_context(
    visible_control: str,
    backchannel_review_script: str,
) -> None:
    assert visible_control in backchannel_review_script


def test_next_button_autoplays_new_candidate(backchannel_review_script: str) -> None:
    assert 'elements.next.addEventListener("click", () => void showNextCandidate(true));' in (
        backchannel_review_script
    )
    assert "await loadNextCandidatePage();" in backchannel_review_script
    assert "if (autoplay) {" in backchannel_review_script
    assert "void play();" in backchannel_review_script


def test_candidate_comes_from_opponent_transcript_inside_merged_connection() -> None:
    annotation = _conversation_annotation(
        duration_seconds=20.0,
        speaker1=_speaker_annotation(
            side=SpeakerSide.SPEAKER1,
            segments=(
                _segment(2.0, 3.0, "I was thinking"),
                _segment(4.0, 5.0, "that we should leave"),
                _segment(8.0, 9.0, "a separate thought"),
                _segment(10.0, 11.0, "continues later"),
            ),
            connections=(
                _connection(3.0, 4.0, merge_confidence=0.82),
                _connection(9.0, 10.0, merge_confidence=0.42),
            ),
        ),
        speaker2=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            segments=(
                _segment(3.2, 3.6, "yes"),
                _segment(9.2, 9.6, "right"),
                _segment(
                    3.3,
                    3.5,
                    "[untranscribed audio activity]",
                    evidence_source=AnnotationEvidenceSource.AUDIO_ACTIVITY,
                ),
                _segment(3.4, 6.4, "this is an obviously long response"),
            ),
        ),
    )

    candidates = find_ambiguous_backchannel_candidates(
        sample_id=SAMPLE_ID,
        external_id="sample-a",
        annotation=annotation,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.floor_holder_side is SpeakerSide.SPEAKER1
    assert candidate.possible_backchannel_side is SpeakerSide.SPEAKER2
    assert candidate.floor_holder_before.text == "I was thinking"
    assert candidate.possible_backchannel.text == "yes"
    assert candidate.floor_holder_after.text == "that we should leave"
    assert candidate.floor_holder_connection.merge_confidence == pytest.approx(0.82)
    assert candidate.window_start_seconds == pytest.approx(0.0)
    assert candidate.window_end_seconds == pytest.approx(10.0)


def test_candidate_window_stays_ten_seconds_at_annotation_end() -> None:
    annotation = _conversation_annotation(
        duration_seconds=20.0,
        speaker1=_speaker_annotation(
            side=SpeakerSide.SPEAKER1,
            segments=(
                _segment(16.0, 17.0, "before"),
                _segment(18.0, 19.0, "after"),
            ),
            connections=(_connection(17.0, 18.0, merge_confidence=0.7),),
        ),
        speaker2=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            segments=(_segment(17.2, 17.5, "mhm"),),
        ),
    )

    candidate = find_ambiguous_backchannel_candidates(
        sample_id=SAMPLE_ID,
        external_id="sample-b",
        annotation=annotation,
    )[0]

    assert candidate.window_start_seconds == pytest.approx(10.0)
    assert candidate.window_end_seconds == pytest.approx(20.0)


def test_candidate_page_stops_after_bounded_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    samples = [DashboardSample.model_construct() for _ in range(20)]
    fake_repository = FakeBackchannelRepository(samples=samples)

    def repeated_candidates(
        dashboard_sample: DashboardSample,
    ) -> tuple[BackchannelReviewCandidate, ...]:
        del dashboard_sample
        return (candidate,) * 10

    monkeypatch.setattr(
        backchannel_review_router,
        "_candidates_for_sample",
        repeated_candidates,
    )

    page = _candidate_page(
        sample_repository=fake_repository,
        offset=40,
        limit=20,
    )

    assert len(page.candidates) == 20
    assert page.offset == 40
    assert page.next_offset == 60
    assert fake_repository.requested_offsets == [0]


def _candidate() -> BackchannelReviewCandidate:
    annotation = _conversation_annotation(
        duration_seconds=20.0,
        speaker1=_speaker_annotation(
            side=SpeakerSide.SPEAKER1,
            segments=(
                _segment(2.0, 3.0, "before"),
                _segment(4.0, 5.0, "after"),
            ),
            connections=(_connection(3.0, 4.0, merge_confidence=0.82),),
        ),
        speaker2=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            segments=(_segment(3.2, 3.6, "yes"),),
        ),
    )
    return find_ambiguous_backchannel_candidates(
        sample_id=SAMPLE_ID,
        external_id="sample-a",
        annotation=annotation,
    )[0]


def _segment(
    start_seconds: float,
    end_seconds: float,
    text: str,
    evidence_source: AnnotationEvidenceSource = AnnotationEvidenceSource.TRANSCRIPT,
) -> SegmentAnnotationTarget:
    return SegmentAnnotationTarget(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        text=text,
        evidence_source=evidence_source,
        keep_playing_confidence=0.2,
        turn_confidence=0.8,
        interruption_confidence=0.0,
    )


def _connection(
    earlier_end_seconds: float,
    later_start_seconds: float,
    merge_confidence: float,
) -> ConnectionAnnotationTarget:
    return ConnectionAnnotationTarget(
        earlier_end_seconds=earlier_end_seconds,
        later_start_seconds=later_start_seconds,
        gap_seconds=later_start_seconds - earlier_end_seconds,
        pause_confidence=merge_confidence * 0.6,
        merge_confidence=merge_confidence,
    )


def _speaker_annotation(
    side: SpeakerSide,
    segments: tuple[SegmentAnnotationTarget, ...],
    connections: tuple[ConnectionAnnotationTarget, ...] = (),
) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=tuple(
            AnnotationSpan(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=segment.text,
            )
            for segment in segments
        ),
        pauses=(),
        backchannels=(),
        turns=(),
        interruptions=(),
        segment_targets=segments,
        connection_targets=connections,
        speech_duration_seconds=sum(
            segment.end_seconds - segment.start_seconds for segment in segments
        ),
        pause_duration_seconds=0.0,
        backchannel_duration_seconds=0.0,
    )


def _conversation_annotation(
    duration_seconds: float,
    speaker1: SpeakerConversationAnnotation,
    speaker2: SpeakerConversationAnnotation,
) -> ConversationAnnotation:
    return ConversationAnnotation(
        annotation_version="test",
        analyzed_duration_seconds=duration_seconds,
        speaker1=speaker1,
        speaker2=speaker2,
        speech_segment_count=len(speaker1.segment_targets) + len(speaker2.segment_targets),
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
