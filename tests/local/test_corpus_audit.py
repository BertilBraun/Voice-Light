from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionAnalysis,
    ConversationRegionConfig,
    ConversationRegionReason,
    UnusableConversationRegion,
)
from app.local.corpus_audit.models import (
    CorpusAuditRejectionReason,
    CorpusAuditRequest,
)
from app.local.corpus_audit.repository import CorpusAuditEvidence
from app.local.corpus_audit.service import (
    _floor_validity_index,
    _supervision_coverage,
    generate_corpus_audit,
)
from app.local.main import app
from app.local.training_samples.models import TrainingSamplePropositionKind
from app.local.training_samples.service import build_frame_previews
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationSpan,
    ConversationAnnotation,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)

DATASET_ID = UUID("11111111-1111-1111-1111-111111111111")
SAMPLE_ID = UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture(scope="module")
def corpus_audit_assets() -> Iterator[str]:
    with TestClient(app) as client:
        page_response = client.get("/training/corpus-audit")
        script_response = client.get("/pages/corpus-audit/app.js")
    assert page_response.status_code == 200
    assert script_response.status_code == 200
    yield page_response.text + script_response.text


@pytest.mark.parametrize(
    "expected_text",
    (
        "Training corpus audit",
        "Run corpus audit",
        "/api/corpus-audit",
        "dataset_ids",
        "minimum_quality",
        "effective_supervised_duration_seconds",
        "pilot_metrics",
        "conversation-table",
    ),
)
def test_corpus_audit_page_exposes_report(expected_text: str, corpus_audit_assets: str) -> None:
    assert expected_text in corpus_audit_assets


def test_dense_audit_uses_both_orientations_and_sixteen_second_stride() -> None:
    report = generate_corpus_audit(
        evidence=(_evidence(conversation_regions=_regions(duration_seconds=40.0)),),
        request=_request(),
    )

    assert report.conversation_count == 1
    assert report.accepted_conversation_count == 1
    assert report.candidate_window_count == 6
    assert report.accepted_window_count == 4
    assert report.supervised_duration_seconds == pytest.approx(64.0)
    assert report.effective_supervised_duration_seconds == pytest.approx(64.0)
    assert report.physical_events.turn_shift_count == 3
    assert (
        sum(
            category.accepted_window_count
            for category in report.categories
            if category.kind is not TrainingSamplePropositionKind.BACKGROUND
        )
        > 0
    )


def test_lightweight_coverage_matches_full_frame_preview() -> None:
    annotation = _annotation()
    frames = build_frame_previews(
        start_seconds=16.0,
        end_seconds=36.0,
        annotation_end_seconds=40.0,
        user=annotation.speaker1,
        assistant=annotation.speaker2,
    )
    supervised_frames = tuple(frame for frame in frames if frame.supervised)
    expected_primary = sum(frame.user_has_floor_valid for frame in supervised_frames) / len(
        supervised_frames
    )
    future_targets = tuple(
        target for frame in supervised_frames for target in frame.future_activity
    )
    expected_future = sum(target.valid for target in future_targets) / len(future_targets)

    primary, future = _supervision_coverage(
        start_seconds=16.0,
        end_seconds=36.0,
        annotation_end_seconds=40.0,
        floor_validity=_floor_validity_index(annotation.speaker1),
    )

    assert primary == pytest.approx(expected_primary)
    assert future == pytest.approx(expected_future)


def test_missing_region_analysis_rejects_every_window() -> None:
    report = generate_corpus_audit(
        evidence=(_evidence(conversation_regions=None),),
        request=_request(),
    )

    assert report.candidate_window_count == 6
    assert report.accepted_window_count == 0
    missing_regions = next(
        rejection
        for rejection in report.rejections
        if rejection.reason is CorpusAuditRejectionReason.MISSING_REGION_ANALYSIS
    )
    assert missing_regions.window_count == 6


def test_excessive_masking_rejects_affected_orientations() -> None:
    regions = ConversationRegionAnalysis(
        analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        annotation_version="test",
        config=ConversationRegionConfig(),
        duration_seconds=40.0,
        usable_duration_seconds=30.0,
        unusable_duration_seconds=10.0,
        usable_ratio=0.75,
        unusable_regions=(
            UnusableConversationRegion(
                start_seconds=4.0,
                end_seconds=14.0,
                reasons=(ConversationRegionReason.ONE_SIDED_ACTIVITY,),
            ),
        ),
    )
    report = generate_corpus_audit(
        evidence=(_evidence(conversation_regions=regions),),
        request=_request(),
    )

    excessive_masking = next(
        rejection
        for rejection in report.rejections
        if rejection.reason is CorpusAuditRejectionReason.EXCESSIVE_MASKING
    )
    assert excessive_masking.window_count == 2
    assert report.accepted_window_count == 2


def _request() -> CorpusAuditRequest:
    return CorpusAuditRequest(dataset_ids=(DATASET_ID,), minimum_quality=0.9)


def _evidence(
    conversation_regions: ConversationRegionAnalysis | None,
) -> CorpusAuditEvidence:
    return CorpusAuditEvidence(
        dataset_id=DATASET_ID,
        dataset_name="test-dataset",
        sample_id=SAMPLE_ID,
        external_id="conversation-1",
        represented_duration_seconds=40.0,
        quality_score=0.97,
        annotation=_annotation(),
        conversation_regions=conversation_regions,
    )


def _regions(duration_seconds: float) -> ConversationRegionAnalysis:
    return ConversationRegionAnalysis(
        analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        annotation_version="test",
        config=ConversationRegionConfig(),
        duration_seconds=duration_seconds,
        usable_duration_seconds=duration_seconds,
        unusable_duration_seconds=0.0,
        usable_ratio=1.0,
        unusable_regions=(),
    )


def _annotation() -> ConversationAnnotation:
    speaker1 = _speaker(
        side=SpeakerSide.SPEAKER1,
        spans=((0.0, 10.0), (20.0, 30.0)),
    )
    speaker2 = _speaker(
        side=SpeakerSide.SPEAKER2,
        spans=((10.0, 20.0), (30.0, 40.0)),
    )
    return ConversationAnnotation(
        annotation_version="test",
        analyzed_duration_seconds=40.0,
        speaker1=speaker1,
        speaker2=speaker2,
        speech_segment_count=4,
        turn_count=4,
        turn_taking_count=3,
        interaction_count=3,
        pause_count=0,
        backchannel_count=0,
        interruption_count=0,
        usable_event_count=3,
        events_per_hour=270.0,
        speaker_balance_score=1.0,
        quality_score=1.0,
    )


def _speaker(
    side: SpeakerSide,
    spans: tuple[tuple[float, float], ...],
) -> SpeakerConversationAnnotation:
    speech_segments = tuple(
        AnnotationSpan(start_seconds=start, end_seconds=end, text="speech") for start, end in spans
    )
    segment_targets = tuple(
        SegmentAnnotationTarget(
            start_seconds=start,
            end_seconds=end,
            text="speech",
            evidence_source=AnnotationEvidenceSource.TRANSCRIPT,
            keep_playing_confidence=0.0,
            turn_confidence=1.0,
            interruption_confidence=0.0,
        )
        for start, end in spans
    )
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=speech_segments,
        pauses=(),
        backchannels=(),
        turns=(),
        interruptions=(),
        segment_targets=segment_targets,
        connection_targets=(),
        speech_duration_seconds=sum(end - start for start, end in spans),
        pause_duration_seconds=0.0,
        backchannel_duration_seconds=0.0,
    )
