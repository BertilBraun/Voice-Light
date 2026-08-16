from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionAnalysis,
    ConversationRegionConfig,
)
from app.local.corpus_audit.repository import CorpusAuditEvidence
from app.local.corpus_review.models import (
    CorpusReviewDatasetSelection,
    CorpusReviewDecision,
    CorpusReviewGateIssueCode,
    CorpusReviewItemRecord,
    CorpusReviewPlan,
    CorpusReviewSetRecord,
    CorpusReviewSetRequest,
    CorpusReviewStatus,
)
from app.local.corpus_review.service import corpus_review_readiness, plan_corpus_review
from app.local.db.models import TrackSide
from app.local.ingestion.conversation import ANNOTATION_VERSION
from app.shared.quality import (
    METRIC_VERSION,
    ConversationAnnotation,
    SpeakerConversationAnnotation,
    SpeakerSide,
)

DATASET_1 = UUID("11111111-1111-1111-1111-111111111111")
DATASET_2 = UUID("22222222-2222-2222-2222-222222222222")


def test_review_plan_is_reproducible_and_samples_distinct_recordings() -> None:
    request = CorpusReviewSetRequest(
        name="prepublish-v1",
        seed="voice-light-corpus-v1",
        items_per_dataset=3,
        datasets=(
            CorpusReviewDatasetSelection(dataset_id=DATASET_1, minimum_quality=0.95),
            CorpusReviewDatasetSelection(dataset_id=DATASET_2, minimum_quality=0.0),
        ),
    )
    evidence = tuple(
        evidence_item(dataset_id, sample_number)
        for dataset_id in (DATASET_1, DATASET_2)
        for sample_number in range(1, 6)
    )

    first = plan_corpus_review(evidence=evidence, request=request)
    second = plan_corpus_review(evidence=tuple(reversed(evidence)), request=request)

    assert first == second
    assert len(first) == 6
    for dataset_id in (DATASET_1, DATASET_2):
        selected = tuple(item for item in first if item.dataset_id == dataset_id)
        assert len({item.sample_id for item in selected}) == 3
    assert all(round(item.start_seconds / 0.08) * 0.08 == item.start_seconds for item in first)


def test_review_plan_requires_enough_full_length_region_analyzed_recordings() -> None:
    request = CorpusReviewSetRequest(
        name="prepublish-v1",
        seed="seed",
        items_per_dataset=2,
        datasets=(CorpusReviewDatasetSelection(dataset_id=DATASET_1, minimum_quality=0.0),),
    )
    valid = evidence_item(DATASET_1, 1)
    too_short = evidence_item(DATASET_1, 2, duration_seconds=10.0)

    try:
        plan_corpus_review(evidence=(valid, too_short), request=request)
    except ValueError as error:
        assert "only 1 eligible recordings" in str(error)
    else:
        raise AssertionError("Expected insufficient eligible recordings to fail.")


def test_review_plan_is_stable_across_database_dataset_ids() -> None:
    first_request = CorpusReviewSetRequest(
        name="portable-review",
        seed="portable-seed",
        items_per_dataset=3,
        datasets=(CorpusReviewDatasetSelection(dataset_id=DATASET_1, minimum_quality=0.0),),
    )
    second_request = first_request.model_copy(
        update={
            "datasets": (CorpusReviewDatasetSelection(dataset_id=DATASET_2, minimum_quality=0.0),)
        }
    )
    first = plan_corpus_review(
        evidence=tuple(
            evidence_item(DATASET_1, sample_number, dataset_name="portable-dataset")
            for sample_number in range(1, 7)
        ),
        request=first_request,
    )
    second = plan_corpus_review(
        evidence=tuple(
            evidence_item(DATASET_2, sample_number, dataset_name="portable-dataset")
            for sample_number in range(1, 7)
        ),
        request=second_request,
    )

    assert tuple((item.external_id, item.user_side, item.start_seconds) for item in first) == tuple(
        (item.external_id, item.user_side, item.start_seconds) for item in second
    )


@pytest.mark.parametrize(
    ("audio", "annotation", "labels", "overall"),
    (
        ("pass", "pending", "pass", "pass"),
        ("pass", "pass", "pass", "fail"),
    ),
)
def test_review_decision_rejects_inconsistent_overall_status(
    audio: str,
    annotation: str,
    labels: str,
    overall: str,
) -> None:
    with pytest.raises(ValueError):
        CorpusReviewDecision(
            audio_status=CorpusReviewStatus(audio),
            annotation_status=CorpusReviewStatus(annotation),
            label_status=CorpusReviewStatus(labels),
            overall_status=CorpusReviewStatus(overall),
            notes="",
        )


def test_review_readiness_requires_current_provenance_and_all_passes() -> None:
    pending = corpus_review_readiness(review_plan(CorpusReviewStatus.PENDING))

    assert not pending.ready_to_publish
    assert pending.pending_item_count == 2
    assert tuple(issue.code for issue in pending.issues) == (
        CorpusReviewGateIssueCode.INCOMPLETE_REVIEW,
    )

    passed = corpus_review_readiness(review_plan(CorpusReviewStatus.PASS))

    assert passed.ready_to_publish
    assert passed.passed_item_count == 2
    assert passed.issues == ()


def review_plan(overall_status: CorpusReviewStatus) -> CorpusReviewPlan:
    request = CorpusReviewSetRequest(
        name="prepublish-v1",
        seed="seed",
        items_per_dataset=1,
        datasets=(
            CorpusReviewDatasetSelection(dataset_id=DATASET_1, minimum_quality=0.95),
            CorpusReviewDatasetSelection(dataset_id=DATASET_2, minimum_quality=0.0),
        ),
    )
    timestamp = datetime.now(UTC)
    review_set_id = uuid4()
    return CorpusReviewPlan(
        review_set=CorpusReviewSetRecord(
            id=review_set_id,
            name=request.name,
            seed=request.seed,
            items_per_dataset=request.items_per_dataset,
            config=request,
            created_at=timestamp,
            updated_at=timestamp,
        ),
        items=tuple(
            review_item(
                review_set_id=review_set_id,
                dataset_id=dataset_id,
                overall_status=overall_status,
                timestamp=timestamp,
            )
            for dataset_id in (DATASET_1, DATASET_2)
        ),
    )


def review_item(
    review_set_id: UUID,
    dataset_id: UUID,
    overall_status: CorpusReviewStatus,
    timestamp: datetime,
) -> CorpusReviewItemRecord:
    component_status = (
        CorpusReviewStatus.PASS
        if overall_status is CorpusReviewStatus.PASS
        else CorpusReviewStatus.PENDING
    )
    return CorpusReviewItemRecord(
        id=uuid4(),
        review_set_id=review_set_id,
        dataset_id=dataset_id,
        dataset_name=f"dataset-{dataset_id.hex[0]}",
        sample_id=uuid4(),
        external_id="sample_001",
        quality_score=0.97,
        quality_result_id=uuid4(),
        conversation_region_result_id=uuid4(),
        speaker1_audio_sha256="1" * 64,
        speaker2_audio_sha256="2" * 64,
        quality_metric_version=METRIC_VERSION,
        annotation_version=ANNOTATION_VERSION,
        region_analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        training_label_version="turn-taking-frame-labels-v1",
        input_duration_seconds=20.0,
        frame_seconds=0.08,
        provenance_current=True,
        user_side=TrackSide.SPEAKER1,
        start_seconds=0.0,
        audio_status=component_status,
        annotation_status=component_status,
        label_status=component_status,
        overall_status=overall_status,
        notes="",
        created_at=timestamp,
        updated_at=timestamp,
    )


def evidence_item(
    dataset_id: UUID,
    sample_number: int,
    duration_seconds: float = 120.0,
    dataset_name: str | None = None,
) -> CorpusAuditEvidence:
    return CorpusAuditEvidence(
        dataset_id=dataset_id,
        dataset_name=dataset_name or f"dataset-{dataset_id.hex[0]}",
        sample_id=UUID(int=dataset_id.int + sample_number),
        external_id=f"sample_{sample_number:03d}",
        represented_duration_seconds=duration_seconds,
        quality_score=0.97,
        annotation=conversation_annotation(duration_seconds),
        conversation_regions=ConversationRegionAnalysis(
            analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
            annotation_version="test",
            config=ConversationRegionConfig(),
            duration_seconds=duration_seconds,
            usable_duration_seconds=duration_seconds,
            unusable_duration_seconds=0.0,
            usable_ratio=1.0,
            unusable_regions=(),
        ),
    )


def conversation_annotation(duration_seconds: float) -> ConversationAnnotation:
    speaker1 = empty_speaker(SpeakerSide.SPEAKER1)
    speaker2 = empty_speaker(SpeakerSide.SPEAKER2)
    return ConversationAnnotation(
        annotation_version="test",
        analyzed_duration_seconds=duration_seconds,
        speaker1=speaker1,
        speaker2=speaker2,
        speech_segment_count=0,
        turn_count=0,
        turn_taking_count=0,
        interaction_count=0,
        pause_count=0,
        backchannel_count=0,
        interruption_count=0,
        usable_event_count=0,
        events_per_hour=0.0,
        speaker_balance_score=0.0,
        quality_score=0.0,
    )


def empty_speaker(side: SpeakerSide) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=(),
        pauses=(),
        backchannels=(),
        turns=(),
        interruptions=(),
        segment_targets=(),
        connection_targets=(),
        speech_duration_seconds=0.0,
        pause_duration_seconds=0.0,
        backchannel_duration_seconds=0.0,
    )
