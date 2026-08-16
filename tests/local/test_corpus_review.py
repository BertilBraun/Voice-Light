from __future__ import annotations

from uuid import UUID

from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionAnalysis,
    ConversationRegionConfig,
)
from app.local.corpus_audit.repository import CorpusAuditEvidence
from app.local.corpus_review.models import (
    CorpusReviewDatasetSelection,
    CorpusReviewSetRequest,
)
from app.local.corpus_review.service import plan_corpus_review
from app.shared.quality import (
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


def evidence_item(
    dataset_id: UUID,
    sample_number: int,
    duration_seconds: float = 120.0,
) -> CorpusAuditEvidence:
    return CorpusAuditEvidence(
        dataset_id=dataset_id,
        dataset_name=f"dataset-{dataset_id.hex[0]}",
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
