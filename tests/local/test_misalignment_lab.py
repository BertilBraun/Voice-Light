from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.local.db.models import (
    DashboardSample,
    QualityResultRecord,
    SampleRecord,
    SampleTrackRecord,
    TrackSide,
)
from app.local.main import app
from app.local.misalignment_lab.models import (
    MisalignmentJudgment,
    MisalignmentStoredJudgment,
)
from app.local.misalignment_lab.service import (
    _candidate_starts,
    build_misalignment_queue,
    estimate_piecewise_repair,
    interaction_window_metrics,
)
from app.local.synchronization_review.models import (
    SynchronizationAuditKind,
    SynchronizationAuditResult,
    SynchronizationAuditWindow,
    SynchronizationEvidenceSource,
)
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationPoint,
    AnnotationSpan,
    ConversationAnnotation,
    ProcessingStatus,
    QualityResult,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)


@pytest.fixture(scope="module")
def misalignment_lab_assets() -> Iterator[tuple[str, str]]:
    with TestClient(app) as client:
        page_response = client.get("/training/misalignment-lab")
        script_response = client.get("/pages/misalignment-lab/app.js")
    assert page_response.status_code == 200
    assert script_response.status_code == 200
    yield page_response.text, script_response.text


@pytest.mark.parametrize(
    "required_text",
    (
        "Misalignment lab",
        "Raw tracks - shift 0.0 s",
        "Plausibly aligned",
        "Likely misaligned",
        "Skip / unsure",
        "quarantines the whole recording",
        "raw waveform",
        "end of turn",
        "Test quarantine repairs",
        "Original 0.0 s",
        "Predicted",
        "Prediction sounds plausible",
        "Prediction does not fix it",
    ),
)
def test_misalignment_lab_exposes_rapid_triage_controls(
    required_text: str,
    misalignment_lab_assets: tuple[str, str],
) -> None:
    page, _ = misalignment_lab_assets
    assert required_text in page


@pytest.mark.parametrize(
    "required_script",
    (
        "QUEUE_SIZE = 50",
        "/api/misalignment-lab/queue",
        "/api/misalignment-lab/preview/",
        "/api/misalignment-lab/judgments",
        "/api/synchronization-review/audio-window/",
        'submitJudgment("plausibly_aligned")',
        'submitJudgment("likely_misaligned")',
        'submitJudgment("unsure")',
        "new AudioContext()",
        "createBufferSource",
        "source.start(scheduledTime, state.playbackOffsetSeconds)",
        "Playing both raw tracks on one AudioContext clock",
        "loadCandidate(state.index + 1, true)",
        "/pages/shared/annotation-timeline.js",
        "/api/misalignment-lab/repair-queue",
        "/api/misalignment-lab/repair-judgments",
        "speaker2Predicted",
        "setComparisonShift",
    ),
)
def test_misalignment_lab_uses_zero_shift_shared_clock_and_auto_advance(
    required_script: str,
    misalignment_lab_assets: tuple[str, str],
) -> None:
    _, script = misalignment_lab_assets
    assert required_script in script


def test_interaction_score_rewards_alternating_late_exchange() -> None:
    quiet = interaction_window_metrics(
        annotation=_annotation(duration_seconds=600.0, dense_late_exchange=False),
        start_seconds=570.0,
        end_seconds=590.0,
    )
    dense = interaction_window_metrics(
        annotation=_annotation(duration_seconds=600.0, dense_late_exchange=True),
        start_seconds=570.0,
        end_seconds=590.0,
    )

    assert dense.alternating_speaker_boundaries >= 4
    assert dense.rapid_speaker_boundaries >= 4
    assert dense.backchannel_count == 1
    assert dense.interaction_score > quiet.interaction_score


def test_queue_is_deterministic_unique_and_selects_late_interaction() -> None:
    samples = tuple(
        _dashboard_sample(external_id=f"pmt_{index:03}", dense_late_exchange=True)
        for index in range(1, 7)
    )

    first = build_misalignment_queue(
        dashboard_samples=samples,
        audit_report=None,
        judgments=(),
        seed="repeatable",
        limit=5,
    )
    second = build_misalignment_queue(
        dashboard_samples=samples,
        audit_report=None,
        judgments=(),
        seed="repeatable",
        limit=5,
    )

    assert first.candidates == second.candidates
    assert len({candidate.sample_id for candidate in first.candidates}) == 5
    assert all(candidate.window_start_seconds >= 570.0 for candidate in first.candidates)
    assert all(candidate.sampling_weight > 0.0 for candidate in first.candidates)


def test_short_recording_candidates_stay_in_final_forty_percent() -> None:
    starts = _candidate_starts(
        annotation=_annotation(duration_seconds=300.0, dense_late_exchange=False),
        duration_seconds=300.0,
    )

    assert min(starts) >= 180.0


def test_candidate_identity_changes_when_audio_is_replaced() -> None:
    original = _dashboard_sample(external_id="pmt_200", dense_late_exchange=True)
    replacement_tracks = (
        original.tracks[0],
        original.tracks[1].model_copy(update={"audio_sha256": "3" * 64}),
    )
    replacement = original.model_copy(update={"tracks": replacement_tracks})

    original_queue = build_misalignment_queue(
        dashboard_samples=(original,),
        audit_report=None,
        judgments=(),
        seed="audio-revision",
        limit=1,
    )
    replacement_queue = build_misalignment_queue(
        dashboard_samples=(replacement,),
        audit_report=None,
        judgments=(),
        seed="audio-revision",
        limit=1,
    )

    assert original_queue.candidates[0].candidate_id != replacement_queue.candidates[0].candidate_id


def test_likely_misaligned_judgment_quarantines_whole_session() -> None:
    samples = (
        _dashboard_sample(external_id="pmt_201", dense_late_exchange=True),
        _dashboard_sample(external_id="pmt_202", dense_late_exchange=True),
    )
    initial = build_misalignment_queue(
        dashboard_samples=samples,
        audit_report=None,
        judgments=(),
        seed="quarantine",
        limit=2,
    )
    rejected = initial.candidates[0]
    judgment = _stored_judgment(
        candidate_id=rejected.candidate_id,
        sample_id=rejected.sample_id,
        external_id=rejected.external_id,
        judgment=MisalignmentJudgment.LIKELY_MISALIGNED,
    )

    remaining = build_misalignment_queue(
        dashboard_samples=samples,
        audit_report=None,
        judgments=(judgment,),
        seed="quarantine",
        limit=2,
    )

    assert all(candidate.sample_id != rejected.sample_id for candidate in remaining.candidates)
    assert remaining.progress.quarantined_session_count == 1


def test_aligned_judgment_avoids_exact_snippet_without_quarantining_session() -> None:
    sample = _dashboard_sample(external_id="pmt_203", dense_late_exchange=True)
    initial = build_misalignment_queue(
        dashboard_samples=(sample,),
        audit_report=None,
        judgments=(),
        seed="aligned",
        limit=1,
    )
    reviewed = initial.candidates[0]
    judgment = _stored_judgment(
        candidate_id=reviewed.candidate_id,
        sample_id=reviewed.sample_id,
        external_id=reviewed.external_id,
        judgment=MisalignmentJudgment.PLAUSIBLY_ALIGNED,
    )

    later = build_misalignment_queue(
        dashboard_samples=(sample,),
        audit_report=None,
        judgments=(judgment,),
        seed="aligned",
        limit=1,
    )

    assert len(later.candidates) == 1
    assert later.candidates[0].sample_id == reviewed.sample_id
    assert later.candidates[0].candidate_id != reviewed.candidate_id
    assert later.progress.quarantined_session_count == 0


def test_piecewise_repair_estimate_requires_stable_distinct_second_part() -> None:
    sample_id = uuid4()
    estimate = estimate_piecewise_repair(
        audit_result=_audit_result(
            sample_id=sample_id,
            suffix_shifts=(4.0, 4.2, 4.3, 4.1, 4.2, 4.4),
        )
    )

    assert estimate is not None
    assert estimate.first_part_shift_seconds == 0.0
    assert estimate.predicted_second_part_shift_seconds == 4.2
    assert estimate.shift_change_seconds == 4.2
    assert estimate.supporting_window_count == 6
    assert estimate.stable_second_part_duration_seconds == 480.0
    assert estimate.conservative_first_part_end_seconds == 240.0
    assert estimate.conservative_second_part_start_seconds == 780.0


@pytest.mark.parametrize(
    "suffix_shifts",
    (
        (4.0, 4.1, 4.2),
        (0.3, 0.4, 0.2, 0.3, 0.4),
        (4.0, 5.5, 4.0, 5.5, 4.0),
    ),
)
def test_piecewise_repair_estimate_rejects_insufficient_or_unstable_suffix(
    suffix_shifts: tuple[float, ...],
) -> None:
    estimate = estimate_piecewise_repair(
        audit_result=_audit_result(
            sample_id=uuid4(),
            suffix_shifts=suffix_shifts,
        )
    )

    assert estimate is None


def _dashboard_sample(external_id: str, dense_late_exchange: bool) -> DashboardSample:
    sample_id = uuid4()
    annotation = _annotation(
        duration_seconds=600.0,
        dense_late_exchange=dense_late_exchange,
    )
    quality_result = QualityResult(
        metric_version="test",
        sample_id=str(sample_id),
        status=ProcessingStatus.COMPLETED,
        speaker1_uri="speaker1.wav",
        speaker2_uri="speaker2.wav",
        duration_seconds=600.0,
        interaction_density=None,
        timing_reliability=None,
        audio_quality=None,
        conversation_annotation=annotation,
        conversation_count_estimate=None,
        event_candidates=(),
        raw_quality_score=1.0,
        quality_flags=(),
        total_quality_score=1.0,
        error=None,
    )
    now = datetime.now(UTC)
    tracks = tuple(
        SampleTrackRecord(
            id=uuid4(),
            sample_id=sample_id,
            side=side,
            speaker_index=speaker_index,
            storage_uri=f"{side.value}.wav",
            access_uri=f"{side.value}.wav",
            duration_seconds=600.0,
            sample_rate=48_000,
            channels=1,
            sample_count=28_800_000,
            audio_sha256=str(speaker_index) * 64,
            created_at=now,
            updated_at=now,
        )
        for side, speaker_index in (
            (TrackSide.SPEAKER1, 1),
            (TrackSide.SPEAKER2, 2),
        )
    )
    return DashboardSample(
        sample=SampleRecord(
            id=sample_id,
            dataset_id=uuid4(),
            external_id=external_id,
            duration_seconds=600.0,
            quality_score=1.0,
            quality_flags=(),
            created_at=now,
            updated_at=now,
        ),
        tracks=tracks,
        latest_quality=QualityResultRecord.model_construct(
            payload=quality_result.model_dump(mode="json")
        ),
        latest_asr_run=None,
        latest_asr_evaluation=None,
        language_assessments=(),
    )


def _annotation(
    duration_seconds: float,
    dense_late_exchange: bool,
) -> ConversationAnnotation:
    if dense_late_exchange:
        speaker1_segments = (
            _segment(571.0, 573.0, "one"),
            _segment(575.0, 577.0, "three"),
            _segment(579.0, 581.0, "five"),
        )
        speaker2_segments = (
            _segment(573.1, 574.8, "two"),
            _segment(577.1, 578.8, "four"),
            _segment(581.1, 583.0, "six"),
        )
        speaker2_backchannels = (
            AnnotationSpan(start_seconds=584.0, end_seconds=584.5, text="mhm"),
        )
    else:
        speaker1_segments = (_segment(571.0, 581.0, "long monologue"),)
        speaker2_segments = ()
        speaker2_backchannels = ()
    speaker1 = _speaker(
        side=SpeakerSide.SPEAKER1,
        segments=speaker1_segments,
        backchannels=(),
    )
    speaker2 = _speaker(
        side=SpeakerSide.SPEAKER2,
        segments=speaker2_segments,
        backchannels=speaker2_backchannels,
    )
    return ConversationAnnotation(
        annotation_version="full-duration-test-v1",
        analyzed_duration_seconds=duration_seconds,
        speaker1=speaker1,
        speaker2=speaker2,
        speech_segment_count=len(speaker1_segments) + len(speaker2_segments),
        turn_count=len(speaker1.turns) + len(speaker2.turns),
        turn_taking_count=5 if dense_late_exchange else 0,
        interaction_count=5 if dense_late_exchange else 0,
        pause_count=0,
        backchannel_count=len(speaker2_backchannels),
        interruption_count=0,
        usable_event_count=6 if dense_late_exchange else 1,
        events_per_hour=36.0,
        speaker_balance_score=1.0,
        quality_score=1.0,
    )


def _speaker(
    side: SpeakerSide,
    segments: tuple[SegmentAnnotationTarget, ...],
    backchannels: tuple[AnnotationSpan, ...],
) -> SpeakerConversationAnnotation:
    speech_segments = tuple(
        AnnotationSpan(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            text=segment.text,
        )
        for segment in segments
    )
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=speech_segments,
        pauses=(),
        backchannels=backchannels,
        turns=tuple(
            AnnotationPoint(
                time_seconds=segment.end_seconds,
                confidence=0.9,
                text=segment.text,
            )
            for segment in segments
        ),
        interruptions=(),
        segment_targets=segments,
        connection_targets=(),
        speech_duration_seconds=sum(
            segment.end_seconds - segment.start_seconds for segment in speech_segments
        ),
        pause_duration_seconds=0.0,
        backchannel_duration_seconds=sum(
            segment.end_seconds - segment.start_seconds for segment in backchannels
        ),
    )


def _segment(
    start_seconds: float,
    end_seconds: float,
    text: str,
) -> SegmentAnnotationTarget:
    return SegmentAnnotationTarget(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        text=text,
        evidence_source=AnnotationEvidenceSource.TRANSCRIPT,
        keep_playing_confidence=0.1,
        turn_confidence=0.9,
        interruption_confidence=0.0,
    )


def _stored_judgment(
    candidate_id: UUID,
    sample_id: UUID,
    external_id: str,
    judgment: MisalignmentJudgment,
) -> MisalignmentStoredJudgment:
    now = datetime.now(UTC)
    return MisalignmentStoredJudgment(
        candidate_id=candidate_id,
        sample_id=sample_id,
        external_id=external_id,
        window_start_seconds=570.0,
        window_end_seconds=590.0,
        judgment=judgment,
        queue_seed="test",
        created_at=now,
        updated_at=now,
    )


def _audit_result(
    sample_id: UUID,
    suffix_shifts: tuple[float, ...],
) -> SynchronizationAuditResult:
    prefix = tuple(
        _audit_window(
            start_seconds=float(index * 60),
            shift_seconds=0.0,
            accepted=False,
            confidence_score=0.0,
        )
        for index in range(5)
    )
    suffix = tuple(
        _audit_window(
            start_seconds=600.0 + float(index * 60),
            shift_seconds=shift_seconds,
            accepted=True,
            confidence_score=0.9,
        )
        for index, shift_seconds in enumerate(suffix_shifts)
    )
    return SynchronizationAuditResult(
        sample_id=sample_id,
        external_id="pmt_test",
        kind=SynchronizationAuditKind.TEMPORAL_CHANGE,
        anomaly_score=0.9,
        strongest_window_start_seconds=600.0,
        strongest_window_end_seconds=780.0,
        strongest_shift_seconds=suffix_shifts[0],
        temporal_shift_range_seconds=abs(suffix_shifts[0]),
        summary="test",
        windows=(*prefix, *suffix),
    )


def _audit_window(
    start_seconds: float,
    shift_seconds: float,
    accepted: bool,
    confidence_score: float,
) -> SynchronizationAuditWindow:
    return SynchronizationAuditWindow(
        start_seconds=start_seconds,
        end_seconds=start_seconds + 180.0,
        estimated_b_shift_seconds=shift_seconds,
        confidence_score=confidence_score,
        bad_state_improvement=0.02 if accepted else 0.0,
        competing_margin=0.01,
        basin_width_seconds=0.12,
        persistence_window_count=4 if accepted else 0,
        agreeing_transcript_sources=(
            SynchronizationEvidenceSource.PARAKEET,
            SynchronizationEvidenceSource.CANARY,
        )
        if accepted
        else (),
        accepted=accepted,
        maximum_lag_boundary=False,
    )
