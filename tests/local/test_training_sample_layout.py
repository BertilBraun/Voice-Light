from __future__ import annotations

import pytest

from app.local.training_samples.models import (
    CandidateSource,
    SupervisionMaskReason,
    TrainingFramePreview,
)
from app.local.training_samples.service import build_frame_previews
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)


def test_burn_in_masks_every_training_head() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=_speaker_annotation(side=SpeakerSide.SPEAKER2),
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    burn_in_frame = frames[25]

    assert not burn_in_frame.supervised
    assert not burn_in_frame.user_yield_valid
    assert burn_in_frame.user_yield_mask_reason is SupervisionMaskReason.BURN_IN
    assert not burn_in_frame.user_floor_take_valid
    assert burn_in_frame.user_floor_take_mask_reason is SupervisionMaskReason.BURN_IN
    assert not burn_in_frame.interaction_event_valid
    assert burn_in_frame.interaction_event_mask_reason is SupervisionMaskReason.BURN_IN
    assert all(not target.valid for target in burn_in_frame.future_activity)
    assert all(
        target.mask_reason is SupervisionMaskReason.BURN_IN
        for target in burn_in_frame.future_activity
    )


def test_user_yield_is_zero_during_user_speech_and_one_after_confirmed_completion() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(AnnotationSpan(start_seconds=4.2, end_seconds=5.2, text="answer"),),
        segment_targets=(_segment_target(start_seconds=4.2, end_seconds=5.2),),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    speech_frame = _frame_at(frames, 4.44)
    post_completion_frame = _frame_at(frames, 5.24)

    assert speech_frame.user_yield_valid
    assert speech_frame.user_yield_target == pytest.approx(0.0)
    assert post_completion_frame.user_yield_valid
    assert post_completion_frame.user_yield_target == pytest.approx(1.0)
    assert post_completion_frame.interaction_event_valid
    assert post_completion_frame.interaction_event_distribution is not None
    assert post_completion_frame.interaction_event_distribution.turn_completion == pytest.approx(
        1.0
    )


def test_high_confidence_connection_is_continuation_and_ambiguous_connection_is_masked() -> None:
    continuation_user = _user_with_connection(merge_confidence=0.9)
    ambiguous_user = _user_with_connection(merge_confidence=0.5)
    assistant = _speaker_annotation(side=SpeakerSide.SPEAKER1)

    continuation_frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=continuation_user,
        assistant=assistant,
    )
    ambiguous_frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=ambiguous_user,
        assistant=assistant,
    )
    continuation_frame = _frame_at(continuation_frames, 5.24)
    ambiguous_frame = _frame_at(ambiguous_frames, 5.24)

    assert continuation_frame.candidate_source is CandidateSource.CONNECTION
    assert continuation_frame.user_yield_valid
    assert continuation_frame.user_yield_target == pytest.approx(0.0)
    assert continuation_frame.interaction_event_valid
    assert continuation_frame.interaction_event_distribution is not None
    assert continuation_frame.interaction_event_distribution.continuation_pause == pytest.approx(
        1.0
    )
    assert not ambiguous_frame.user_yield_valid
    assert ambiguous_frame.user_yield_mask_reason is SupervisionMaskReason.AMBIGUOUS_ANNOTATION
    assert not ambiguous_frame.interaction_event_valid
    assert (
        ambiguous_frame.interaction_event_mask_reason is SupervisionMaskReason.AMBIGUOUS_ANNOTATION
    )


@pytest.mark.parametrize(
    (
        "start_seconds",
        "end_seconds",
        "keep_playing_confidence",
        "turn_confidence",
        "interruption_confidence",
        "expected_target",
    ),
    (
        (5.0, 6.0, 0.0, 1.0, 1.0, 1.0),
        (6.5, 6.8, 1.0, 0.0, 0.0, 0.0),
    ),
)
def test_floor_take_and_non_floor_feedback_use_overlap_onset_window(
    start_seconds: float,
    end_seconds: float,
    keep_playing_confidence: float,
    turn_confidence: float,
    interruption_confidence: float,
    expected_target: float,
) -> None:
    user_segment = _segment_target(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        keep_playing_confidence=keep_playing_confidence,
        turn_confidence=turn_confidence,
        interruption_confidence=interruption_confidence,
    )
    assistant = _speaker_annotation(
        side=SpeakerSide.SPEAKER1,
        speech_segments=(
            AnnotationSpan(start_seconds=4.0, end_seconds=8.0, text="assistant speaking"),
        ),
        segment_targets=(_segment_target(start_seconds=4.0, end_seconds=8.0),),
    )
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(
            AnnotationSpan(
                start_seconds=user_segment.start_seconds,
                end_seconds=user_segment.end_seconds,
                text=user_segment.text,
            ),
        ),
        segment_targets=(user_segment,),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=10.0,
        annotation_end_seconds=10.0,
        user=user,
        assistant=assistant,
    )
    onset_frame = next(
        frame for frame in frames if frame.candidate_source is CandidateSource.OVERLAP_ONSET
    )
    late_frame = _frame_at(frames, user_segment.start_seconds + 0.6)

    assert onset_frame.user_floor_take_valid
    assert onset_frame.user_floor_take_target == pytest.approx(expected_target)
    assert onset_frame.interaction_event_valid
    assert onset_frame.interaction_event_distribution is not None
    assert onset_frame.interaction_event_distribution.floor_take == pytest.approx(expected_target)
    assert onset_frame.interaction_event_distribution.non_floor_feedback == pytest.approx(
        1.0 - expected_target
    )
    assert not late_frame.user_floor_take_valid
    assert late_frame.user_floor_take_mask_reason is SupervisionMaskReason.OUTSIDE_OVERLAP_ONSET


def test_interaction_event_is_valid_on_only_one_causal_anchor_frame() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=_user_with_connection(merge_confidence=0.9),
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    event_frames = [frame for frame in frames if frame.interaction_event_valid]

    assert len(event_frames) == 1
    assert event_frames[0].time_seconds >= 5.2
    assert all(
        frame.interaction_event_mask_reason is SupervisionMaskReason.NO_EVENT_ANCHOR
        for frame in frames
        if frame.supervised and not frame.candidate
    )


def test_event_anchors_outside_crop_are_not_clamped_into_edge_frames() -> None:
    frames = build_frame_previews(
        start_seconds=10.0,
        end_seconds=18.0,
        annotation_end_seconds=20.0,
        user=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            speech_segments=(AnnotationSpan(start_seconds=1.0, end_seconds=2.0, text="old"),),
            segment_targets=(_segment_target(start_seconds=1.0, end_seconds=2.0),),
        ),
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    assert not any(frame.interaction_event_valid for frame in frames)
    assert all(
        frame.interaction_event_mask_reason is SupervisionMaskReason.NO_EVENT_ANCHOR
        for frame in frames
        if frame.supervised
    )


def test_audio_activity_segment_only_trains_future_activity() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(AnnotationSpan(start_seconds=4.2, end_seconds=5.2, text=None),),
        segment_targets=(
            _segment_target(
                start_seconds=4.2,
                end_seconds=5.2,
                evidence_source=AnnotationEvidenceSource.AUDIO_ACTIVITY,
            ),
        ),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )
    speech_frame = _frame_at(frames, 4.44)
    boundary_frame = _frame_at(frames, 5.24)

    assert not speech_frame.user_yield_valid
    assert speech_frame.user_yield_mask_reason is SupervisionMaskReason.AMBIGUOUS_ANNOTATION
    assert speech_frame.future_activity[0].valid
    assert speech_frame.future_activity[0].occupancy == pytest.approx(1.0)
    assert not boundary_frame.interaction_event_valid
    assert (
        boundary_frame.interaction_event_mask_reason is SupervisionMaskReason.AMBIGUOUS_ANNOTATION
    )


def test_low_turn_confidence_masks_no_connection_completion() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(AnnotationSpan(start_seconds=4.2, end_seconds=5.2, text="maybe"),),
        segment_targets=(
            _segment_target(
                start_seconds=4.2,
                end_seconds=5.2,
                turn_confidence=0.5,
            ),
        ),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )
    post_speech_frame = _frame_at(frames, 5.24)

    assert not post_speech_frame.user_yield_valid
    assert post_speech_frame.user_yield_mask_reason is SupervisionMaskReason.AMBIGUOUS_ANNOTATION
    assert not post_speech_frame.interaction_event_valid


def test_future_activity_is_soft_union_occupancy_and_masks_censored_horizons() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(
            AnnotationSpan(start_seconds=4.3, end_seconds=4.4, text="partial"),
            AnnotationSpan(start_seconds=4.45, end_seconds=4.7, text="next"),
        ),
        backchannels=(AnnotationSpan(start_seconds=4.3, end_seconds=4.4, text="partial"),),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=6.0,
        annotation_end_seconds=5.25,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    frame = _frame_at(frames, 4.2)
    near_end_frame = _frame_at(frames, 5.16)

    assert frame.future_activity[0].occupancy == pytest.approx(0.5)
    assert frame.future_activity[1].occupancy == pytest.approx(5.0 / 6.0)
    assert frame.future_activity[2].occupancy == pytest.approx(0.0)
    assert frame.future_activity[3].occupancy is None
    assert tuple(target.valid for target in frame.future_activity) == (True, True, True, False)
    assert near_end_frame.future_activity[0].occupancy is None
    assert (
        near_end_frame.future_activity[0].mask_reason
        is SupervisionMaskReason.FUTURE_HORIZON_CENSORED
    )


def _frame_at(
    frames: tuple[TrainingFramePreview, ...],
    time_seconds: float,
) -> TrainingFramePreview:
    return min(frames, key=lambda frame: abs(frame.time_seconds - time_seconds))


def _user_with_connection(merge_confidence: float) -> SpeakerConversationAnnotation:
    return _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(
            AnnotationSpan(start_seconds=4.2, end_seconds=5.2, text="first"),
            AnnotationSpan(start_seconds=6.0, end_seconds=7.0, text="second"),
        ),
        segment_targets=(
            _segment_target(start_seconds=4.2, end_seconds=5.2),
            _segment_target(start_seconds=6.0, end_seconds=7.0),
        ),
        connection_targets=(
            ConnectionAnnotationTarget(
                earlier_end_seconds=5.2,
                later_start_seconds=6.0,
                gap_seconds=0.8,
                pause_confidence=1.0,
                merge_confidence=merge_confidence,
            ),
        ),
    )


def _segment_target(
    start_seconds: float,
    end_seconds: float,
    keep_playing_confidence: float = 0.0,
    turn_confidence: float = 1.0,
    interruption_confidence: float = 0.0,
    evidence_source: AnnotationEvidenceSource = AnnotationEvidenceSource.TRANSCRIPT,
) -> SegmentAnnotationTarget:
    return SegmentAnnotationTarget(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        text="speech",
        evidence_source=evidence_source,
        keep_playing_confidence=keep_playing_confidence,
        turn_confidence=turn_confidence,
        interruption_confidence=interruption_confidence,
    )


def _speaker_annotation(
    side: SpeakerSide,
    speech_segments: tuple[AnnotationSpan, ...] = (),
    pauses: tuple[AnnotationSpan, ...] = (),
    backchannels: tuple[AnnotationSpan, ...] = (),
    segment_targets: tuple[SegmentAnnotationTarget, ...] = (),
    connection_targets: tuple[ConnectionAnnotationTarget, ...] = (),
) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=speech_segments,
        pauses=pauses,
        backchannels=backchannels,
        turns=(),
        interruptions=(),
        segment_targets=segment_targets,
        connection_targets=connection_targets,
        speech_duration_seconds=sum(
            span.end_seconds - span.start_seconds for span in speech_segments
        ),
        pause_duration_seconds=sum(span.end_seconds - span.start_seconds for span in pauses),
        backchannel_duration_seconds=sum(
            span.end_seconds - span.start_seconds for span in backchannels
        ),
    )
