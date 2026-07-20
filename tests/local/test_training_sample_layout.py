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
    AnnotationPoint,
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
    assert not burn_in_frame.user_has_floor_valid
    assert burn_in_frame.user_has_floor_mask_reason is SupervisionMaskReason.BURN_IN
    assert all(
        not target.valid and target.mask_reason is SupervisionMaskReason.BURN_IN
        for target in (
            burn_in_frame.interaction_auxiliary.turn_completion,
            burn_in_frame.interaction_auxiliary.continuation_pause,
            burn_in_frame.interaction_auxiliary.non_floor_feedback,
            burn_in_frame.interaction_auxiliary.floor_take,
        )
    )
    assert all(not target.valid for target in burn_in_frame.future_activity)
    assert all(
        target.mask_reason is SupervisionMaskReason.BURN_IN
        for target in burn_in_frame.future_activity
    )


def test_user_yield_predicts_floor_availability_in_500_milliseconds() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(AnnotationSpan(start_seconds=4.2, end_seconds=5.2, text="answer"),),
        segment_targets=(_segment_target(start_seconds=4.2, end_seconds=5.2),),
        turns=(_turn_point(5.2),),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    speech_frame = _frame_at(frames, 4.44)
    approaching_completion_frame = _frame_at(frames, 4.84)
    post_completion_frame = _frame_at(frames, 5.24)

    assert speech_frame.user_yield_valid
    assert speech_frame.user_yield_target == pytest.approx(0.0)
    assert speech_frame.user_has_floor_target == pytest.approx(1.0)
    assert approaching_completion_frame.user_yield_valid
    assert approaching_completion_frame.user_yield_target == pytest.approx(1.0)
    assert post_completion_frame.user_yield_valid
    assert post_completion_frame.user_yield_target == pytest.approx(1.0)
    assert post_completion_frame.user_has_floor_target == pytest.approx(0.0)
    assert post_completion_frame.interaction_auxiliary.turn_completion.valid
    assert post_completion_frame.interaction_auxiliary.turn_completion.target == pytest.approx(1.0)

    inactive_frame = _frame_at(frames, 7.24)
    assert not inactive_frame.user_yield_valid
    assert inactive_frame.user_yield_mask_reason is SupervisionMaskReason.OUTSIDE_USER_YIELD_CONTEXT


def test_user_yield_masks_when_its_future_floor_horizon_is_censored() -> None:
    segment = _segment_target(start_seconds=4.2, end_seconds=5.2)
    speech_segments = (
        AnnotationSpan(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            text=segment.text,
        ),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=5.6,
        annotation_end_seconds=5.6,
        user=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            speech_segments=speech_segments,
            segment_targets=(segment,),
        ),
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    available_frame = _frame_at(frames, 4.84)
    censored_frame = _frame_at(frames, 5.24)

    assert available_frame.user_yield_valid
    assert available_frame.user_yield_target == pytest.approx(1.0)
    assert not censored_frame.user_yield_valid
    assert censored_frame.user_yield_mask_reason is SupervisionMaskReason.FUTURE_HORIZON_CENSORED


def test_connection_probability_softly_supervises_floor_state_and_yield() -> None:
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
    assert continuation_frame.user_has_floor_valid
    assert continuation_frame.user_has_floor_target == pytest.approx(0.9)
    assert continuation_frame.user_yield_valid
    assert continuation_frame.user_yield_target == pytest.approx(0.1)
    assert continuation_frame.interaction_auxiliary.continuation_pause.valid
    assert continuation_frame.interaction_auxiliary.continuation_pause.target == pytest.approx(0.9)
    assert ambiguous_frame.user_yield_valid
    assert ambiguous_frame.user_yield_target == pytest.approx(0.5)
    assert ambiguous_frame.user_has_floor_valid
    assert ambiguous_frame.user_has_floor_target == pytest.approx(0.5)
    assert ambiguous_frame.interaction_auxiliary.continuation_pause.valid
    assert ambiguous_frame.interaction_auxiliary.continuation_pause.target == pytest.approx(0.5)


def test_connection_state_supervision_covers_untranscribed_activity_inside_pause() -> None:
    user = _user_with_connection(merge_confidence=0.75)
    user = user.model_copy(
        update={
            "segment_targets": (
                *user.segment_targets,
                _segment_target(
                    start_seconds=5.3,
                    end_seconds=5.7,
                    evidence_source=AnnotationEvidenceSource.AUDIO_ACTIVITY,
                ),
            )
        }
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    pause_frame = _frame_at(frames, 5.44)

    assert pause_frame.user_has_floor_valid
    assert pause_frame.user_has_floor_target == pytest.approx(0.75)
    assert pause_frame.user_yield_valid
    assert pause_frame.user_yield_target == pytest.approx(0.25)


def test_user_yield_masks_after_two_seconds_even_inside_a_long_connection() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        segment_targets=(
            _segment_target(start_seconds=4.2, end_seconds=5.2),
            _segment_target(start_seconds=10.0, end_seconds=11.0),
        ),
        connection_targets=(
            ConnectionAnnotationTarget(
                earlier_end_seconds=5.2,
                later_start_seconds=10.0,
                gap_seconds=4.8,
                pause_confidence=0.4,
                merge_confidence=0.3,
            ),
        ),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=12.0,
        annotation_end_seconds=12.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    release_window_frame = _frame_at(frames, 7.16)
    inactive_frame = _frame_at(frames, 7.24)

    assert release_window_frame.user_yield_valid
    assert not inactive_frame.user_yield_valid
    assert inactive_frame.user_yield_mask_reason is SupervisionMaskReason.OUTSIDE_USER_YIELD_CONTEXT


@pytest.mark.parametrize(
    (
        "start_seconds",
        "end_seconds",
        "keep_playing_confidence",
        "turn_confidence",
        "interruption_confidence",
        "expected_has_floor",
    ),
    (
        (5.0, 6.0, 0.0, 1.0, 1.0, 1.0),
        (6.5, 6.8, 1.0, 0.0, 0.0, 0.0),
    ),
)
def test_user_has_floor_is_dense_while_floor_take_remains_a_soft_onset_event(
    start_seconds: float,
    end_seconds: float,
    keep_playing_confidence: float,
    turn_confidence: float,
    interruption_confidence: float,
    expected_has_floor: float,
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

    assert onset_frame.user_has_floor_valid
    assert onset_frame.user_has_floor_target == pytest.approx(expected_has_floor)
    assert late_frame.user_has_floor_valid
    assert late_frame.user_has_floor_target == pytest.approx(expected_has_floor)
    assert onset_frame.interaction_auxiliary.floor_take.valid
    assert onset_frame.interaction_auxiliary.floor_take.target == pytest.approx(
        turn_confidence * interruption_confidence
    )
    assert not late_frame.interaction_auxiliary.floor_take.valid


def test_non_floor_feedback_probability_covers_the_annotated_backchannel_span() -> None:
    backchannel = AnnotationSpan(start_seconds=5.0, end_seconds=5.6, text="right")
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(backchannel,),
        backchannels=(backchannel,),
        segment_targets=(
            _segment_target(
                start_seconds=5.08,
                end_seconds=5.52,
                keep_playing_confidence=0.65,
                turn_confidence=0.2,
            ),
        ),
    )
    assistant = _speaker_annotation(
        side=SpeakerSide.SPEAKER1,
        speech_segments=(AnnotationSpan(start_seconds=4.0, end_seconds=5.2, text="talk"),),
        segment_targets=(_segment_target(start_seconds=4.0, end_seconds=5.2),),
    )

    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=user,
        assistant=assistant,
    )
    feedback_frames = [
        frame for frame in frames if frame.interaction_auxiliary.non_floor_feedback.valid
    ]

    assert len(feedback_frames) > 1
    assert all(5.0 <= frame.time_seconds < 5.6 for frame in feedback_frames)
    assert all(
        frame.interaction_auxiliary.non_floor_feedback.target == pytest.approx(0.65)
        for frame in feedback_frames
    )


def test_completion_is_a_point_target_while_pause_covers_the_connection_span() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=8.0,
        annotation_end_seconds=8.0,
        user=_user_with_connection(merge_confidence=0.9),
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    completion_frames = [
        frame for frame in frames if frame.interaction_auxiliary.turn_completion.valid
    ]
    pause_frames = [
        frame for frame in frames if frame.interaction_auxiliary.continuation_pause.valid
    ]

    assert len(completion_frames) == 1
    assert completion_frames[0].time_seconds >= 5.2
    assert completion_frames[0].interaction_auxiliary.turn_completion.target == pytest.approx(0.1)
    assert all(frame.time_seconds >= 5.2 for frame in pause_frames)
    assert all(frame.time_seconds < 6.0 for frame in pause_frames)
    assert len(pause_frames) > 1
    assert all(
        frame.interaction_auxiliary.continuation_pause.mask_reason
        is SupervisionMaskReason.NO_AUXILIARY_ANNOTATION
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

    assert not any(frame.interaction_auxiliary.turn_completion.valid for frame in frames)
    assert all(
        frame.interaction_auxiliary.turn_completion.mask_reason
        is SupervisionMaskReason.NO_AUXILIARY_ANNOTATION
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
    assert not speech_frame.user_has_floor_valid
    assert speech_frame.user_has_floor_mask_reason is SupervisionMaskReason.AMBIGUOUS_ANNOTATION
    assert speech_frame.future_activity[0].valid
    assert speech_frame.future_activity[0].occupancy == pytest.approx(1.0)
    assert not boundary_frame.interaction_auxiliary.turn_completion.valid
    assert (
        boundary_frame.interaction_auxiliary.turn_completion.mask_reason
        is SupervisionMaskReason.AMBIGUOUS_ANNOTATION
    )


def test_uncertain_transcript_softly_supervises_runtime_heads() -> None:
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

    assert post_speech_frame.user_yield_valid
    assert post_speech_frame.user_yield_target == pytest.approx(1.0)
    ambiguous_speech_frame = _frame_at(frames, 4.44)
    assert ambiguous_speech_frame.user_has_floor_valid
    assert ambiguous_speech_frame.user_has_floor_target == pytest.approx(0.5)
    assert post_speech_frame.interaction_auxiliary.turn_completion.valid
    assert post_speech_frame.interaction_auxiliary.turn_completion.target == pytest.approx(0.5)


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


def _turn_point(time_seconds: float) -> AnnotationPoint:
    return AnnotationPoint(
        time_seconds=time_seconds,
        confidence=None,
        text=None,
    )


def _speaker_annotation(
    side: SpeakerSide,
    speech_segments: tuple[AnnotationSpan, ...] = (),
    pauses: tuple[AnnotationSpan, ...] = (),
    backchannels: tuple[AnnotationSpan, ...] = (),
    turns: tuple[AnnotationPoint, ...] = (),
    segment_targets: tuple[SegmentAnnotationTarget, ...] = (),
    connection_targets: tuple[ConnectionAnnotationTarget, ...] = (),
) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=speech_segments,
        pauses=pauses,
        backchannels=backchannels,
        turns=turns,
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
