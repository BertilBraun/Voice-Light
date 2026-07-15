from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.local.main import app
from app.local.training_samples.models import CandidateSource, ReliabilitySource
from app.local.training_samples.service import (
    ProbabilitySpan,
    _interesting_location_score,
    build_frame_previews,
)
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)


@pytest.fixture(scope="module")
def training_sample_script() -> Iterator[str]:
    with TestClient(app) as client:
        page_response = client.get("/training/sample-lab")
        script_response = client.get("/pages/training-samples/app.js")
    assert page_response.status_code == 200
    assert "Training sample lab" in page_response.text
    assert script_response.status_code == 200
    yield script_response.text


@pytest.mark.parametrize(
    "label_field",
    (
        "candidate",
        "assistant_speaking_input",
        "candidate_source",
        "yield_probability",
        "hold_probability",
        "primary_reliability",
        "primary_valid",
        "event_distribution",
        "future_activity",
        "nextRandomButton",
        "loadNextRandomSample",
        "/api/training-samples/options?limit=40",
        "/api/training-samples/random-preview",
        "playBothInput",
        "assistantAudio",
        "synchronizeAudioTracks",
        "minimumQualityInput",
        "samplingModeSelect",
        "preview.quality.total_score",
        "Future user activity",
    ),
)
def test_training_sample_lab_displays_target(label_field: str, training_sample_script: str) -> None:
    assert label_field in training_sample_script


def test_connection_confidence_remains_a_soft_target() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(
            AnnotationSpan(start_seconds=0.0, end_seconds=2.0, text="first"),
            AnnotationSpan(start_seconds=2.96, end_seconds=4.0, text="second"),
        ),
        segment_targets=(
            _segment_target(start_seconds=0.0, end_seconds=2.0),
            _segment_target(start_seconds=2.96, end_seconds=4.0),
        ),
        connection_targets=(
            ConnectionAnnotationTarget(
                earlier_end_seconds=2.0,
                later_start_seconds=2.96,
                gap_seconds=0.96,
                pause_confidence=0.30793,
                merge_confidence=0.30793,
            ),
        ),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=5.0,
        annotation_end_seconds=5.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    candidate_frame = frames[25]

    assert candidate_frame.time_seconds == pytest.approx(2.04)
    assert candidate_frame.candidate
    assert candidate_frame.candidate_source is CandidateSource.CONNECTION
    assert candidate_frame.yield_probability == pytest.approx(0.69207)
    assert candidate_frame.hold_probability == pytest.approx(0.30793)
    assert candidate_frame.primary_reliability is None
    assert candidate_frame.primary_reliability_source is ReliabilitySource.UNMEASURED
    assert candidate_frame.event_distribution is not None
    assert sum(candidate_frame.event_distribution.model_dump().values()) == pytest.approx(1.0)


def test_primary_target_is_dense_for_user_speech_and_silence() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=3.0,
        annotation_end_seconds=3.0,
        user=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            speech_segments=(AnnotationSpan(start_seconds=0.4, end_seconds=1.4, text="hello"),),
            segment_targets=(_segment_target(start_seconds=0.4, end_seconds=1.4),),
        ),
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    speech_frame = frames[6]
    silence_frame = frames[25]

    assert speech_frame.hold_probability == pytest.approx(1.0)
    assert speech_frame.yield_probability == pytest.approx(0.0)
    assert speech_frame.primary_valid
    assert silence_frame.hold_probability == pytest.approx(0.0)
    assert silence_frame.yield_probability == pytest.approx(1.0)
    assert silence_frame.primary_valid


def test_assistant_speaking_is_an_input_and_respects_playback_pauses() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=3.0,
        annotation_end_seconds=3.0,
        user=_speaker_annotation(side=SpeakerSide.SPEAKER2),
        assistant=_speaker_annotation(
            side=SpeakerSide.SPEAKER1,
            speech_segments=(
                AnnotationSpan(start_seconds=0.5, end_seconds=2.5, text="assistant turn"),
            ),
            pauses=(AnnotationSpan(start_seconds=1.0, end_seconds=1.5, text=None),),
            backchannels=(AnnotationSpan(start_seconds=2.7, end_seconds=2.9, text="mhm"),),
        ),
    )

    assert frames[7].assistant_speaking_input
    assert not frames[15].assistant_speaking_input
    assert frames[34].assistant_speaking_input


def test_backchannel_probability_reduces_yield_and_populates_event_target() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(AnnotationSpan(start_seconds=0.0, end_seconds=1.0, text="yeah"),),
        segment_targets=(
            _segment_target(
                start_seconds=0.0,
                end_seconds=1.0,
                keep_playing_confidence=0.8,
                turn_confidence=0.2,
            ),
        ),
        connection_targets=(
            ConnectionAnnotationTarget(
                earlier_end_seconds=1.0,
                later_start_seconds=2.0,
                gap_seconds=1.0,
                pause_confidence=1.0,
                merge_confidence=1.0,
            ),
        ),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=2.0,
        annotation_end_seconds=2.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    speech_frame = frames[6]
    candidate_frame = frames[12]

    assert speech_frame.yield_probability == pytest.approx(0.8)
    assert speech_frame.hold_probability == pytest.approx(0.2)
    assert candidate_frame.yield_probability == pytest.approx(0.0)
    assert candidate_frame.event_distribution is not None
    assert candidate_frame.event_distribution.backchannel == pytest.approx(0.8)
    assert candidate_frame.event_distribution.continuation_pause == pytest.approx(0.2)


def test_censored_boundary_masks_only_event_target() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=2.0,
        annotation_end_seconds=2.0,
        user=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            speech_segments=(AnnotationSpan(start_seconds=0.0, end_seconds=1.0, text="last"),),
            segment_targets=(_segment_target(start_seconds=0.0, end_seconds=1.0),),
        ),
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    candidate_frame = frames[12]

    assert candidate_frame.candidate
    assert candidate_frame.candidate_source is CandidateSource.CENSORED
    assert candidate_frame.primary_valid
    assert candidate_frame.yield_probability == pytest.approx(1.0)
    assert candidate_frame.primary_reliability_source is ReliabilitySource.UNMEASURED
    assert not candidate_frame.event_valid


def test_future_activity_bins_are_hard_and_masked_at_annotation_end() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(
            AnnotationSpan(start_seconds=0.0, end_seconds=1.0, text="first"),
            AnnotationSpan(start_seconds=1.25, end_seconds=2.0, text="second"),
        ),
        segment_targets=(_segment_target(start_seconds=0.0, end_seconds=1.0),),
        connection_targets=(
            ConnectionAnnotationTarget(
                earlier_end_seconds=1.0,
                later_start_seconds=1.25,
                gap_seconds=0.25,
                pause_confidence=1.0,
                merge_confidence=1.0,
            ),
        ),
    )
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=2.0,
        annotation_end_seconds=2.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    future_targets = frames[12].future_activity

    assert tuple(target.active for target in future_targets) == (False, True, True, None)
    assert tuple(target.valid for target in future_targets) == (True, True, True, False)


def test_future_activity_is_populated_outside_candidate_frames() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=1.0,
        annotation_end_seconds=1.0,
        user=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            speech_segments=(AnnotationSpan(start_seconds=0.4, end_seconds=0.8, text="hello"),),
            segment_targets=(_segment_target(start_seconds=0.4, end_seconds=0.8),),
        ),
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
    )

    frame_before_speech = frames[4]

    assert not frame_before_speech.candidate
    assert frame_before_speech.future_activity[0].active


def test_interesting_location_score_rewards_dense_events_or_ambiguous_targets() -> None:
    quiet_score = _interesting_location_score(
        start_seconds=0.0,
        duration_seconds=20.0,
        event_times=(),
        probability_spans=(),
    )
    dense_score = _interesting_location_score(
        start_seconds=0.0,
        duration_seconds=20.0,
        event_times=(5.0, 6.0, 7.0),
        probability_spans=(),
    )
    ambiguous_score = _interesting_location_score(
        start_seconds=0.0,
        duration_seconds=20.0,
        event_times=(),
        probability_spans=(
            ProbabilitySpan(start_seconds=5.0, end_seconds=7.0, yield_probability=0.5),
        ),
    )

    assert dense_score > quiet_score
    assert ambiguous_score > quiet_score


def _segment_target(
    start_seconds: float,
    end_seconds: float,
    keep_playing_confidence: float = 0.0,
    turn_confidence: float = 1.0,
    interruption_confidence: float = 0.0,
) -> SegmentAnnotationTarget:
    return SegmentAnnotationTarget(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        text="speech",
        evidence_source=AnnotationEvidenceSource.TRANSCRIPT,
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
