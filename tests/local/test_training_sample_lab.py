from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.local.main import app
from app.local.training_samples.models import AgentActionLabel
from app.local.training_samples.service import build_frame_previews
from app.shared.quality import (
    AnnotationPoint,
    AnnotationSpan,
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
        "agent_action",
        "assistant_playback_active",
        "user_turn_active",
        "user_speech_active",
        "user_pause",
        "user_end_of_turn",
        "user_end_within_0_5_seconds",
        "user_end_within_1_second",
        "user_end_within_2_seconds",
        "user_backchannel",
        "user_interruption",
        "assistant_backchannel",
    ),
)
def test_training_sample_lab_displays_frame_label(
    label_field: str, training_sample_script: str
) -> None:
    assert label_field in training_sample_script


def test_assistant_pause_keeps_speak_target_but_clears_playback_input() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=20.0,
        user=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            speech_segments=(AnnotationSpan(start_seconds=0.0, end_seconds=8.0, text=None),),
            pauses=(AnnotationSpan(start_seconds=4.5, end_seconds=5.0, text=None),),
            turns=(AnnotationPoint(time_seconds=8.0, confidence=None, text=None),),
        ),
        assistant=_speaker_annotation(
            side=SpeakerSide.SPEAKER1,
            speech_segments=(AnnotationSpan(start_seconds=8.5, end_seconds=15.0, text=None),),
            pauses=(AnnotationSpan(start_seconds=12.0, end_seconds=13.0, text=None),),
        ),
    )

    assistant_pause_frame = frames[150]

    assert assistant_pause_frame.time_seconds == pytest.approx(12.04)
    assert assistant_pause_frame.agent_action is AgentActionLabel.SPEAK
    assert assistant_pause_frame.assistant_turn_active
    assert not assistant_pause_frame.assistant_playback_active
    assert not assistant_pause_frame.assistant_speech_active


def test_frame_targets_include_burn_in_horizons_and_events() -> None:
    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=20.0,
        user=_speaker_annotation(
            side=SpeakerSide.SPEAKER2,
            speech_segments=(AnnotationSpan(start_seconds=0.0, end_seconds=8.0, text=None),),
            pauses=(AnnotationSpan(start_seconds=4.5, end_seconds=5.0, text=None),),
            backchannels=(AnnotationSpan(start_seconds=10.0, end_seconds=10.4, text="yeah"),),
            turns=(AnnotationPoint(time_seconds=8.0, confidence=None, text=None),),
            interruptions=(AnnotationPoint(time_seconds=12.04, confidence=0.9, text="wait"),),
        ),
        assistant=_speaker_annotation(
            side=SpeakerSide.SPEAKER1,
            backchannels=(AnnotationSpan(start_seconds=2.0, end_seconds=2.4, text="mhm"),),
        ),
    )

    assert not frames[0].supervised
    assert frames[50].supervised
    assert frames[87].user_end_within_1_second
    assert not frames[87].user_end_within_0_5_seconds
    assert frames[57].user_pause
    assert frames[125].user_backchannel
    assert frames[150].user_interruption
    assert frames[25].assistant_backchannel
    assert frames[25].agent_action is AgentActionLabel.SPEAK


def _speaker_annotation(
    side: SpeakerSide,
    speech_segments: tuple[AnnotationSpan, ...] = (),
    pauses: tuple[AnnotationSpan, ...] = (),
    backchannels: tuple[AnnotationSpan, ...] = (),
    turns: tuple[AnnotationPoint, ...] = (),
    interruptions: tuple[AnnotationPoint, ...] = (),
) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=speech_segments,
        pauses=pauses,
        backchannels=backchannels,
        turns=turns,
        interruptions=interruptions,
        segment_targets=(),
        connection_targets=(),
        speech_duration_seconds=sum(
            span.end_seconds - span.start_seconds for span in speech_segments
        ),
        pause_duration_seconds=sum(span.end_seconds - span.start_seconds for span in pauses),
        backchannel_duration_seconds=sum(
            span.end_seconds - span.start_seconds for span in backchannels
        ),
    )
