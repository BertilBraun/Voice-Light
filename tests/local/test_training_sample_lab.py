from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionAnalysis,
    ConversationRegionConfig,
    ConversationRegionReason,
    UnusableConversationRegion,
)
from app.local.main import app
from app.local.training_samples.models import TrainingSamplePropositionKind
from app.local.training_samples.service import (
    ProbabilitySpan,
    PropositionAnchor,
    _build_proposition,
    _interesting_location_score,
    build_frame_previews,
)
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationSpan,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)


@pytest.fixture(scope="module")
def training_sample_script() -> Iterator[str]:
    with TestClient(app) as client:
        page_response = client.get("/training/sample-lab")
        script_response = client.get("/pages/training-samples/app.js")
        context_script_response = client.get("/pages/training-samples/context-overview.js")
    assert page_response.status_code == 200
    assert "Training sample lab" in page_response.text
    assert script_response.status_code == 200
    assert context_script_response.status_code == 200
    yield script_response.text + context_script_response.text


@pytest.mark.parametrize(
    "label_field",
    (
        "candidate",
        "assistant_speaking_input",
        "candidate_source",
        "user_yield_target",
        "user_yield_valid",
        "user_yield_mask_reason",
        "user_has_floor_target",
        "user_has_floor_valid",
        "user_has_floor_mask_reason",
        "interaction_event_distribution",
        "interaction_event_valid",
        "interaction_event_mask_reason",
        "future_activity",
        "occupancy",
        "nextRandomButton",
        "loadNextRandomSample",
        "/api/training-samples/options?${parameters}",
        "/api/training-samples/random-preview",
        "/api/training-samples/propositions",
        "renderPropositions",
        "masked_supervised_seconds",
        "playBothInput",
        "assistantAudio",
        "synchronizeAudioTracks",
        "minimumQualityInput",
        "samplingModeSelect",
        "datasetSelect",
        "contextOverview",
        "createConversationContextOverview",
        "contextOverviewController",
        "const CONTEXT_DURATION_SECONDS = 180",
        'canvas.addEventListener("wheel", zoomAtEvent',
        "wheelZoomMultiplier(event)",
        "requestGeneration += 1",
        "wheel to zoom",
        "createMediaElementGainController",
        "commonWaveformDisplayScale",
        "preview.user_gain.default_gain",
        "preview.assistant_gain.default_gain",
        "User automatic gain",
        "formatGain",
        "drawUnusableRegionOverlay",
        "recording_${role}_spans",
        "conversation_regions",
        "preview.quality.total_score",
        "preview.annotation_version",
        "preview.annotation_generated_at",
        "preview.user_audio_sha256",
        "preview.assistant_audio_sha256",
        "preview.assistant_waveform",
        "segment_targets",
        "connection_targets",
        "drawSourceAnnotationTimeline",
        "/pages/shared/annotation-timeline.js",
        'cache: "no-store"',
        "Future user activity",
    ),
)
def test_training_sample_lab_displays_target(label_field: str, training_sample_script: str) -> None:
    assert label_field in training_sample_script


def test_training_sample_lab_updates_positions_only_on_committed_interactions(
    training_sample_script: str,
) -> None:
    assert 'timeline.addEventListener("click", selectFrameAtEvent)' in training_sample_script
    assert 'timeline.addEventListener("pointermove"' not in training_sample_script
    assert "contextOverviewController.schedule" not in training_sample_script


def test_suggested_crop_autoplays_after_selection(training_sample_script: str) -> None:
    assert "void loadPreview(false, true);" in training_sample_script


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


def test_proposition_allows_short_masked_region_with_explicit_coverage() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(AnnotationSpan(start_seconds=0.0, end_seconds=30.0, text="speech"),),
        segment_targets=(
            SegmentAnnotationTarget(
                start_seconds=0.0,
                end_seconds=30.0,
                text="speech",
                evidence_source=AnnotationEvidenceSource.TRANSCRIPT,
                keep_playing_confidence=0.9,
                turn_confidence=0.1,
                interruption_confidence=0.0,
            ),
        ),
    )
    proposition = _build_proposition(
        anchor=PropositionAnchor(
            kind=TrainingSamplePropositionKind.HOLD_PAUSE,
            time_seconds=12.0,
            confidence=0.9,
            description="Short masked hold",
        ),
        duration_seconds=30.0,
        user=user,
        assistant=_speaker_annotation(side=SpeakerSide.SPEAKER1),
        conversation_regions=ConversationRegionAnalysis(
            analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
            annotation_version="test",
            config=ConversationRegionConfig(),
            duration_seconds=30.0,
            usable_duration_seconds=29.0,
            unusable_duration_seconds=1.0,
            usable_ratio=29.0 / 30.0,
            unusable_regions=(
                UnusableConversationRegion(
                    start_seconds=10.0,
                    end_seconds=11.0,
                    reasons=(ConversationRegionReason.DUAL_SILENCE,),
                ),
            ),
        ),
    )

    assert proposition.masked_supervised_seconds == pytest.approx(1.0)
    assert proposition.masked_supervised_ratio == pytest.approx(1.0 / 16.0)
    assert proposition.primary_supervision_ratio == pytest.approx(1.0)
    assert proposition.score > 0.8


def _speaker_annotation(
    side: SpeakerSide,
    speech_segments: tuple[AnnotationSpan, ...] = (),
    pauses: tuple[AnnotationSpan, ...] = (),
    backchannels: tuple[AnnotationSpan, ...] = (),
    segment_targets: tuple[SegmentAnnotationTarget, ...] = (),
) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=speech_segments,
        pauses=pauses,
        backchannels=backchannels,
        turns=(),
        interruptions=(),
        segment_targets=segment_targets,
        connection_targets=(),
        speech_duration_seconds=sum(
            span.end_seconds - span.start_seconds for span in speech_segments
        ),
        pause_duration_seconds=sum(span.end_seconds - span.start_seconds for span in pauses),
        backchannel_duration_seconds=sum(
            span.end_seconds - span.start_seconds for span in backchannels
        ),
    )
