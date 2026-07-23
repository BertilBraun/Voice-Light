from __future__ import annotations

from collections.abc import Iterator
from random import Random

import pytest
from fastapi.testclient import TestClient

from app.local.conversation_regions.models import (
    CONVERSATION_REGION_ANALYSIS_VERSION,
    ConversationRegionAnalysis,
    ConversationRegionConfig,
    ConversationRegionReason,
    UnusableConversationRegion,
)
from app.local.db.repository import minimum_quality_filter, optional_dataset_filter
from app.local.main import app
from app.local.timeline_repair.transform import CanonicalInterval
from app.local.training_samples.models import TrainingSamplePropositionKind
from app.local.training_samples.service import (
    MAXIMUM_PROPOSITION_MASKED_RATIO,
    ProbabilitySpan,
    PropositionAnchor,
    _build_proposition,
    _interesting_location_score,
    _masked_supervised_ratio,
    _proposition_anchors,
    _random_usable_start_seconds,
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
        "assistant_has_floor_input",
        "assistant_backchannel_target",
        "assistant_backchannel_valid",
        "assistant_backchannel_mask_reason",
        "candidate_source",
        "user_yield_target",
        "user_yield_valid",
        "user_yield_mask_reason",
        "user_has_floor_target",
        "user_has_floor_valid",
        "user_has_floor_mask_reason",
        "interaction_auxiliary",
        "continuation_pause",
        "non_floor_feedback",
        "future_activity",
        "AUX future user audio",
        "occupancy",
        "nextRandomButton",
        "loadNextPreparedSample",
        "prepareNextReviewSample",
        "/api/training-samples/options?${parameters}",
        "/api/training-samples/random-preview",
        "playBothInput",
        "assistantAudio",
        "synchronizeAudioTracks",
        "minimumQualityInput",
        "samplingModeSelect",
        "randomizeInitialInput",
        "PREPARED_PREVIEW_TARGET = 2",
        "ALL_DATASETS_VALUE",
        "All eligible datasets",
        "loadRandomCorpusPreview",
        "randomPreviewParameters",
        "preparedPreviewQueue",
        "fillPreparedPreviewQueue",
        "readJsonResponse",
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


def test_source_annotation_and_frame_preview_precede_context() -> None:
    with TestClient(app) as client:
        page = client.get("/training/sample-lab").text

    annotation_index = page.index("Source end-of-turn annotation")
    context_index = page.index("Conversation context")
    frame_index = page.index("Frame preview")
    assert annotation_index < frame_index < context_index
    assert "Suggested training crops" not in page
    assert "Random first conversation" in page


def test_prefetched_crop_autoplays_after_selection(training_sample_script: str) -> None:
    assert "await applyPreview(payload, true);" in training_sample_script


def test_next_sample_fetches_a_new_conversation(training_sample_script: str) -> None:
    assert "const nextConversation = await fetchNextConversationPreview(sourcePreview);" in (
        training_sample_script
    )
    assert "fillCandidateQueue" not in training_sample_script


def test_minimum_quality_change_reloads_filtered_conversations(
    training_sample_script: str,
) -> None:
    listener_start = training_sample_script.index('minimumQualityInput.addEventListener("change"')
    listener = training_sample_script[listener_start : listener_start + 120]
    assert "loadSamples()" in listener


def test_training_sample_minimum_quality_is_strict() -> None:
    filter_sql, parameters = minimum_quality_filter(0.9)

    assert filter_sql == "AND latest_quality.total_quality_score > %s"
    assert parameters == (0.9,)


def test_training_sample_dataset_filter_can_span_the_corpus() -> None:
    assert optional_dataset_filter(None) == ("", ())


def test_space_toggles_playback_independent_of_focus(training_sample_script: str) -> None:
    assert 'event.code !== "Space"' in training_sample_script
    assert "event.preventDefault();" in training_sample_script
    assert (
        'document.addEventListener("keydown", togglePlaybackFromSpace);' in training_sample_script
    )


def test_assistant_floor_is_a_soft_input_and_excludes_backchannels() -> None:
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
            segment_targets=(
                _segment_target(start_seconds=0.5, end_seconds=2.5),
                _segment_target(start_seconds=2.7, end_seconds=2.9),
            ),
        ),
    )

    assert frames[7].assistant_has_floor_input == pytest.approx(1.0)
    assert frames[15].assistant_has_floor_input == pytest.approx(1.0)
    assert frames[34].assistant_has_floor_input == pytest.approx(0.0)


def test_assistant_backchannel_inside_pause_retains_pause_floor_state() -> None:
    earlier_segment = _segment_target(start_seconds=0.0, end_seconds=1.0)
    later_segment = _segment_target(start_seconds=2.0, end_seconds=3.0)
    backchannel_segment = SegmentAnnotationTarget(
        start_seconds=1.3,
        end_seconds=1.5,
        text="mhm",
        evidence_source=AnnotationEvidenceSource.TRANSCRIPT,
        keep_playing_confidence=0.95,
        turn_confidence=0.0,
        interruption_confidence=0.0,
    )
    assistant = _speaker_annotation(
        side=SpeakerSide.SPEAKER1,
        speech_segments=(
            AnnotationSpan(start_seconds=0.0, end_seconds=1.0, text="first"),
            AnnotationSpan(start_seconds=1.3, end_seconds=1.5, text="mhm"),
            AnnotationSpan(start_seconds=2.0, end_seconds=3.0, text="second"),
        ),
        pauses=(AnnotationSpan(start_seconds=1.0, end_seconds=2.0, text=None),),
        backchannels=(AnnotationSpan(start_seconds=1.3, end_seconds=1.5, text="mhm"),),
        segment_targets=(earlier_segment, backchannel_segment, later_segment),
        connection_targets=(
            ConnectionAnnotationTarget(
                earlier_end_seconds=1.0,
                later_start_seconds=2.0,
                gap_seconds=1.0,
                pause_confidence=0.75,
                merge_confidence=0.9,
            ),
        ),
    )

    frames = build_frame_previews(
        start_seconds=0.0,
        end_seconds=3.0,
        annotation_end_seconds=3.0,
        user=_speaker_annotation(side=SpeakerSide.SPEAKER2),
        assistant=assistant,
    )
    backchannel_frame = min(frames, key=lambda frame: abs(frame.time_seconds - 1.4))

    assert backchannel_frame.assistant_has_floor_input == pytest.approx(0.5625)


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
    anchor = PropositionAnchor(
        kind=TrainingSamplePropositionKind.HOLD_PAUSE,
        time_seconds=12.0,
        confidence=0.9,
        description="Short masked hold",
    )
    proposition = _build_proposition(
        anchor=anchor,
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
        all_anchors=(anchor,),
    )

    assert proposition.masked_supervised_seconds == pytest.approx(1.0)
    assert proposition.masked_supervised_ratio == pytest.approx(1.0 / 16.0)
    assert proposition.primary_supervision_ratio == pytest.approx(1.0)
    assert proposition.score > 0.75


def test_random_location_avoids_excessively_masked_supervision() -> None:
    regions = ConversationRegionAnalysis(
        analysis_version=CONVERSATION_REGION_ANALYSIS_VERSION,
        annotation_version="test",
        config=ConversationRegionConfig(),
        duration_seconds=40.0,
        usable_duration_seconds=28.0,
        unusable_duration_seconds=12.0,
        usable_ratio=0.7,
        unusable_regions=(
            UnusableConversationRegion(
                start_seconds=0.0,
                end_seconds=12.0,
                reasons=(ConversationRegionReason.DUAL_SILENCE,),
            ),
        ),
    )

    start_seconds = _random_usable_start_seconds(
        duration_seconds=40.0,
        generator=Random(7),
        available_intervals=(CanonicalInterval(start_seconds=0.0, end_seconds=40.0),),
        conversation_regions=regions,
    )

    assert (
        _masked_supervised_ratio(
            start_seconds=start_seconds,
            end_seconds=start_seconds + 20.0,
            conversation_regions=regions,
        )
        <= MAXIMUM_PROPOSITION_MASKED_RATIO
    )


def test_candidate_anchors_require_tight_shifts_and_overlapping_backchannels() -> None:
    user = _speaker_annotation(
        side=SpeakerSide.SPEAKER2,
        speech_segments=(
            AnnotationSpan(start_seconds=2.0, end_seconds=4.0, text="user turn"),
            AnnotationSpan(start_seconds=11.2, end_seconds=11.5, text="yeah"),
            AnnotationSpan(start_seconds=15.0, end_seconds=15.3, text="alone"),
        ),
        backchannels=(
            AnnotationSpan(start_seconds=11.2, end_seconds=11.5, text="yeah"),
            AnnotationSpan(start_seconds=15.0, end_seconds=15.3, text="alone"),
        ),
        segment_targets=(_segment_target(start_seconds=2.0, end_seconds=4.0),),
    )
    assistant = _speaker_annotation(
        side=SpeakerSide.SPEAKER1,
        speech_segments=(
            AnnotationSpan(start_seconds=5.0, end_seconds=6.0, text="tight response"),
            AnnotationSpan(start_seconds=10.0, end_seconds=13.0, text="assistant turn"),
        ),
        segment_targets=(
            _segment_target(start_seconds=5.0, end_seconds=6.0),
            _segment_target(start_seconds=10.0, end_seconds=13.0),
        ),
    )

    anchors = _proposition_anchors(user=user, assistant=assistant, duration_seconds=20.0)

    shifts = tuple(
        anchor for anchor in anchors if anchor.kind is TrainingSamplePropositionKind.TURN_SHIFT
    )
    feedback = tuple(
        anchor
        for anchor in anchors
        if anchor.kind is TrainingSamplePropositionKind.NON_FLOOR_FEEDBACK
    )
    assert any(anchor.time_seconds == pytest.approx(4.0) for anchor in shifts)
    assert len(feedback) == 1
    assert feedback[0].time_seconds == pytest.approx(11.35)


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


def _segment_target(
    start_seconds: float,
    end_seconds: float,
) -> SegmentAnnotationTarget:
    return SegmentAnnotationTarget(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        text="speech",
        evidence_source=AnnotationEvidenceSource.TRANSCRIPT,
        keep_playing_confidence=0.0,
        turn_confidence=1.0,
        interruption_confidence=0.0,
    )
