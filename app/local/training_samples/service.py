from __future__ import annotations

import math
import random
import subprocess
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.local.conversation_regions.models import ConversationRegionAnalysis
from app.local.db.models import DashboardSample, SampleTrackRecord, TrackSide
from app.local.ingestion.local_audio import materialize_sample_track
from app.local.training_samples.models import (
    CandidateSource,
    EventTargetDistribution,
    FutureActivityTarget,
    PreviewEventType,
    PreviewPoint,
    PreviewSpan,
    PreviewWaveformPoint,
    SupervisionMaskReason,
    TrainingFramePreview,
    TrainingSamplePreview,
    TrainingSampleProposition,
    TrainingSamplePropositionKind,
    TrainingSampleQuality,
    TrainingSampleSelectionMode,
)
from app.shared.audio.gain import TrackGainNormalization, track_gain_normalization
from app.shared.audio.wav import mono_samples
from app.shared.quality import (
    AnnotationEvidenceSource,
    AnnotationPoint,
    AnnotationSpan,
    ConnectionAnnotationTarget,
    ConversationAnnotation,
    QualityResult,
    SegmentAnnotationTarget,
    SpeakerConversationAnnotation,
    SpeakerSide,
)

INPUT_DURATION_SECONDS = 20.0
BURN_IN_SECONDS = 4.0
FRAME_SECONDS = 0.08
WAVEFORM_POINT_COUNT = 1000
MAXIMUM_CANDIDATE_SILENCE_SECONDS = 2.0
FUTURE_ACTIVITY_WINDOWS_MILLISECONDS = ((0, 200), (200, 500), (500, 1000), (1000, 1500))
INTERESTING_RANDOM_LOCATION_COUNT = 24
INTERESTING_ANCHOR_LIMIT = 64
PROPOSITION_ANCHOR_LIMIT_PER_KIND = 18
PROPOSITION_EVENT_OFFSET_SECONDS = 12.0
MAXIMUM_TURN_SHIFT_GAP_SECONDS = 1.5
MINIMUM_PROPOSITION_SCORE = 0.75
MINIMUM_PROPOSITION_PRIMARY_COVERAGE = 0.9
MINIMUM_PROPOSITION_FUTURE_COVERAGE = 0.95
MAXIMUM_PROPOSITION_MASKED_RATIO = 0.35
HIGH_CONFIDENCE = 0.8
LOW_CONFIDENCE = 0.2
USER_YIELD_HORIZON_SECONDS = 0.5
MAXIMUM_NON_FLOOR_FEEDBACK_SECONDS = 1.2
MINIMUM_ASSISTANT_CONTINUATION_SECONDS = 0.3
MINIMUM_COMPLETION_INACTIVITY_SECONDS = 1.5


@dataclass(frozen=True)
class DecisionBoundary:
    speech_offset_seconds: float
    candidate_end_seconds: float
    source: CandidateSource


@dataclass(frozen=True)
class ScalarSupervision:
    target: float | None
    valid: bool
    mask_reason: SupervisionMaskReason | None


@dataclass(frozen=True)
class InteractionEventAnchor:
    time_seconds: float
    source: CandidateSource
    distribution: EventTargetDistribution | None
    mask_reason: SupervisionMaskReason | None


@dataclass(frozen=True)
class ProbabilitySpan:
    start_seconds: float
    end_seconds: float
    yield_probability: float


@dataclass(frozen=True)
class PropositionAnchor:
    kind: TrainingSamplePropositionKind
    time_seconds: float
    confidence: float
    description: str


def build_training_sample_preview(
    dashboard_sample: DashboardSample,
    user_side: TrackSide,
    requested_start_seconds: float | None,
    selection_mode: TrainingSampleSelectionMode,
    generator: random.Random,
    conversation_regions: ConversationRegionAnalysis | None,
) -> TrainingSamplePreview:
    annotation = _conversation_annotation(dashboard_sample)
    quality = dashboard_sample.latest_quality
    assert quality is not None
    represented_duration_seconds = (
        dashboard_sample.sample.duration_seconds
        if dashboard_sample.sample.duration_seconds is not None
        else annotation.analyzed_duration_seconds
    )
    eligible_duration_seconds = min(
        represented_duration_seconds, annotation.analyzed_duration_seconds
    )
    assistant_side = _other_side(user_side)
    user_annotation = _speaker_annotation(annotation, user_side)
    assistant_annotation = _speaker_annotation(annotation, assistant_side)
    user_gain, assistant_gain = _training_sample_gains(
        dashboard_sample=dashboard_sample,
        user_side=user_side,
    )
    start_seconds = _select_start_seconds(
        duration_seconds=eligible_duration_seconds,
        requested_start_seconds=requested_start_seconds,
        selection_mode=selection_mode,
        user=user_annotation,
        assistant=assistant_annotation,
        generator=generator,
    )
    end_seconds = min(eligible_duration_seconds, start_seconds + INPUT_DURATION_SECONDS)
    user_track = _track(dashboard_sample, user_side)
    assistant_track = _track(dashboard_sample, assistant_side)
    user_track_path = _track_path(user_track)
    assistant_track_path = _track_path(assistant_track)
    sample_rate, waveform = waveform_window(
        path=user_track_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        point_count=WAVEFORM_POINT_COUNT,
    )
    assistant_sample_rate, assistant_waveform = waveform_window(
        path=assistant_track_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        point_count=WAVEFORM_POINT_COUNT,
    )
    frames = build_frame_previews(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        annotation_end_seconds=annotation.analyzed_duration_seconds,
        user=user_annotation,
        assistant=assistant_annotation,
    )
    return TrainingSamplePreview(
        dataset_id=dashboard_sample.sample.dataset_id,
        sample_id=dashboard_sample.sample.id,
        external_id=dashboard_sample.sample.external_id,
        user_side=user_side,
        assistant_side=assistant_side,
        user_audio_sha256=required_track_sha256(user_track),
        assistant_audio_sha256=required_track_sha256(assistant_track),
        user_gain=user_gain,
        assistant_gain=assistant_gain,
        annotation_version=annotation.annotation_version,
        annotation_generated_at=quality.created_at,
        quality_metric_version=quality.metric_version,
        quality=_training_sample_quality(dashboard_sample),
        represented_duration_seconds=represented_duration_seconds,
        annotated_duration_seconds=annotation.analyzed_duration_seconds,
        eligible_duration_seconds=eligible_duration_seconds,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        burn_in_end_seconds=min(end_seconds, start_seconds + BURN_IN_SECONDS),
        input_duration_seconds=end_seconds - start_seconds,
        supervised_duration_seconds=max(0.0, end_seconds - start_seconds - BURN_IN_SECONDS),
        frame_seconds=FRAME_SECONDS,
        waveform_sample_rate=sample_rate,
        assistant_waveform_sample_rate=assistant_sample_rate,
        user_waveform=waveform,
        assistant_waveform=assistant_waveform,
        user_spans=_preview_spans(user_annotation, start_seconds, end_seconds, is_user=True),
        assistant_spans=_preview_spans(
            assistant_annotation, start_seconds, end_seconds, is_user=False
        ),
        user_points=_preview_points(user_annotation, start_seconds, end_seconds, is_user=True),
        assistant_points=_preview_points(
            assistant_annotation, start_seconds, end_seconds, is_user=False
        ),
        user_segment_targets=_preview_segment_targets(
            user_annotation.segment_targets,
            start_seconds,
            end_seconds,
        ),
        assistant_segment_targets=_preview_segment_targets(
            assistant_annotation.segment_targets,
            start_seconds,
            end_seconds,
        ),
        user_connection_targets=_preview_connection_targets(
            user_annotation.connection_targets,
            start_seconds,
            end_seconds,
        ),
        assistant_connection_targets=_preview_connection_targets(
            assistant_annotation.connection_targets,
            start_seconds,
            end_seconds,
        ),
        recording_user_spans=_preview_spans(
            user_annotation,
            0.0,
            eligible_duration_seconds,
            is_user=True,
        ),
        recording_assistant_spans=_preview_spans(
            assistant_annotation,
            0.0,
            eligible_duration_seconds,
            is_user=False,
        ),
        recording_user_points=_preview_points(
            user_annotation,
            0.0,
            eligible_duration_seconds,
            is_user=True,
        ),
        recording_assistant_points=_preview_points(
            assistant_annotation,
            0.0,
            eligible_duration_seconds,
            is_user=False,
        ),
        conversation_regions=conversation_regions,
        frames=frames,
    )


def build_training_sample_propositions(
    dashboard_sample: DashboardSample,
    user_side: TrackSide,
    conversation_regions: ConversationRegionAnalysis | None,
    limit: int,
) -> tuple[TrainingSampleProposition, ...]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    annotation = _conversation_annotation(dashboard_sample)
    represented_duration_seconds = (
        dashboard_sample.sample.duration_seconds
        if dashboard_sample.sample.duration_seconds is not None
        else annotation.analyzed_duration_seconds
    )
    eligible_duration_seconds = min(
        represented_duration_seconds, annotation.analyzed_duration_seconds
    )
    user = _speaker_annotation(annotation, user_side)
    assistant = _speaker_annotation(annotation, _other_side(user_side))
    anchors = _proposition_anchors(
        user=user,
        assistant=assistant,
        duration_seconds=eligible_duration_seconds,
    )
    selected_anchors = _balanced_proposition_anchors(
        anchors=anchors,
        duration_seconds=eligible_duration_seconds,
        conversation_regions=conversation_regions,
        limit=limit,
    )
    propositions = tuple(
        _build_proposition(
            anchor=anchor,
            duration_seconds=eligible_duration_seconds,
            user=user,
            assistant=assistant,
            conversation_regions=conversation_regions,
            all_anchors=anchors,
        )
        for anchor in selected_anchors
    )
    return tuple(
        proposition
        for proposition in propositions
        if proposition.score >= MINIMUM_PROPOSITION_SCORE
        and proposition.primary_supervision_ratio >= MINIMUM_PROPOSITION_PRIMARY_COVERAGE
        and proposition.future_activity_supervision_ratio >= MINIMUM_PROPOSITION_FUTURE_COVERAGE
        and proposition.masked_supervised_ratio <= MAXIMUM_PROPOSITION_MASKED_RATIO
    )


def _balanced_proposition_anchors(
    anchors: Sequence[PropositionAnchor],
    duration_seconds: float,
    conversation_regions: ConversationRegionAnalysis | None,
    limit: int,
) -> tuple[PropositionAnchor, ...]:
    ranked_by_kind = tuple(
        tuple(
            sorted(
                (anchor for anchor in anchors if anchor.kind is proposition_kind),
                key=lambda anchor: _anchor_preference(
                    anchor=anchor,
                    all_anchors=anchors,
                    duration_seconds=duration_seconds,
                    conversation_regions=conversation_regions,
                ),
                reverse=True,
            )
        )
        for proposition_kind in TrainingSamplePropositionKind
    )
    selected: list[PropositionAnchor] = []
    rank = 0
    while len(selected) < limit:
        added = False
        for candidates in ranked_by_kind:
            if rank < len(candidates):
                selected.append(candidates[rank])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        rank += 1
    return tuple(selected)


def _anchor_preference(
    anchor: PropositionAnchor,
    all_anchors: Sequence[PropositionAnchor],
    duration_seconds: float,
    conversation_regions: ConversationRegionAnalysis | None,
) -> tuple[float, float]:
    maximum_start_seconds = max(0.0, duration_seconds - INPUT_DURATION_SECONDS)
    start_seconds = min(
        maximum_start_seconds,
        max(0.0, anchor.time_seconds - PROPOSITION_EVENT_OFFSET_SECONDS),
    )
    end_seconds = min(duration_seconds, start_seconds + INPUT_DURATION_SECONDS)
    supervision_start_seconds = min(end_seconds, start_seconds + BURN_IN_SECONDS)
    masked_seconds, _ = _masked_supervision(
        start_seconds=supervision_start_seconds,
        end_seconds=end_seconds,
        conversation_regions=conversation_regions,
    )
    masked_ratio = _ratio(
        masked_seconds,
        max(0.0, end_seconds - supervision_start_seconds),
    )
    category_event_count = sum(
        candidate.kind is anchor.kind
        and supervision_start_seconds <= candidate.time_seconds < end_seconds
        for candidate in all_anchors
    )
    return (
        anchor.confidence
        + 0.08 * min(3, category_event_count)
        - 0.8 * max(0.0, masked_ratio - 0.2),
        -masked_ratio,
    )


def _proposition_anchors(
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    duration_seconds: float,
) -> tuple[PropositionAnchor, ...]:
    turn_shift_anchors = _tight_turn_shift_anchors(
        user=user,
        assistant=assistant,
    )
    hold_pause_anchors = tuple(
        PropositionAnchor(
            kind=TrainingSamplePropositionKind.HOLD_PAUSE,
            time_seconds=(connection.earlier_end_seconds + connection.later_start_seconds) / 2.0,
            confidence=min(
                1.0,
                connection.merge_confidence + 0.1 * min(1.0, connection.gap_seconds / 1.5),
            ),
            description=(f"User resumes after a {connection.gap_seconds:.2f} s continuation pause"),
        )
        for connection in user.connection_targets
        if connection.gap_seconds >= 0.25 and connection.merge_confidence >= 0.5
    )
    feedback_anchors = tuple(
        PropositionAnchor(
            kind=TrainingSamplePropositionKind.NON_FLOOR_FEEDBACK,
            time_seconds=(span.start_seconds + span.end_seconds) / 2.0,
            confidence=0.8,
            description="Short user response should remain non-floor feedback",
        )
        for span in user.backchannels
        if _overlap_seconds(span, assistant.speech_segments) > 0.0
    )
    interruption_anchors = tuple(
        PropositionAnchor(
            kind=TrainingSamplePropositionKind.OVERLAP_INTERRUPTION,
            time_seconds=point.time_seconds,
            confidence=point.confidence if point.confidence is not None else 0.7,
            description="User begins an interruption attempt",
        )
        for point in user.interruptions
    )
    overlap_anchors = tuple(
        PropositionAnchor(
            kind=TrainingSamplePropositionKind.OVERLAP_INTERRUPTION,
            time_seconds=time_seconds,
            confidence=0.65,
            description="Both speakers are active at the same time",
        )
        for time_seconds in _overlap_onsets(user.speech_segments, assistant.speech_segments)
    )
    return (
        *_spread_anchors(turn_shift_anchors),
        *_spread_anchors(hold_pause_anchors),
        *_spread_anchors(feedback_anchors),
        *_spread_anchors((*interruption_anchors, *overlap_anchors)),
    )


def _tight_turn_shift_anchors(
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
) -> tuple[PropositionAnchor, ...]:
    return (
        *_speaker_shift_anchors(
            releasing_segments=user.segment_targets,
            responding_segments=assistant.segment_targets,
            description="User releases the floor",
        ),
        *_speaker_shift_anchors(
            releasing_segments=assistant.segment_targets,
            responding_segments=user.segment_targets,
            description="User takes the floor after the other speaker",
        ),
    )


def _speaker_shift_anchors(
    releasing_segments: Sequence[SegmentAnnotationTarget],
    responding_segments: Sequence[SegmentAnnotationTarget],
    description: str,
) -> tuple[PropositionAnchor, ...]:
    transcript_responses = tuple(
        segment
        for segment in responding_segments
        if segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
    )
    anchors: list[PropositionAnchor] = []
    for releasing_segment in releasing_segments:
        if releasing_segment.evidence_source is not AnnotationEvidenceSource.TRANSCRIPT:
            continue
        response = next(
            (
                segment
                for segment in transcript_responses
                if segment.start_seconds >= releasing_segment.end_seconds
            ),
            None,
        )
        if response is None:
            continue
        gap_seconds = response.start_seconds - releasing_segment.end_seconds
        if gap_seconds > MAXIMUM_TURN_SHIFT_GAP_SECONDS:
            continue
        timing_confidence = 1.0 - 0.25 * gap_seconds / MAXIMUM_TURN_SHIFT_GAP_SECONDS
        anchors.append(
            PropositionAnchor(
                kind=TrainingSamplePropositionKind.TURN_SHIFT,
                time_seconds=releasing_segment.end_seconds,
                confidence=timing_confidence
                * min(
                    releasing_segment.turn_confidence,
                    response.turn_confidence,
                ),
                description=f"{description} with a {gap_seconds:.2f} s response gap",
            )
        )
    return tuple(anchors)


def _overlap_seconds(
    span: AnnotationSpan,
    candidates: Sequence[AnnotationSpan],
) -> float:
    return sum(
        max(
            0.0,
            min(span.end_seconds, candidate.end_seconds)
            - max(span.start_seconds, candidate.start_seconds),
        )
        for candidate in candidates
    )


def _spread_anchors(
    anchors: Sequence[PropositionAnchor],
) -> tuple[PropositionAnchor, ...]:
    ordered = tuple(
        sorted(
            anchors,
            key=lambda anchor: (
                anchor.time_seconds,
                anchor.kind.value,
                anchor.description,
            ),
        )
    )
    if len(ordered) <= PROPOSITION_ANCHOR_LIMIT_PER_KIND:
        return ordered
    selected_indices = {
        round(index * (len(ordered) - 1) / (PROPOSITION_ANCHOR_LIMIT_PER_KIND - 1))
        for index in range(PROPOSITION_ANCHOR_LIMIT_PER_KIND)
    }
    return tuple(ordered[index] for index in sorted(selected_indices))


def _overlap_onsets(
    user_spans: Sequence[AnnotationSpan],
    assistant_spans: Sequence[AnnotationSpan],
) -> tuple[float, ...]:
    onsets: list[float] = []
    user_index = 0
    assistant_index = 0
    while user_index < len(user_spans) and assistant_index < len(assistant_spans):
        user_span = user_spans[user_index]
        assistant_span = assistant_spans[assistant_index]
        start_seconds = max(user_span.start_seconds, assistant_span.start_seconds)
        end_seconds = min(user_span.end_seconds, assistant_span.end_seconds)
        if end_seconds > start_seconds:
            onsets.append(start_seconds)
        if user_span.end_seconds <= assistant_span.end_seconds:
            user_index += 1
        else:
            assistant_index += 1
    return tuple(onsets)


def _build_proposition(
    anchor: PropositionAnchor,
    duration_seconds: float,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    conversation_regions: ConversationRegionAnalysis | None,
    all_anchors: Sequence[PropositionAnchor],
) -> TrainingSampleProposition:
    maximum_start_seconds = max(0.0, duration_seconds - INPUT_DURATION_SECONDS)
    start_seconds = min(
        maximum_start_seconds,
        max(0.0, anchor.time_seconds - PROPOSITION_EVENT_OFFSET_SECONDS),
    )
    end_seconds = min(duration_seconds, start_seconds + INPUT_DURATION_SECONDS)
    frames = build_frame_previews(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        annotation_end_seconds=duration_seconds,
        user=user,
        assistant=assistant,
    )
    supervised_frames = tuple(frame for frame in frames if frame.supervised)
    primary_supervision_ratio = _ratio(
        sum(frame.user_yield_valid for frame in supervised_frames),
        len(supervised_frames),
    )
    future_targets = tuple(
        target for frame in supervised_frames for target in frame.future_activity
    )
    future_activity_supervision_ratio = _ratio(
        sum(target.valid for target in future_targets),
        len(future_targets),
    )
    event_anchor_count = sum(frame.interaction_event_valid for frame in supervised_frames)
    supervision_start_seconds = min(end_seconds, start_seconds + BURN_IN_SECONDS)
    masked_supervised_seconds, masked_reasons = _masked_supervision(
        start_seconds=supervision_start_seconds,
        end_seconds=end_seconds,
        conversation_regions=conversation_regions,
    )
    supervised_duration_seconds = max(0.0, end_seconds - supervision_start_seconds)
    masked_supervised_ratio = _ratio(masked_supervised_seconds, supervised_duration_seconds)
    category_event_count = sum(
        candidate.kind is anchor.kind
        and supervision_start_seconds <= candidate.time_seconds < end_seconds
        for candidate in all_anchors
    )
    retained_ratio = 1.0 - masked_supervised_ratio
    event_density_score = min(1.0, event_anchor_count / 3.0)
    category_density_score = min(1.0, category_event_count / 3.0)
    excessive_mask_penalty = 0.7 * max(0.0, masked_supervised_ratio - 0.2)
    score = min(
        1.0,
        max(
            0.0,
            0.25 * primary_supervision_ratio
            + 0.15 * future_activity_supervision_ratio
            + 0.2 * anchor.confidence
            + 0.1 * event_density_score
            + 0.15 * retained_ratio
            + 0.15 * category_density_score
            - excessive_mask_penalty,
        ),
    )
    return TrainingSampleProposition(
        kind=anchor.kind,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        anchor_seconds=anchor.time_seconds,
        score=score,
        category_confidence=anchor.confidence,
        primary_supervision_ratio=primary_supervision_ratio,
        future_activity_supervision_ratio=future_activity_supervision_ratio,
        masked_supervised_seconds=masked_supervised_seconds,
        masked_supervised_ratio=masked_supervised_ratio,
        event_anchor_count=event_anchor_count,
        category_event_count=category_event_count,
        masked_reasons=masked_reasons,
        description=anchor.description,
    )


def _masked_supervision(
    start_seconds: float,
    end_seconds: float,
    conversation_regions: ConversationRegionAnalysis | None,
) -> tuple[float, tuple[str, ...]]:
    if conversation_regions is None:
        return 0.0, ()
    overlapping_regions = tuple(
        region
        for region in conversation_regions.unusable_regions
        if region.end_seconds > start_seconds and region.start_seconds < end_seconds
    )
    masked_seconds = sum(
        min(end_seconds, region.end_seconds) - max(start_seconds, region.start_seconds)
        for region in overlapping_regions
    )
    reasons = tuple(
        sorted({reason.value for region in overlapping_regions for reason in region.reasons})
    )
    return masked_seconds, reasons


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _conversation_annotation(dashboard_sample: DashboardSample) -> ConversationAnnotation:
    quality = dashboard_sample.latest_quality
    if quality is None:
        raise ValueError("The selected sample has no conversation quality analysis.")
    result = QualityResult.model_validate(quality.payload)
    if result.conversation_annotation is None:
        raise ValueError("The selected sample has no conversation annotation.")
    return result.conversation_annotation


def _training_sample_gains(
    dashboard_sample: DashboardSample,
    user_side: TrackSide,
) -> tuple[TrackGainNormalization, TrackGainNormalization]:
    quality = dashboard_sample.latest_quality
    if quality is None:
        raise ValueError("The selected sample has no conversation quality analysis.")
    audio_quality = QualityResult.model_validate(quality.payload).audio_quality
    speaker1_quality = audio_quality.speaker1 if audio_quality is not None else None
    speaker2_quality = audio_quality.speaker2 if audio_quality is not None else None
    match user_side:
        case TrackSide.SPEAKER1:
            return (
                track_gain_normalization(speaker1_quality),
                track_gain_normalization(speaker2_quality),
            )
        case TrackSide.SPEAKER2:
            return (
                track_gain_normalization(speaker2_quality),
                track_gain_normalization(speaker1_quality),
            )


def _training_sample_quality(dashboard_sample: DashboardSample) -> TrainingSampleQuality:
    quality = dashboard_sample.latest_quality
    if quality is None:
        raise ValueError("The selected sample has no conversation quality analysis.")
    return TrainingSampleQuality(
        total_score=quality.total_quality_score,
        interaction_density_score=quality.interaction_density_score,
        timing_reliability_score=quality.timing_reliability_score,
        audio_quality_score=quality.audio_quality_score,
        conversation_quality_score=quality.conversation_quality_score,
        usable_event_count=quality.usable_event_count,
        events_per_hour=quality.conversation_events_per_hour,
        flags=quality.flags,
    )


def _select_start_seconds(
    duration_seconds: float,
    requested_start_seconds: float | None,
    selection_mode: TrainingSampleSelectionMode,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    generator: random.Random,
) -> float:
    if requested_start_seconds is not None:
        return _sample_start_seconds(
            duration_seconds=duration_seconds,
            requested_start_seconds=requested_start_seconds,
            generator=generator,
        )
    match selection_mode:
        case TrainingSampleSelectionMode.RANDOM:
            return _sample_start_seconds(
                duration_seconds=duration_seconds,
                requested_start_seconds=None,
                generator=generator,
            )
        case TrainingSampleSelectionMode.INTERESTING:
            return _interesting_start_seconds(
                duration_seconds=duration_seconds,
                user=user,
                assistant=assistant,
                generator=generator,
            )


def _sample_start_seconds(
    duration_seconds: float,
    requested_start_seconds: float | None,
    generator: random.Random,
) -> float:
    maximum_start = max(0.0, duration_seconds - INPUT_DURATION_SECONDS)
    if requested_start_seconds is None:
        return generator.uniform(0.0, maximum_start) if maximum_start > 0.0 else 0.0
    if requested_start_seconds < 0.0 or requested_start_seconds > maximum_start:
        raise ValueError(
            f"start_seconds must be between 0 and {maximum_start:.3f} for this sample."
        )
    return requested_start_seconds


def _interesting_start_seconds(
    duration_seconds: float,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    generator: random.Random,
) -> float:
    maximum_start = max(0.0, duration_seconds - INPUT_DURATION_SECONDS)
    if maximum_start == 0.0:
        return 0.0
    event_times = _activity_event_times((user, assistant))
    probability_spans = _user_probability_spans(user)
    anchors = [
        *event_times,
        *((span.start_seconds + span.end_seconds) / 2.0 for span in probability_spans),
    ]
    if len(anchors) > INTERESTING_ANCHOR_LIMIT:
        anchors = generator.sample(anchors, INTERESTING_ANCHOR_LIMIT)
    candidate_starts = {
        min(maximum_start, max(0.0, anchor - INPUT_DURATION_SECONDS / 2.0)) for anchor in anchors
    }
    candidate_starts.update(
        generator.uniform(0.0, maximum_start) for _ in range(INTERESTING_RANDOM_LOCATION_COUNT)
    )
    ranked_starts = sorted(
        (
            _interesting_location_score(
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                event_times=event_times,
                probability_spans=probability_spans,
            ),
            start_seconds,
        )
        for start_seconds in candidate_starts
    )
    top_starts = ranked_starts[-min(5, len(ranked_starts)) :]
    return top_starts[generator.randrange(len(top_starts))][1]


def _activity_event_times(
    speakers: Sequence[SpeakerConversationAnnotation],
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                time_seconds
                for speaker in speakers
                for time_seconds in (
                    *(segment.end_seconds for segment in speaker.segment_targets),
                    *(span.start_seconds for span in speaker.pauses),
                    *(span.start_seconds for span in speaker.backchannels),
                    *(point.time_seconds for point in speaker.turns),
                    *(point.time_seconds for point in speaker.interruptions),
                )
            }
        )
    )


def _user_probability_spans(
    user: SpeakerConversationAnnotation,
) -> tuple[ProbabilitySpan, ...]:
    return (
        *(
            ProbabilitySpan(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                yield_probability=segment.keep_playing_confidence,
            )
            for segment in user.segment_targets
        ),
        *(
            ProbabilitySpan(
                start_seconds=connection.earlier_end_seconds,
                end_seconds=connection.later_start_seconds,
                yield_probability=1.0 - connection.merge_confidence,
            )
            for connection in user.connection_targets
        ),
    )


def _interesting_location_score(
    start_seconds: float,
    duration_seconds: float,
    event_times: Sequence[float],
    probability_spans: Sequence[ProbabilitySpan],
) -> float:
    supervision_start_seconds = start_seconds + BURN_IN_SECONDS
    end_seconds = min(duration_seconds, start_seconds + INPUT_DURATION_SECONDS)
    event_score = float(
        sum(supervision_start_seconds <= event_time < end_seconds for event_time in event_times)
    )
    ambiguous_duration_seconds = sum(
        max(
            0.0,
            min(end_seconds, span.end_seconds) - max(supervision_start_seconds, span.start_seconds),
        )
        * (1.0 - 2.0 * abs(span.yield_probability - 0.5))
        for span in probability_spans
    )
    ambiguity_score = 4.0 * ambiguous_duration_seconds
    return max(event_score, ambiguity_score) + 0.25 * min(event_score, ambiguity_score)


def _other_side(side: TrackSide) -> TrackSide:
    match side:
        case TrackSide.SPEAKER1:
            return TrackSide.SPEAKER2
        case TrackSide.SPEAKER2:
            return TrackSide.SPEAKER1


def _speaker_annotation(
    annotation: ConversationAnnotation, side: TrackSide
) -> SpeakerConversationAnnotation:
    match side:
        case TrackSide.SPEAKER1:
            expected_side = SpeakerSide.SPEAKER1
            speaker = annotation.speaker1
        case TrackSide.SPEAKER2:
            expected_side = SpeakerSide.SPEAKER2
            speaker = annotation.speaker2
    assert speaker.side == expected_side
    return speaker


def _track(dashboard_sample: DashboardSample, side: TrackSide) -> SampleTrackRecord:
    track = next(
        (candidate for candidate in dashboard_sample.tracks if candidate.side == side), None
    )
    if track is None:
        raise ValueError(f"The selected sample has no {side.value} track.")
    return track


def _track_path(track: SampleTrackRecord) -> Path:
    return materialize_sample_track(track)


def required_track_sha256(track: SampleTrackRecord) -> str:
    if track.audio_sha256 is None:
        raise ValueError(f"Track has not been materialized by ingestion: {track.id}")
    return track.audio_sha256


def build_frame_previews(
    start_seconds: float,
    end_seconds: float,
    annotation_end_seconds: float,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
) -> tuple[TrainingFramePreview, ...]:
    boundaries = _decision_boundaries(
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    )
    event_anchors = _interaction_event_anchors(
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    )
    frame_count = max(1, math.ceil((end_seconds - start_seconds) / FRAME_SECONDS))
    frames: list[TrainingFramePreview] = []
    for frame_index in range(frame_count):
        time_seconds = min(
            end_seconds,
            start_seconds + (frame_index + 0.5) * FRAME_SECONDS,
        )
        supervised = time_seconds >= start_seconds + BURN_IN_SECONDS
        boundary = _active_boundary(time_seconds=time_seconds, boundaries=boundaries)
        event_anchor = _event_anchor_for_frame(
            frame_index=frame_index,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            frame_count=frame_count,
            anchors=event_anchors,
        )
        yield_supervision = _user_yield_supervision(
            time_seconds=time_seconds,
            supervised=supervised,
            annotation_end_seconds=annotation_end_seconds,
            user=user,
        )
        has_floor_supervision = _user_has_floor_supervision(
            time_seconds=time_seconds,
            supervised=supervised,
            user=user,
        )
        event_distribution, event_valid, event_mask_reason = _interaction_event_supervision(
            supervised=supervised,
            anchor=event_anchor,
        )
        candidate = boundary is not None or event_anchor is not None
        candidate_source = (
            event_anchor.source
            if event_anchor is not None
            else boundary.source
            if boundary is not None
            else None
        )
        anchor_time_seconds = (
            event_anchor.time_seconds
            if event_anchor is not None
            else boundary.speech_offset_seconds
            if boundary is not None
            else None
        )
        frames.append(
            TrainingFramePreview(
                frame_index=frame_index,
                time_seconds=time_seconds,
                relative_time_seconds=time_seconds - start_seconds,
                supervised=supervised,
                assistant_speaking_input=_assistant_speaking_at(
                    time_seconds=time_seconds,
                    assistant=assistant,
                ),
                candidate=candidate,
                candidate_source=candidate_source,
                seconds_since_speech_offset=(
                    time_seconds - anchor_time_seconds if anchor_time_seconds is not None else None
                ),
                user_yield_target=yield_supervision.target,
                user_yield_valid=yield_supervision.valid,
                user_yield_mask_reason=yield_supervision.mask_reason,
                user_has_floor_target=has_floor_supervision.target,
                user_has_floor_valid=has_floor_supervision.valid,
                user_has_floor_mask_reason=has_floor_supervision.mask_reason,
                interaction_event_distribution=event_distribution,
                interaction_event_valid=event_valid,
                interaction_event_mask_reason=event_mask_reason,
                future_activity=_future_activity_targets(
                    time_seconds=time_seconds,
                    supervised=supervised,
                    annotation_end_seconds=annotation_end_seconds,
                    user=user,
                ),
            )
        )
    return tuple(frames)


def _user_yield_supervision(
    time_seconds: float,
    supervised: bool,
    annotation_end_seconds: float,
    user: SpeakerConversationAnnotation,
) -> ScalarSupervision:
    if not supervised:
        return _masked_scalar(SupervisionMaskReason.BURN_IN)
    horizon_time_seconds = time_seconds + USER_YIELD_HORIZON_SECONDS
    if horizon_time_seconds > annotation_end_seconds:
        return _masked_scalar(SupervisionMaskReason.FUTURE_HORIZON_CENSORED)
    future_floor = _user_floor_state_supervision(
        time_seconds=horizon_time_seconds,
        user=user,
    )
    if not future_floor.valid:
        assert future_floor.mask_reason is not None
        return _masked_scalar(future_floor.mask_reason)
    assert future_floor.target is not None
    return _valid_scalar(1.0 - future_floor.target)


def _user_has_floor_supervision(
    time_seconds: float,
    supervised: bool,
    user: SpeakerConversationAnnotation,
) -> ScalarSupervision:
    if not supervised:
        return _masked_scalar(SupervisionMaskReason.BURN_IN)
    return _user_floor_state_supervision(
        time_seconds=time_seconds,
        user=user,
    )


def _user_floor_state_supervision(
    time_seconds: float,
    user: SpeakerConversationAnnotation,
) -> ScalarSupervision:
    active_transcript_segment = next(
        (
            segment
            for segment in user.segment_targets
            if segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
            and segment.start_seconds <= time_seconds < segment.end_seconds
        ),
        None,
    )
    if active_transcript_segment is not None:
        return _valid_scalar(active_transcript_segment.turn_confidence)
    active_connection = next(
        (
            connection
            for connection in user.connection_targets
            if connection.earlier_end_seconds <= time_seconds < connection.later_start_seconds
        ),
        None,
    )
    if active_connection is not None:
        earlier_segment = next(
            (
                segment
                for segment in user.segment_targets
                if segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
                and abs(segment.end_seconds - active_connection.earlier_end_seconds)
                <= FRAME_SECONDS
            ),
            None,
        )
        later_segment = next(
            (
                segment
                for segment in user.segment_targets
                if segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
                and abs(segment.start_seconds - active_connection.later_start_seconds)
                <= FRAME_SECONDS
            ),
            None,
        )
        if earlier_segment is None or later_segment is None:
            return _masked_scalar(SupervisionMaskReason.AMBIGUOUS_ANNOTATION)
        # A gap holds the floor only to the extent that it joins two floor-taking segments.
        return _valid_scalar(
            active_connection.merge_confidence
            * min(
                earlier_segment.turn_confidence,
                later_segment.turn_confidence,
            )
        )
    active_audio_activity = any(
        segment.evidence_source is AnnotationEvidenceSource.AUDIO_ACTIVITY
        and segment.start_seconds <= time_seconds < segment.end_seconds
        for segment in user.segment_targets
    )
    if active_audio_activity:
        return _masked_scalar(SupervisionMaskReason.AMBIGUOUS_ANNOTATION)
    return _valid_scalar(0.0)


def _detector_turn_point_at(
    time_seconds: float,
    points: Sequence[AnnotationPoint],
) -> bool:
    return any(
        abs(point.time_seconds - time_seconds) <= FRAME_SECONDS
        and (point.confidence is None or point.confidence >= HIGH_CONFIDENCE)
        for point in points
    )


def _valid_scalar(target: float) -> ScalarSupervision:
    return ScalarSupervision(target=target, valid=True, mask_reason=None)


def _masked_scalar(reason: SupervisionMaskReason) -> ScalarSupervision:
    return ScalarSupervision(target=None, valid=False, mask_reason=reason)


def _assistant_speaking_at(
    time_seconds: float,
    assistant: SpeakerConversationAnnotation,
) -> bool:
    inside_turn = _inside_span(time_seconds, assistant.speech_segments)
    inside_pause = _inside_span(time_seconds, assistant.pauses)
    inside_backchannel = _inside_span(time_seconds, assistant.backchannels)
    return (inside_turn and not inside_pause) or inside_backchannel


def _assistant_substantively_speaking_at(
    time_seconds: float,
    assistant: SpeakerConversationAnnotation,
) -> bool:
    return _assistant_substantive_segment_at(
        time_seconds, assistant
    ) is not None and _assistant_speaking_at(time_seconds, assistant)


def _assistant_substantive_segment_at(
    time_seconds: float,
    assistant: SpeakerConversationAnnotation,
) -> SegmentAnnotationTarget | None:
    return next(
        (
            segment
            for segment in assistant.segment_targets
            if segment.start_seconds <= time_seconds < segment.end_seconds
            and segment.turn_confidence >= HIGH_CONFIDENCE
            and segment.evidence_source is AnnotationEvidenceSource.TRANSCRIPT
        ),
        None,
    )


def _inside_span(time_seconds: float, spans: Sequence[AnnotationSpan]) -> bool:
    return any(span.start_seconds <= time_seconds < span.end_seconds for span in spans)


def _decision_boundaries(
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> tuple[DecisionBoundary, ...]:
    return tuple(
        _decision_boundary(
            segment=segment,
            user=user,
            assistant=assistant,
            annotation_end_seconds=annotation_end_seconds,
        )
        for segment in user.segment_targets
    )


def _decision_boundary(
    segment: SegmentAnnotationTarget,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> DecisionBoundary:
    connection = _matching_connection(segment=segment, connections=user.connection_targets)
    next_assistant = _next_segment_after(
        time_seconds=segment.end_seconds,
        segments=assistant.segment_targets,
    )
    next_user = _next_segment_after(
        time_seconds=segment.end_seconds,
        segments=user.segment_targets,
    )
    next_activity_seconds = min(
        (
            candidate.start_seconds
            for candidate in (next_user, next_assistant)
            if candidate is not None
        ),
        default=annotation_end_seconds,
    )
    candidate_end_seconds = min(
        annotation_end_seconds,
        segment.end_seconds + MAXIMUM_CANDIDATE_SILENCE_SECONDS,
        next_activity_seconds,
    )
    source: CandidateSource
    if connection is not None:
        source = CandidateSource.CONNECTION
    elif (
        next_assistant is not None
        or segment.end_seconds + MAXIMUM_CANDIDATE_SILENCE_SECONDS <= annotation_end_seconds
    ):
        source = CandidateSource.SEGMENT_END
    else:
        source = CandidateSource.CENSORED
    return DecisionBoundary(
        speech_offset_seconds=segment.end_seconds,
        candidate_end_seconds=max(segment.end_seconds + FRAME_SECONDS, candidate_end_seconds),
        source=source,
    )


def _matching_connection(
    segment: SegmentAnnotationTarget,
    connections: Sequence[ConnectionAnnotationTarget],
) -> ConnectionAnnotationTarget | None:
    return next(
        (
            connection
            for connection in connections
            if abs(connection.earlier_end_seconds - segment.end_seconds) <= FRAME_SECONDS
        ),
        None,
    )


def _next_segment_after(
    time_seconds: float,
    segments: Sequence[SegmentAnnotationTarget],
) -> SegmentAnnotationTarget | None:
    return min(
        (segment for segment in segments if segment.start_seconds >= time_seconds),
        key=lambda segment: segment.start_seconds,
        default=None,
    )


def _active_boundary(
    time_seconds: float, boundaries: Sequence[DecisionBoundary]
) -> DecisionBoundary | None:
    return max(
        (
            boundary
            for boundary in boundaries
            if boundary.speech_offset_seconds <= time_seconds < boundary.candidate_end_seconds
        ),
        key=lambda boundary: boundary.speech_offset_seconds,
        default=None,
    )


def _interaction_event_anchors(
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> tuple[InteractionEventAnchor, ...]:
    anchors: list[InteractionEventAnchor] = []
    for segment in user.segment_targets:
        anchors.append(
            _boundary_event_anchor(
                segment=segment,
                user=user,
                assistant=assistant,
                annotation_end_seconds=annotation_end_seconds,
            )
        )
        assistant_segment = _assistant_substantive_segment_at(segment.start_seconds, assistant)
        if assistant_segment is not None and _assistant_speaking_at(
            segment.start_seconds, assistant
        ):
            distribution = _overlap_event_distribution(
                user_segment=segment,
                assistant_segment=assistant_segment,
            )
            anchors.append(
                InteractionEventAnchor(
                    time_seconds=segment.start_seconds,
                    source=CandidateSource.OVERLAP_ONSET,
                    distribution=distribution,
                    mask_reason=(
                        None
                        if distribution is not None
                        else SupervisionMaskReason.AMBIGUOUS_ANNOTATION
                    ),
                )
            )
    return tuple(anchors)


def _boundary_event_anchor(
    segment: SegmentAnnotationTarget,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> InteractionEventAnchor:
    connection = _matching_connection(segment, user.connection_targets)
    if segment.evidence_source is not AnnotationEvidenceSource.TRANSCRIPT:
        source = (
            CandidateSource.CONNECTION if connection is not None else CandidateSource.SEGMENT_END
        )
        return _masked_boundary_event(segment, source)
    if connection is not None and connection.merge_confidence >= HIGH_CONFIDENCE:
        return InteractionEventAnchor(
            time_seconds=segment.end_seconds,
            source=CandidateSource.CONNECTION,
            distribution=_one_hot_event_distribution(continuation_pause=1.0),
            mask_reason=None,
        )
    if connection is not None and connection.merge_confidence > LOW_CONFIDENCE:
        return _masked_boundary_event(segment, CandidateSource.CONNECTION)
    source = CandidateSource.CONNECTION if connection is not None else CandidateSource.SEGMENT_END
    if segment.turn_confidence >= HIGH_CONFIDENCE and (
        _detector_turn_point_at(time_seconds=segment.end_seconds, points=user.turns)
        or _completion_confirmed(
            segment=segment,
            user=user,
            assistant=assistant,
            annotation_end_seconds=annotation_end_seconds,
        )
    ):
        return InteractionEventAnchor(
            time_seconds=segment.end_seconds,
            source=source,
            distribution=_one_hot_event_distribution(turn_completion=1.0),
            mask_reason=None,
        )
    if _completion_is_censored(
        segment=segment,
        user=user,
        assistant=assistant,
        annotation_end_seconds=annotation_end_seconds,
    ):
        return InteractionEventAnchor(
            time_seconds=segment.end_seconds,
            source=CandidateSource.CENSORED,
            distribution=None,
            mask_reason=SupervisionMaskReason.CENSORED_ANNOTATION,
        )
    return _masked_boundary_event(segment, source)


def _masked_boundary_event(
    segment: SegmentAnnotationTarget,
    source: CandidateSource,
) -> InteractionEventAnchor:
    return InteractionEventAnchor(
        time_seconds=segment.end_seconds,
        source=source,
        distribution=None,
        mask_reason=SupervisionMaskReason.AMBIGUOUS_ANNOTATION,
    )


def _overlap_event_distribution(
    user_segment: SegmentAnnotationTarget,
    assistant_segment: SegmentAnnotationTarget,
) -> EventTargetDistribution | None:
    if (
        user_segment.evidence_source is not AnnotationEvidenceSource.TRANSCRIPT
        or assistant_segment.evidence_source is not AnnotationEvidenceSource.TRANSCRIPT
    ):
        return None
    floor_take = (
        user_segment.turn_confidence >= HIGH_CONFIDENCE
        and user_segment.interruption_confidence >= HIGH_CONFIDENCE
        and user_segment.keep_playing_confidence < HIGH_CONFIDENCE
    )
    non_floor_feedback = (
        user_segment.keep_playing_confidence >= HIGH_CONFIDENCE
        and user_segment.end_seconds - user_segment.start_seconds
        <= MAXIMUM_NON_FLOOR_FEEDBACK_SECONDS
        and assistant_segment.start_seconds <= user_segment.start_seconds
        and assistant_segment.end_seconds
        >= user_segment.end_seconds + MINIMUM_ASSISTANT_CONTINUATION_SECONDS
    )
    if floor_take == non_floor_feedback:
        return None
    if floor_take:
        return _one_hot_event_distribution(floor_take=1.0)
    return _one_hot_event_distribution(non_floor_feedback=1.0)


def _one_hot_event_distribution(
    turn_completion: float = 0.0,
    continuation_pause: float = 0.0,
    non_floor_feedback: float = 0.0,
    floor_take: float = 0.0,
) -> EventTargetDistribution:
    return EventTargetDistribution(
        turn_completion=turn_completion,
        continuation_pause=continuation_pause,
        non_floor_feedback=non_floor_feedback,
        floor_take=floor_take,
    )


def _completion_confirmed(
    segment: SegmentAnnotationTarget,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> bool:
    next_user = _next_segment_after(segment.end_seconds, user.segment_targets)
    next_assistant = _next_segment_after(segment.end_seconds, assistant.segment_targets)
    if next_assistant is not None and (
        next_user is None or next_assistant.start_seconds < next_user.start_seconds
    ):
        return True
    inactivity_end_seconds = segment.end_seconds + MINIMUM_COMPLETION_INACTIVITY_SECONDS
    return annotation_end_seconds >= inactivity_end_seconds and (
        next_user is None or next_user.start_seconds >= inactivity_end_seconds
    )


def _completion_is_censored(
    segment: SegmentAnnotationTarget,
    user: SpeakerConversationAnnotation,
    assistant: SpeakerConversationAnnotation,
    annotation_end_seconds: float,
) -> bool:
    next_user = _next_segment_after(segment.end_seconds, user.segment_targets)
    next_assistant = _next_segment_after(segment.end_seconds, assistant.segment_targets)
    return (
        next_user is None
        and next_assistant is None
        and annotation_end_seconds < segment.end_seconds + MINIMUM_COMPLETION_INACTIVITY_SECONDS
    )


def _event_anchor_for_frame(
    frame_index: int,
    start_seconds: float,
    end_seconds: float,
    frame_count: int,
    anchors: Sequence[InteractionEventAnchor],
) -> InteractionEventAnchor | None:
    matching_anchors = tuple(
        anchor
        for anchor in anchors
        if start_seconds <= anchor.time_seconds <= end_seconds
        and _causal_frame_index(anchor.time_seconds, start_seconds, frame_count) == frame_index
    )
    if not matching_anchors:
        return None
    if len(matching_anchors) == 1:
        return matching_anchors[0]
    source = (
        CandidateSource.OVERLAP_ONSET
        if any(anchor.source is CandidateSource.OVERLAP_ONSET for anchor in matching_anchors)
        else matching_anchors[0].source
    )
    return InteractionEventAnchor(
        time_seconds=min(anchor.time_seconds for anchor in matching_anchors),
        source=source,
        distribution=None,
        mask_reason=SupervisionMaskReason.AMBIGUOUS_ANNOTATION,
    )


def _causal_frame_index(
    time_seconds: float,
    start_seconds: float,
    frame_count: int,
) -> int:
    first_frame_time_seconds = start_seconds + FRAME_SECONDS / 2.0
    unbounded_index = math.ceil((time_seconds - first_frame_time_seconds) / FRAME_SECONDS - 1e-9)
    return min(frame_count - 1, max(0, unbounded_index))


def _interaction_event_supervision(
    supervised: bool,
    anchor: InteractionEventAnchor | None,
) -> tuple[EventTargetDistribution | None, bool, SupervisionMaskReason | None]:
    if not supervised:
        return None, False, SupervisionMaskReason.BURN_IN
    if anchor is None:
        return None, False, SupervisionMaskReason.NO_EVENT_ANCHOR
    if anchor.distribution is None:
        assert anchor.mask_reason is not None
        return None, False, anchor.mask_reason
    return anchor.distribution, True, None


def _future_activity_targets(
    time_seconds: float,
    supervised: bool,
    annotation_end_seconds: float,
    user: SpeakerConversationAnnotation,
) -> tuple[FutureActivityTarget, ...]:
    activity_spans = (*user.speech_segments, *user.backchannels)
    targets: list[FutureActivityTarget] = []
    for start_milliseconds, end_milliseconds in FUTURE_ACTIVITY_WINDOWS_MILLISECONDS:
        start = time_seconds + start_milliseconds / 1000.0
        end = time_seconds + end_milliseconds / 1000.0
        horizon_available = end <= annotation_end_seconds
        valid = supervised and horizon_available
        mask_reason: SupervisionMaskReason | None
        if not supervised:
            mask_reason = SupervisionMaskReason.BURN_IN
        elif not horizon_available:
            mask_reason = SupervisionMaskReason.FUTURE_HORIZON_CENSORED
        else:
            mask_reason = None
        targets.append(
            FutureActivityTarget(
                start_milliseconds=start_milliseconds,
                end_milliseconds=end_milliseconds,
                occupancy=_active_fraction(start, end, activity_spans) if valid else None,
                valid=valid,
                mask_reason=mask_reason,
            )
        )
    return tuple(targets)


def _active_fraction(
    start_seconds: float,
    end_seconds: float,
    spans: Sequence[AnnotationSpan],
) -> float:
    duration_seconds = end_seconds - start_seconds
    assert duration_seconds > 0.0
    clipped_intervals = sorted(
        (
            max(start_seconds, span.start_seconds),
            min(end_seconds, span.end_seconds),
        )
        for span in spans
        if span.end_seconds > start_seconds and span.start_seconds < end_seconds
    )
    active_seconds = 0.0
    merged_end_seconds = start_seconds
    for interval_start_seconds, interval_end_seconds in clipped_intervals:
        if interval_end_seconds <= merged_end_seconds:
            continue
        active_seconds += interval_end_seconds - max(interval_start_seconds, merged_end_seconds)
        merged_end_seconds = interval_end_seconds
    return min(1.0, active_seconds / duration_seconds)


def _preview_spans(
    annotation: SpeakerConversationAnnotation,
    start_seconds: float,
    end_seconds: float,
    is_user: bool,
) -> tuple[PreviewSpan, ...]:
    speech_type = PreviewEventType.USER_SPEECH if is_user else PreviewEventType.ASSISTANT_SPEECH
    pause_type = PreviewEventType.USER_PAUSE if is_user else PreviewEventType.ASSISTANT_PAUSE
    backchannel_type = (
        PreviewEventType.USER_BACKCHANNEL if is_user else PreviewEventType.ASSISTANT_BACKCHANNEL
    )
    spans = [
        *_spans_of_type(annotation.speech_segments, speech_type),
        *_spans_of_type(annotation.pauses, pause_type),
        *_spans_of_type(annotation.backchannels, backchannel_type),
    ]
    return tuple(
        span
        for span in sorted(
            spans, key=lambda candidate: (candidate.start_seconds, candidate.end_seconds)
        )
        if span.end_seconds >= start_seconds and span.start_seconds <= end_seconds
    )


def _spans_of_type(
    spans: Sequence[AnnotationSpan], event_type: PreviewEventType
) -> list[PreviewSpan]:
    return [
        PreviewSpan(
            event_type=event_type,
            start_seconds=span.start_seconds,
            end_seconds=span.end_seconds,
            text=span.text,
        )
        for span in spans
    ]


def _preview_points(
    annotation: SpeakerConversationAnnotation,
    start_seconds: float,
    end_seconds: float,
    is_user: bool,
) -> tuple[PreviewPoint, ...]:
    turn_type = PreviewEventType.USER_END_OF_TURN
    interruption_type = PreviewEventType.USER_INTERRUPTION
    if not is_user:
        turn_type = PreviewEventType.ASSISTANT_END_OF_TURN
        interruption_type = PreviewEventType.ASSISTANT_INTERRUPTION
    points = [
        *(_points_of_type(annotation.turns, turn_type)),
        *(_points_of_type(annotation.interruptions, interruption_type)),
    ]
    return tuple(
        point
        for point in sorted(points, key=lambda candidate: candidate.time_seconds)
        if start_seconds <= point.time_seconds <= end_seconds
    )


def _points_of_type(
    points: Sequence[AnnotationPoint], event_type: PreviewEventType
) -> list[PreviewPoint]:
    return [
        PreviewPoint(
            event_type=event_type,
            time_seconds=point.time_seconds,
            confidence=point.confidence,
            text=point.text,
        )
        for point in points
    ]


def _preview_segment_targets(
    targets: Sequence[SegmentAnnotationTarget],
    start_seconds: float,
    end_seconds: float,
) -> tuple[SegmentAnnotationTarget, ...]:
    return tuple(
        target
        for target in targets
        if target.end_seconds >= start_seconds and target.start_seconds <= end_seconds
    )


def _preview_connection_targets(
    targets: Sequence[ConnectionAnnotationTarget],
    start_seconds: float,
    end_seconds: float,
) -> tuple[ConnectionAnnotationTarget, ...]:
    return tuple(
        target
        for target in targets
        if target.later_start_seconds >= start_seconds and target.earlier_end_seconds <= end_seconds
    )


def waveform_window(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    point_count: int,
) -> tuple[int, tuple[PreviewWaveformPoint, ...]]:
    if path.suffix.lower() != ".wav":
        return ffmpeg_waveform_window(
            path=path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            point_count=point_count,
        )
    try:
        with wave.open(str(path), "rb") as wave_reader:
            sample_rate = wave_reader.getframerate()
            sample_width = wave_reader.getsampwidth()
            channel_count = wave_reader.getnchannels()
            start_frame = round(start_seconds * sample_rate)
            frame_count = max(0, round((end_seconds - start_seconds) * sample_rate))
            wave_reader.setpos(min(start_frame, wave_reader.getnframes()))
            fragment = wave_reader.readframes(frame_count)
    except (OSError, wave.Error) as error:
        raise ValueError(f"Could not read WAV file {path}: {error}") from error
    if not fragment:
        return sample_rate, ()
    samples = mono_samples(
        fragment=fragment,
        sample_width=sample_width,
        channel_count=channel_count,
    )
    frames_per_point = max(1, math.ceil(len(samples) / point_count))
    maximum_amplitude = float(1 << (sample_width * 8 - 1))
    points: list[PreviewWaveformPoint] = []
    for point_start in range(0, len(samples), frames_per_point):
        point_samples = samples[point_start : point_start + frames_per_point]
        points.append(
            PreviewWaveformPoint(
                minimum_amplitude=max(-1.0, float(np.min(point_samples)) / maximum_amplitude),
                maximum_amplitude=min(1.0, float(np.max(point_samples)) / maximum_amplitude),
            )
        )
    return sample_rate, tuple(points)


def ffmpeg_waveform_window(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    point_count: int,
) -> tuple[int, tuple[PreviewWaveformPoint, ...]]:
    waveform_sample_rate = 100
    completed = subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_seconds),
            "-i",
            str(path),
            "-t",
            str(max(0.0, end_seconds - start_seconds)),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(waveform_sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ),
        check=True,
        capture_output=True,
    )
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    if len(samples) == 0:
        return waveform_sample_rate, ()
    frames_per_point = max(1, math.ceil(len(samples) / point_count))
    points = tuple(
        PreviewWaveformPoint(
            minimum_amplitude=max(-1.0, float(np.min(point_samples))),
            maximum_amplitude=min(1.0, float(np.max(point_samples))),
        )
        for point_start in range(0, len(samples), frames_per_point)
        if len(point_samples := samples[point_start : point_start + frames_per_point])
    )
    return waveform_sample_rate, points
