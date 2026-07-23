from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.db.models import TrackSide
from app.local.synchronization_review.calibration import (
    ReviewedAlignment,
    is_unresolved_alignment,
    reviewed_alignment,
)
from app.local.synchronization_review.models import (
    AlignmentEstimateOrigin,
    OffsetPattern,
    SynchronizationCandidate,
    SynchronizationCandidateListResponse,
    SynchronizationEvidence,
    SynchronizationEvidenceSource,
    SynchronizationWindowEstimate,
)
from app.local.synchronization_review.optimized_alignment import (
    ESTIMATOR_VERSION as OPTIMIZED_ESTIMATOR_VERSION,
)
from app.local.synchronization_review.repository import (
    StoredConversationAnnotation,
    SynchronizationReviewRepository,
)
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.audio.gain import (
    default_gain_for_active_rms as default_gain_for_active_rms,
)
from app.shared.audio.gain import (
    estimated_active_rms_dbfs as estimated_active_rms_dbfs,
)
from app.shared.audio.gain import (
    speech_only_gain_normalization as speech_only_gain_normalization,
)
from app.shared.audio.gain import (
    track_gain_normalization,
)
from app.shared.quality import METRIC_VERSION, AnnotationSpan

FRAME_DURATION_SECONDS = 0.1
MERGE_WORD_GAP_SECONDS = 0.5
MAXIMUM_LAG_SECONDS = 12.0
ANALYSIS_MARGIN_SECONDS = 12.0
WINDOW_DURATION_SECONDS = 180.0
MINIMUM_WINDOW_DURATION_SECONDS = 120.0
LONG_OVERLAP_SECONDS = 0.8
LONG_SILENCE_SECONDS = 1.2
MAXIMUM_CYCLE_GAP_SECONDS = 5.0
MINIMUM_MEANINGFUL_LAG_SECONDS = 0.4
MINIMUM_MEANINGFUL_IMPROVEMENT = 0.02
MINIMUM_JOINT_REDUCTION = 0.0
RECOMMENDED_SHIFT_RESOLUTION_SECONDS = 0.1
ESTIMATOR_VERSION = OPTIMIZED_ESTIMATOR_VERSION


class TimelineState(StrEnum):
    SPEAKER1 = "speaker1"
    SPEAKER2 = "speaker2"
    BOTH = "both"
    NEITHER = "neither"


@dataclass(frozen=True)
class SpeechSpan:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class StateRun:
    state: TimelineState
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class TimelineMetrics:
    overlap_ratio: float
    silence_ratio: float
    overlap_silence_cycle_count: int
    best_lag_seconds: float
    bad_state_improvement: float
    overlap_reduction: float
    silence_reduction: float

    @property
    def joint_reduction(self) -> float:
        return max(0.0, min(self.overlap_reduction, self.silence_reduction))


@dataclass(frozen=True)
class SourceMetrics:
    source: SynchronizationEvidenceSource
    metrics: TimelineMetrics


def synchronization_candidates(
    repository: SynchronizationReviewRepository,
) -> SynchronizationCandidateListResponse:
    total_session_count = repository.count_local_samples()
    stored_alignments = repository.load_reviewed_alignments()
    stored_predictions = repository.load_unreviewed_predictions(estimator_version=ESTIMATOR_VERSION)
    prediction_by_sample_id = {
        prediction.sample_id: prediction for prediction in stored_predictions
    }
    annotations = repository.load_annotations(metric_version=METRIC_VERSION)
    transcripts = FullRecordingAsrRepository(
        database_url=repository.database_url
    ).list_latest_transcripts(
        model_ids=(AsrModelId.PARAKEET_TDT, AsrModelId.CANARY),
    )
    transcript_by_key = {
        (transcript.sample_external_id, transcript.model_id, transcript.side): transcript
        for transcript in transcripts
        if transcript.error is None
    }
    candidates: list[SynchronizationCandidate] = []
    analyzed_session_count = 0
    offset_candidate_count = 0

    for stored_annotation in annotations:
        required_transcripts = tuple(
            transcript_by_key.get((stored_annotation.external_id, model_id, side))
            for model_id in (AsrModelId.PARAKEET_TDT, AsrModelId.CANARY)
            for side in TrackSide
        )
        if any(transcript is None for transcript in required_transcripts):
            continue
        (
            parakeet_speaker1,
            parakeet_speaker2,
            canary_speaker1,
            canary_speaker2,
        ) = required_transcripts
        assert parakeet_speaker1 is not None
        assert parakeet_speaker2 is not None
        assert canary_speaker1 is not None
        assert canary_speaker2 is not None
        analyzed_session_count += 1
        candidate = _candidate(
            stored_annotation=stored_annotation,
            parakeet_speaker1=parakeet_speaker1,
            parakeet_speaker2=parakeet_speaker2,
            canary_speaker1=canary_speaker1,
            canary_speaker2=canary_speaker2,
            stored_alignments=stored_alignments,
        )
        stored_prediction = prediction_by_sample_id.get(candidate.sample_id)
        if (
            candidate.alignment_estimate_origin is AlignmentEstimateOrigin.PREDICTED
            and stored_prediction is not None
        ):
            assert stored_prediction.external_id == candidate.external_id
            candidate = _overlay_stored_prediction(
                current_candidate=candidate,
                stored_prediction=stored_prediction,
            )
        if candidate.is_offset_candidate:
            offset_candidate_count += 1
        candidates.append(candidate)

    return SynchronizationCandidateListResponse(
        candidates=tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.likelihood_score,
                    abs(candidate.estimated_b_shift_seconds),
                ),
                reverse=True,
            )
        ),
        analyzed_session_count=analyzed_session_count,
        offset_candidate_count=offset_candidate_count,
        excluded_session_count=total_session_count - analyzed_session_count,
    )


def _overlay_stored_prediction(
    current_candidate: SynchronizationCandidate,
    stored_prediction: SynchronizationCandidate,
) -> SynchronizationCandidate:
    current_evidence = tuple(
        evidence
        for evidence in current_candidate.evidence
        if evidence.source is not SynchronizationEvidenceSource.AUDIO_ACTIVITY
    )
    stored_audio_evidence = tuple(
        evidence
        for evidence in stored_prediction.evidence
        if evidence.source is SynchronizationEvidenceSource.AUDIO_ACTIVITY
    )
    current_windows = tuple(
        window
        for window in current_candidate.window_estimates
        if window.source is not SynchronizationEvidenceSource.AUDIO_ACTIVITY
    )
    stored_audio_windows = tuple(
        window
        for window in stored_prediction.window_estimates
        if window.source is SynchronizationEvidenceSource.AUDIO_ACTIVITY
    )
    return current_candidate.model_copy(
        update={
            "is_offset_candidate": stored_prediction.is_offset_candidate,
            "estimated_b_shift_seconds": stored_prediction.estimated_b_shift_seconds,
            "full_recording_estimated_b_shift_seconds": (
                stored_prediction.full_recording_estimated_b_shift_seconds
            ),
            "offset_confidence_score": stored_prediction.offset_confidence_score,
            "offset_pattern": stored_prediction.offset_pattern,
            "static_offset_valid": stored_prediction.static_offset_valid,
            "drift_warning": stored_prediction.drift_warning,
            "evidence": (*current_evidence, *stored_audio_evidence),
            "window_estimates": (*current_windows, *stored_audio_windows),
        }
    )


def _candidate(
    stored_annotation: StoredConversationAnnotation,
    parakeet_speaker1: FullRecordingAsrTranscriptRecord,
    parakeet_speaker2: FullRecordingAsrTranscriptRecord,
    canary_speaker1: FullRecordingAsrTranscriptRecord,
    canary_speaker2: FullRecordingAsrTranscriptRecord,
    stored_alignments: tuple[ReviewedAlignment, ...],
) -> SynchronizationCandidate:
    annotation = stored_annotation.annotation
    annotation_duration_seconds = annotation.analyzed_duration_seconds
    annotation_speaker1_spans = annotation_spans(annotation.speaker1.speech_segments)
    annotation_speaker2_spans = annotation_spans(annotation.speaker2.speech_segments)
    parakeet_duration_seconds = min(
        parakeet_speaker1.prepared_duration_seconds,
        parakeet_speaker2.prepared_duration_seconds,
    )
    canary_duration_seconds = min(
        canary_speaker1.prepared_duration_seconds,
        canary_speaker2.prepared_duration_seconds,
    )
    parakeet_speaker1_spans = timed_word_spans(parakeet_speaker1.words)
    parakeet_speaker2_spans = timed_word_spans(parakeet_speaker2.words)
    canary_speaker1_spans = timed_word_spans(canary_speaker1.words)
    canary_speaker2_spans = timed_word_spans(canary_speaker2.words)
    source_metrics = (
        SourceMetrics(
            source=SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
            metrics=timeline_metrics(
                speaker1_spans=annotation_speaker1_spans,
                speaker2_spans=annotation_speaker2_spans,
                duration_seconds=annotation_duration_seconds,
            ),
        ),
        SourceMetrics(
            source=SynchronizationEvidenceSource.PARAKEET,
            metrics=timeline_metrics(
                speaker1_spans=parakeet_speaker1_spans,
                speaker2_spans=parakeet_speaker2_spans,
                duration_seconds=parakeet_duration_seconds,
            ),
        ),
        SourceMetrics(
            source=SynchronizationEvidenceSource.CANARY,
            metrics=timeline_metrics(
                speaker1_spans=canary_speaker1_spans,
                speaker2_spans=canary_speaker2_spans,
                duration_seconds=canary_duration_seconds,
            ),
        ),
    )
    meaningful_sources = tuple(
        source_metric
        for source_metric in source_metrics
        if meaningful_metrics(source_metric.metrics)
    )
    window_estimates = (
        *timeline_window_estimates(
            source=SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
            speaker1_spans=annotation_speaker1_spans,
            speaker2_spans=annotation_speaker2_spans,
            duration_seconds=annotation_duration_seconds,
        ),
        *timeline_window_estimates(
            source=SynchronizationEvidenceSource.PARAKEET,
            speaker1_spans=parakeet_speaker1_spans,
            speaker2_spans=parakeet_speaker2_spans,
            duration_seconds=parakeet_duration_seconds,
        ),
        *timeline_window_estimates(
            source=SynchronizationEvidenceSource.CANARY,
            speaker1_spans=canary_speaker1_spans,
            speaker2_spans=canary_speaker2_spans,
            duration_seconds=canary_duration_seconds,
        ),
    )
    meaningful_windows = tuple(estimate for estimate in window_estimates if estimate.meaningful)
    is_offset_candidate = len(meaningful_sources) >= 2

    transcript_sources = tuple(
        source_metric
        for source_metric in source_metrics
        if source_metric.source is not SynchronizationEvidenceSource.CONVERSATION_ANNOTATION
    )
    meaningful_transcript_sources = tuple(
        source_metric
        for source_metric in transcript_sources
        if meaningful_metrics(source_metric.metrics)
    )
    estimation_sources = meaningful_transcript_sources
    source_agreement = lag_agreement(meaningful_sources=estimation_sources)
    offset_pattern = classify_offset_pattern(
        meaningful_sources=meaningful_sources,
        meaningful_windows=meaningful_windows,
    )
    full_recording_estimated_shift = (
        robust_shift_estimate(meaningful_sources=estimation_sources) if estimation_sources else 0.0
    )
    predicted_shift = recommended_shift_estimate(
        offset_pattern=offset_pattern,
        full_recording_estimated_shift=full_recording_estimated_shift,
        window_estimates=window_estimates,
    )
    reviewed = reviewed_alignment(
        external_id=stored_annotation.external_id,
        stored_alignments=stored_alignments,
    )
    unresolved = reviewed is None and is_unresolved_alignment(
        external_id=stored_annotation.external_id
    )
    estimated_shift = reviewed.speaker2_shift_seconds if reviewed is not None else predicted_shift
    estimate_origin = (
        AlignmentEstimateOrigin.REVIEWED
        if reviewed is not None
        else (
            AlignmentEstimateOrigin.UNRESOLVED if unresolved else AlignmentEstimateOrigin.PREDICTED
        )
    )
    offset_confidence = offset_confidence_score(
        estimate_origin=estimate_origin,
        predicted_shift=predicted_shift,
        full_recording_estimated_shift=full_recording_estimated_shift,
        source_metrics=source_metrics,
        meaningful_windows=meaningful_windows,
    )
    likelihood_score = candidate_likelihood(
        source_metrics=source_metrics,
        source_agreement=source_agreement,
        offset_pattern=offset_pattern,
    )
    return SynchronizationCandidate(
        sample_id=stored_annotation.sample_id,
        external_id=stored_annotation.external_id,
        likelihood_score=likelihood_score,
        is_offset_candidate=is_offset_candidate,
        estimated_b_shift_seconds=estimated_shift,
        full_recording_estimated_b_shift_seconds=full_recording_estimated_shift,
        alignment_estimate_origin=estimate_origin,
        offset_confidence_score=offset_confidence,
        offset_pattern=offset_pattern,
        static_offset_valid=offset_pattern is not OffsetPattern.VARIABLE,
        drift_warning=(
            "Window estimates vary within at least one evidence source; review for drift "
            "or structural mismatch."
            if offset_pattern is OffsetPattern.VARIABLE
            else None
        ),
        duration_mismatch_seconds=(
            stored_annotation.audio_quality.duration_gap_seconds
            if stored_annotation.audio_quality is not None
            else None
        ),
        source_agreement=source_agreement,
        evidence=tuple(
            SynchronizationEvidence(
                source=source_metric.source,
                estimated_b_shift_seconds=source_metric.metrics.best_lag_seconds,
                bad_state_improvement=source_metric.metrics.bad_state_improvement,
                overlap_reduction=source_metric.metrics.overlap_reduction,
                silence_reduction=source_metric.metrics.silence_reduction,
            )
            for source_metric in source_metrics
        ),
        window_estimates=window_estimates,
        overlap_silence_cycle_count=round(
            sum(
                source_metric.metrics.overlap_silence_cycle_count
                for source_metric in source_metrics
            )
            / len(source_metrics)
        ),
        speaker1_gain=track_gain_normalization(
            stored_annotation.audio_quality.speaker1
            if stored_annotation.audio_quality is not None
            else None
        ),
        speaker2_gain=track_gain_normalization(
            stored_annotation.audio_quality.speaker2
            if stored_annotation.audio_quality is not None
            else None
        ),
    )


def timed_word_spans(words: tuple[TimestampedWord, ...]) -> tuple[SpeechSpan, ...]:
    timed_words = tuple(
        word
        for word in words
        if word.start_seconds is not None
        and word.end_seconds is not None
        and word.end_seconds > word.start_seconds
    )
    if not timed_words:
        return ()
    first_word = timed_words[0]
    assert first_word.start_seconds is not None
    assert first_word.end_seconds is not None
    current_start = first_word.start_seconds
    current_end = first_word.end_seconds
    spans: list[SpeechSpan] = []
    for word in timed_words[1:]:
        assert word.start_seconds is not None
        assert word.end_seconds is not None
        if word.start_seconds - current_end <= MERGE_WORD_GAP_SECONDS:
            current_end = max(current_end, word.end_seconds)
            continue
        spans.append(SpeechSpan(start_seconds=current_start, end_seconds=current_end))
        current_start = word.start_seconds
        current_end = word.end_seconds
    spans.append(SpeechSpan(start_seconds=current_start, end_seconds=current_end))
    return tuple(spans)


def annotation_spans(spans: tuple[AnnotationSpan, ...]) -> tuple[SpeechSpan, ...]:
    return tuple(
        SpeechSpan(start_seconds=span.start_seconds, end_seconds=span.end_seconds) for span in spans
    )


def activity_mask(
    spans: tuple[SpeechSpan, ...],
    duration_seconds: float,
) -> NDArray[np.bool_]:
    frame_count = math.ceil(duration_seconds / FRAME_DURATION_SECONDS)
    mask = np.zeros(frame_count, dtype=bool)
    for span in spans:
        start_index = max(0, math.floor(span.start_seconds / FRAME_DURATION_SECONDS))
        end_index = min(frame_count, math.ceil(span.end_seconds / FRAME_DURATION_SECONDS))
        mask[start_index:end_index] = True
    return mask


def shifted_mask(mask: NDArray[np.bool_], lag_frames: int) -> NDArray[np.bool_]:
    shifted = np.zeros_like(mask)
    if lag_frames > 0:
        shifted[lag_frames:] = mask[:-lag_frames]
    elif lag_frames < 0:
        shifted[:lag_frames] = mask[-lag_frames:]
    else:
        shifted[:] = mask
    return shifted


def state_runs(
    speaker1_mask: NDArray[np.bool_],
    speaker2_mask: NDArray[np.bool_],
) -> tuple[StateRun, ...]:
    if len(speaker1_mask) == 0:
        return ()
    states = np.where(
        speaker1_mask & speaker2_mask,
        2,
        np.where(speaker1_mask, 0, np.where(speaker2_mask, 1, 3)),
    )
    state_by_index = (
        TimelineState.SPEAKER1,
        TimelineState.SPEAKER2,
        TimelineState.BOTH,
        TimelineState.NEITHER,
    )
    runs: list[StateRun] = []
    start_index = 0
    for index in range(1, len(states) + 1):
        if index < len(states) and states[index] == states[start_index]:
            continue
        runs.append(
            StateRun(
                state=state_by_index[int(states[start_index])],
                start_seconds=start_index * FRAME_DURATION_SECONDS,
                end_seconds=index * FRAME_DURATION_SECONDS,
            )
        )
        start_index = index
    return tuple(runs)


def overlap_silence_cycle_count(runs: tuple[StateRun, ...]) -> int:
    significant_runs = tuple(
        run
        for run in runs
        if (run.state == TimelineState.BOTH and run.duration_seconds >= LONG_OVERLAP_SECONDS)
        or (run.state == TimelineState.NEITHER and run.duration_seconds >= LONG_SILENCE_SECONDS)
    )
    return sum(
        1
        for earlier, later in zip(significant_runs, significant_runs[1:], strict=False)
        if earlier.state != later.state
        and later.start_seconds - earlier.end_seconds <= MAXIMUM_CYCLE_GAP_SECONDS
    )


def overlap_and_silence_ratios(
    speaker1_mask: NDArray[np.bool_],
    speaker2_mask: NDArray[np.bool_],
    evaluation_slice: slice,
) -> tuple[float, float]:
    speaker1 = speaker1_mask[evaluation_slice]
    speaker2 = speaker2_mask[evaluation_slice]
    return (
        float(np.mean(speaker1 & speaker2)),
        float(np.mean(~speaker1 & ~speaker2)),
    )


def best_lag_and_improvement(
    speaker1_mask: NDArray[np.bool_],
    speaker2_mask: NDArray[np.bool_],
) -> tuple[float, float, float, float]:
    maximum_lag_frames = round(MAXIMUM_LAG_SECONDS / FRAME_DURATION_SECONDS)
    margin_frames = round(ANALYSIS_MARGIN_SECONDS / FRAME_DURATION_SECONDS)
    if len(speaker1_mask) <= 2 * margin_frames:
        return 0.0, 0.0, 0.0, 0.0
    evaluation_slice = slice(margin_frames, len(speaker1_mask) - margin_frames)
    zero_overlap_ratio, zero_silence_ratio = overlap_and_silence_ratios(
        speaker1_mask=speaker1_mask,
        speaker2_mask=speaker2_mask,
        evaluation_slice=evaluation_slice,
    )
    lag_ratios = tuple(
        (
            lag_frames,
            overlap_and_silence_ratios(
                speaker1_mask=speaker1_mask,
                speaker2_mask=shifted_mask(speaker2_mask, lag_frames),
                evaluation_slice=evaluation_slice,
            ),
        )
        for lag_frames in range(-maximum_lag_frames, maximum_lag_frames + 1)
    )
    best_lag_frames, best_ratios = min(
        lag_ratios,
        key=lambda item: item[1][0] + item[1][1],
    )
    best_overlap_ratio, best_silence_ratio = best_ratios
    return (
        best_lag_frames * FRAME_DURATION_SECONDS,
        max(
            0.0,
            zero_overlap_ratio + zero_silence_ratio - best_overlap_ratio - best_silence_ratio,
        ),
        zero_overlap_ratio - best_overlap_ratio,
        zero_silence_ratio - best_silence_ratio,
    )


def timeline_metrics(
    speaker1_spans: tuple[SpeechSpan, ...],
    speaker2_spans: tuple[SpeechSpan, ...],
    duration_seconds: float,
) -> TimelineMetrics:
    speaker1_mask = activity_mask(spans=speaker1_spans, duration_seconds=duration_seconds)
    speaker2_mask = activity_mask(spans=speaker2_spans, duration_seconds=duration_seconds)
    runs = state_runs(speaker1_mask=speaker1_mask, speaker2_mask=speaker2_mask)
    best_lag_seconds, improvement, overlap_reduction, silence_reduction = best_lag_and_improvement(
        speaker1_mask=speaker1_mask,
        speaker2_mask=speaker2_mask,
    )
    return TimelineMetrics(
        overlap_ratio=float(np.mean(speaker1_mask & speaker2_mask)),
        silence_ratio=float(np.mean(~speaker1_mask & ~speaker2_mask)),
        overlap_silence_cycle_count=overlap_silence_cycle_count(runs=runs),
        best_lag_seconds=best_lag_seconds,
        bad_state_improvement=improvement,
        overlap_reduction=overlap_reduction,
        silence_reduction=silence_reduction,
    )


def timeline_window_estimates(
    source: SynchronizationEvidenceSource,
    speaker1_spans: tuple[SpeechSpan, ...],
    speaker2_spans: tuple[SpeechSpan, ...],
    duration_seconds: float,
) -> tuple[SynchronizationWindowEstimate, ...]:
    speaker1_mask = activity_mask(
        spans=speaker1_spans,
        duration_seconds=duration_seconds,
    )
    speaker2_mask = activity_mask(
        spans=speaker2_spans,
        duration_seconds=duration_seconds,
    )
    window_frame_count = round(WINDOW_DURATION_SECONDS / FRAME_DURATION_SECONDS)
    estimates: list[SynchronizationWindowEstimate] = []
    for start_index in range(0, len(speaker1_mask), window_frame_count):
        end_index = min(len(speaker1_mask), start_index + window_frame_count)
        if end_index - start_index < round(
            MINIMUM_WINDOW_DURATION_SECONDS / FRAME_DURATION_SECONDS
        ):
            continue
        lag_seconds, improvement, overlap_reduction, silence_reduction = best_lag_and_improvement(
            speaker1_mask=speaker1_mask[start_index:end_index],
            speaker2_mask=speaker2_mask[start_index:end_index],
        )
        metrics = TimelineMetrics(
            overlap_ratio=0.0,
            silence_ratio=0.0,
            overlap_silence_cycle_count=0,
            best_lag_seconds=lag_seconds,
            bad_state_improvement=improvement,
            overlap_reduction=overlap_reduction,
            silence_reduction=silence_reduction,
        )
        estimates.append(
            SynchronizationWindowEstimate(
                source=source,
                start_seconds=start_index * FRAME_DURATION_SECONDS,
                end_seconds=end_index * FRAME_DURATION_SECONDS,
                estimated_b_shift_seconds=lag_seconds,
                bad_state_improvement=improvement,
                overlap_reduction=overlap_reduction,
                silence_reduction=silence_reduction,
                meaningful=reliable_window_metrics(metrics),
            )
        )
    return tuple(estimates)


def meaningful_metrics(metrics: TimelineMetrics) -> bool:
    return (
        abs(metrics.best_lag_seconds) >= MINIMUM_MEANINGFUL_LAG_SECONDS
        and metrics.bad_state_improvement >= MINIMUM_MEANINGFUL_IMPROVEMENT
        and metrics.joint_reduction >= MINIMUM_JOINT_REDUCTION
    )


def reliable_window_metrics(metrics: TimelineMetrics) -> bool:
    return (
        metrics.bad_state_improvement >= MINIMUM_MEANINGFUL_IMPROVEMENT
        and metrics.joint_reduction >= MINIMUM_JOINT_REDUCTION
    )


def robust_shift_estimate(meaningful_sources: tuple[SourceMetrics, ...]) -> float:
    ordered = sorted(
        (
            source_metric.metrics.best_lag_seconds,
            max(
                0.001,
                source_metric.metrics.bad_state_improvement
                + 2.0 * source_metric.metrics.joint_reduction,
            ),
        )
        for source_metric in meaningful_sources
    )
    total_weight = sum(weight for _, weight in ordered)
    midpoint = total_weight / 2.0
    accumulated_weight = 0.0
    for lag_seconds, weight in ordered:
        accumulated_weight += weight
        if accumulated_weight >= midpoint:
            return lag_seconds
    return ordered[-1][0]


def recommended_shift_estimate(
    offset_pattern: OffsetPattern,
    full_recording_estimated_shift: float,
    window_estimates: tuple[SynchronizationWindowEstimate, ...],
) -> float:
    meaningful_windows = tuple(window for window in window_estimates if window.meaningful)
    review_anchor = (
        meaningful_windows[0]
        if meaningful_windows
        else (window_estimates[0] if window_estimates else None)
    )
    raw_estimate = (
        review_anchor.estimated_b_shift_seconds
        if offset_pattern is OffsetPattern.VARIABLE and review_anchor is not None
        else full_recording_estimated_shift
    )
    quantized_steps = round(raw_estimate / RECOMMENDED_SHIFT_RESOLUTION_SECONDS)
    return round(quantized_steps * RECOMMENDED_SHIFT_RESOLUTION_SECONDS, 1)


def lag_agreement(meaningful_sources: tuple[SourceMetrics, ...]) -> bool:
    if len(meaningful_sources) < 2:
        return False
    return (
        _lag_spread(
            tuple(source_metric.metrics.best_lag_seconds for source_metric in meaningful_sources)
        )
        <= 0.8
    )


def classify_offset_pattern(
    meaningful_sources: tuple[SourceMetrics, ...],
    meaningful_windows: tuple[SynchronizationWindowEstimate, ...],
) -> OffsetPattern:
    if _variable_window_source_count(windows=meaningful_windows) >= 2:
        return OffsetPattern.VARIABLE
    if lag_agreement(meaningful_sources):
        return OffsetPattern.CONSTANT
    return OffsetPattern.UNCERTAIN


def candidate_likelihood(
    source_metrics: tuple[SourceMetrics, ...],
    source_agreement: bool,
    offset_pattern: OffsetPattern,
) -> float:
    metrics = tuple(source_metric.metrics for source_metric in source_metrics)
    mean_overlap = sum(metric.overlap_ratio for metric in metrics) / len(metrics)
    mean_improvement = sum(metric.bad_state_improvement for metric in metrics) / len(metrics)
    mean_joint_reduction = sum(metric.joint_reduction for metric in metrics) / len(metrics)
    mean_cycles = sum(metric.overlap_silence_cycle_count for metric in metrics) / len(metrics)
    raw_score = (
        4.0 * mean_improvement
        + 8.0 * mean_joint_reduction
        + 1.5 * mean_overlap
        + 0.025 * mean_cycles
        + (0.15 if source_agreement else 0.0)
        + (0.10 if offset_pattern is OffsetPattern.VARIABLE else 0.0)
    )
    return min(1.0, 1.0 - math.exp(-raw_score / 1.2))


def offset_confidence_score(
    estimate_origin: AlignmentEstimateOrigin,
    predicted_shift: float,
    full_recording_estimated_shift: float,
    source_metrics: tuple[SourceMetrics, ...],
    meaningful_windows: tuple[SynchronizationWindowEstimate, ...],
) -> float:
    if estimate_origin is AlignmentEstimateOrigin.REVIEWED:
        return 1.0
    if estimate_origin is AlignmentEstimateOrigin.UNRESOLVED:
        return 0.0
    metrics = tuple(source_metric.metrics for source_metric in source_metrics)
    evidence_strength = sum(
        0.5 * min(1.0, metric.bad_state_improvement / 0.20)
        + 0.5 * min(1.0, metric.joint_reduction / 0.10)
        for metric in metrics
    ) / len(metrics)
    source_spread = _lag_spread(tuple(metric.best_lag_seconds for metric in metrics))
    source_agreement = math.exp(-source_spread / 1.5)
    recommendation_consistency = math.exp(
        -abs(predicted_shift - full_recording_estimated_shift) / 2.0
    )
    window_spread = _maximum_within_source_window_spread(windows=meaningful_windows)
    temporal_consistency = math.exp(-window_spread / 4.0)
    confidence = (
        evidence_strength
        * (0.35 + 0.65 * source_agreement)
        * (0.50 + 0.50 * recommendation_consistency)
        * (0.60 + 0.40 * temporal_consistency)
    )
    return min(1.0, max(0.0, confidence))


def _lag_spread(lags: tuple[float, ...]) -> float:
    if len(lags) < 2:
        return 0.0
    return max(lags) - min(lags)


def _maximum_within_source_window_spread(
    windows: tuple[SynchronizationWindowEstimate, ...],
) -> float:
    windows_by_source = {
        source: tuple(window for window in windows if window.source is source)
        for source in SynchronizationEvidenceSource
    }
    return max(
        (
            _lag_spread(tuple(window.estimated_b_shift_seconds for window in source_windows))
            for source_windows in windows_by_source.values()
        ),
        default=0.0,
    )


def _variable_window_source_count(
    windows: tuple[SynchronizationWindowEstimate, ...],
) -> int:
    return sum(
        _lag_spread(
            tuple(window.estimated_b_shift_seconds for window in windows if window.source is source)
        )
        >= 1.5
        for source in SynchronizationEvidenceSource
    )
