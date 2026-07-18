from __future__ import annotations

from app.local.synchronization_review.evaluation import (
    EvidenceScope,
    OffsetEvidence,
    OffsetEvidenceRecord,
    OffsetWindowEvidence,
)
from app.local.synchronization_review.models import SynchronizationEvidenceSource
from app.local.synchronization_review.repository import (
    StoredConversationAnnotation,
    TranscriptPair,
)
from app.local.synchronization_review.service import (
    SpeechSpan,
    annotation_spans,
    timed_word_spans,
    timeline_metrics,
)
from app.shared.asr import AsrModelId

INITIAL_DURATION_SECONDS = 180.0
INITIAL_WINDOW_SECONDS = 60.0


def initial_recording_evidence_records(
    transcript_pairs: tuple[TranscriptPair, ...],
    annotations: tuple[StoredConversationAnnotation, ...],
) -> tuple[OffsetEvidenceRecord, ...]:
    pair_by_key = {(pair.external_id, pair.model_id): pair for pair in transcript_pairs}
    records: list[OffsetEvidenceRecord] = []
    for stored_annotation in annotations:
        parakeet = pair_by_key.get((stored_annotation.external_id, AsrModelId.PARAKEET_TDT))
        canary = pair_by_key.get((stored_annotation.external_id, AsrModelId.CANARY))
        if parakeet is None or canary is None:
            continue
        duration_seconds = min(
            stored_annotation.annotation.analyzed_duration_seconds,
            INITIAL_DURATION_SECONDS,
        )
        annotation_speaker1 = annotation_spans(
            stored_annotation.annotation.speaker1.speech_segments
        )
        annotation_speaker2 = annotation_spans(
            stored_annotation.annotation.speaker2.speech_segments
        )
        records.append(
            OffsetEvidenceRecord(
                external_id=stored_annotation.external_id,
                scope=EvidenceScope.INITIAL_180_SECONDS,
                sources=(
                    _source_evidence(
                        source=SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
                        speaker1_spans=annotation_speaker1,
                        speaker2_spans=annotation_speaker2,
                        duration_seconds=duration_seconds,
                    ),
                    _source_evidence(
                        source=SynchronizationEvidenceSource.PARAKEET,
                        speaker1_spans=timed_word_spans(parakeet.speaker1_words),
                        speaker2_spans=timed_word_spans(parakeet.speaker2_words),
                        duration_seconds=duration_seconds,
                    ),
                    _source_evidence(
                        source=SynchronizationEvidenceSource.CANARY,
                        speaker1_spans=timed_word_spans(canary.speaker1_words),
                        speaker2_spans=timed_word_spans(canary.speaker2_words),
                        duration_seconds=duration_seconds,
                    ),
                ),
                windows=_annotation_windows(
                    speaker1_spans=annotation_speaker1,
                    speaker2_spans=annotation_speaker2,
                    duration_seconds=duration_seconds,
                ),
            )
        )
    return tuple(records)


def _source_evidence(
    source: SynchronizationEvidenceSource,
    speaker1_spans: tuple[SpeechSpan, ...],
    speaker2_spans: tuple[SpeechSpan, ...],
    duration_seconds: float,
) -> OffsetEvidence:
    metrics = timeline_metrics(
        speaker1_spans=speaker1_spans,
        speaker2_spans=speaker2_spans,
        duration_seconds=duration_seconds,
    )
    return OffsetEvidence(
        source=source,
        estimated_shift_seconds=metrics.best_lag_seconds,
        bad_state_improvement=metrics.bad_state_improvement,
        overlap_reduction=metrics.overlap_reduction,
        silence_reduction=metrics.silence_reduction,
    )


def _annotation_windows(
    speaker1_spans: tuple[SpeechSpan, ...],
    speaker2_spans: tuple[SpeechSpan, ...],
    duration_seconds: float,
) -> tuple[OffsetWindowEvidence, ...]:
    windows: list[OffsetWindowEvidence] = []
    start_seconds = 0.0
    while start_seconds < duration_seconds:
        end_seconds = min(duration_seconds, start_seconds + INITIAL_WINDOW_SECONDS)
        if end_seconds - start_seconds < INITIAL_WINDOW_SECONDS:
            break
        metrics = timeline_metrics(
            speaker1_spans=_window_spans(
                spans=speaker1_spans,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            ),
            speaker2_spans=_window_spans(
                spans=speaker2_spans,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            ),
            duration_seconds=INITIAL_WINDOW_SECONDS,
        )
        windows.append(
            OffsetWindowEvidence(
                source=SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                estimated_shift_seconds=metrics.best_lag_seconds,
                bad_state_improvement=metrics.bad_state_improvement,
                overlap_reduction=metrics.overlap_reduction,
                silence_reduction=metrics.silence_reduction,
            )
        )
        start_seconds += INITIAL_WINDOW_SECONDS
    return tuple(windows)


def _window_spans(
    spans: tuple[SpeechSpan, ...],
    start_seconds: float,
    end_seconds: float,
) -> tuple[SpeechSpan, ...]:
    return tuple(
        SpeechSpan(
            start_seconds=max(span.start_seconds, start_seconds) - start_seconds,
            end_seconds=min(span.end_seconds, end_seconds) - start_seconds,
        )
        for span in spans
        if span.end_seconds > start_seconds and span.start_seconds < end_seconds
    )
