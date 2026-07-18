from __future__ import annotations

from dataclasses import dataclass

from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.db.models import TrackSide
from app.local.synchronization_review.evaluation import (
    EvidenceScope,
    OffsetEvidence,
    OffsetEvidenceRecord,
    OffsetWindowEvidence,
)
from app.local.synchronization_review.models import SynchronizationEvidenceSource
from app.local.synchronization_review.repository import StoredConversationAnnotation
from app.local.synchronization_review.service import (
    SpeechSpan,
    annotation_spans,
    timed_word_spans,
    timeline_metrics,
)
from app.shared.asr import AsrModelId

FULL_EVIDENCE_WINDOW_SECONDS = 180.0
MINIMUM_FULL_EVIDENCE_WINDOW_SECONDS = 120.0
SUPPORTED_FULL_EVIDENCE_MODELS = (AsrModelId.PARAKEET_TDT, AsrModelId.CANARY)


@dataclass(frozen=True)
class FullTranscriptPair:
    external_id: str
    model_id: AsrModelId
    speaker1: FullRecordingAsrTranscriptRecord
    speaker2: FullRecordingAsrTranscriptRecord


def full_recording_evidence_records(
    transcripts: tuple[FullRecordingAsrTranscriptRecord, ...],
    model_ids: tuple[AsrModelId, ...] = SUPPORTED_FULL_EVIDENCE_MODELS,
    annotations: tuple[StoredConversationAnnotation, ...] = (),
) -> tuple[OffsetEvidenceRecord, ...]:
    _validate_model_ids(model_ids=model_ids)
    selected_transcripts = _transcripts_for_annotations(
        transcripts=transcripts,
        annotations=annotations,
    )
    pairs = _complete_transcript_pairs(
        transcripts=selected_transcripts,
        model_ids=model_ids,
    )
    pairs_by_external_id: dict[str, list[FullTranscriptPair]] = {}
    for pair in pairs:
        pairs_by_external_id.setdefault(pair.external_id, []).append(pair)
    annotation_by_external_id = {annotation.external_id: annotation for annotation in annotations}
    missing_annotations = tuple(
        external_id
        for external_id in pairs_by_external_id
        if annotations and external_id not in annotation_by_external_id
    )
    if missing_annotations:
        raise ValueError(
            "Missing full-recording conversation annotations: "
            + ", ".join(missing_annotations[:10])
        )
    return tuple(
        _full_recording_evidence_record(
            external_id=external_id,
            pairs=tuple(sorted(sample_pairs, key=lambda pair: pair.model_id.value)),
            annotation=annotation_by_external_id.get(external_id),
        )
        for external_id, sample_pairs in sorted(pairs_by_external_id.items())
    )


def _transcripts_for_annotations(
    transcripts: tuple[FullRecordingAsrTranscriptRecord, ...],
    annotations: tuple[StoredConversationAnnotation, ...],
) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
    if not annotations:
        return transcripts
    external_ids = {annotation.external_id for annotation in annotations}
    return tuple(
        transcript for transcript in transcripts if transcript.sample_external_id in external_ids
    )


def _complete_transcript_pairs(
    transcripts: tuple[FullRecordingAsrTranscriptRecord, ...],
    model_ids: tuple[AsrModelId, ...],
) -> tuple[FullTranscriptPair, ...]:
    requested_models = set(model_ids)
    requested_transcripts = tuple(
        transcript for transcript in transcripts if transcript.model_id in requested_models
    )
    if not requested_transcripts:
        labels = ", ".join(model_id.value for model_id in model_ids)
        raise ValueError(f"No full-recording ASR transcripts found for models: {labels}")
    failed = tuple(
        transcript for transcript in requested_transcripts if transcript.error is not None
    )
    if failed:
        labels = ", ".join(
            f"{transcript.sample_external_id}/{transcript.side.value}/{transcript.model_id.value}"
            for transcript in failed[:10]
        )
        raise ValueError(f"Full-recording ASR transcripts contain errors: {labels}")

    transcript_by_key: dict[
        tuple[str, AsrModelId, TrackSide], FullRecordingAsrTranscriptRecord
    ] = {}
    for transcript in requested_transcripts:
        key = (transcript.sample_external_id, transcript.model_id, transcript.side)
        if key in transcript_by_key:
            raise ValueError(
                "Duplicate latest full-recording ASR transcript for "
                f"{transcript.sample_external_id}/{transcript.model_id.value}/"
                f"{transcript.side.value}"
            )
        transcript_by_key[key] = transcript

    external_ids = sorted({transcript.sample_external_id for transcript in requested_transcripts})
    missing = tuple(
        (external_id, model_id, side)
        for external_id in external_ids
        for model_id in model_ids
        for side in TrackSide
        if (external_id, model_id, side) not in transcript_by_key
    )
    if missing:
        labels = ", ".join(
            f"{external_id}/{model_id.value}/{side.value}"
            for external_id, model_id, side in missing[:10]
        )
        raise ValueError(f"Incomplete full-recording ASR model/track coverage: {labels}")

    return tuple(
        FullTranscriptPair(
            external_id=external_id,
            model_id=model_id,
            speaker1=transcript_by_key[(external_id, model_id, TrackSide.SPEAKER1)],
            speaker2=transcript_by_key[(external_id, model_id, TrackSide.SPEAKER2)],
        )
        for external_id in external_ids
        for model_id in model_ids
    )


def _full_recording_evidence_record(
    external_id: str,
    pairs: tuple[FullTranscriptPair, ...],
    annotation: StoredConversationAnnotation | None,
) -> OffsetEvidenceRecord:
    sources: list[OffsetEvidence] = []
    windows: list[OffsetWindowEvidence] = []
    if annotation is not None:
        annotation_duration_seconds = annotation.annotation.analyzed_duration_seconds
        speaker1_annotation_spans = annotation_spans(annotation.annotation.speaker1.speech_segments)
        speaker2_annotation_spans = annotation_spans(annotation.annotation.speaker2.speech_segments)
        annotation_metrics = timeline_metrics(
            speaker1_spans=speaker1_annotation_spans,
            speaker2_spans=speaker2_annotation_spans,
            duration_seconds=annotation_duration_seconds,
        )
        sources.append(
            OffsetEvidence(
                source=SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
                estimated_shift_seconds=annotation_metrics.best_lag_seconds,
                bad_state_improvement=annotation_metrics.bad_state_improvement,
                overlap_reduction=annotation_metrics.overlap_reduction,
                silence_reduction=annotation_metrics.silence_reduction,
            )
        )
        windows.extend(
            _window_evidence(
                source=SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
                speaker1_spans=speaker1_annotation_spans,
                speaker2_spans=speaker2_annotation_spans,
                duration_seconds=annotation_duration_seconds,
            )
        )
    for pair in pairs:
        source = _evidence_source(model_id=pair.model_id)
        duration_seconds = min(
            pair.speaker1.prepared_duration_seconds,
            pair.speaker2.prepared_duration_seconds,
        )
        speaker1_spans = timed_word_spans(pair.speaker1.words)
        speaker2_spans = timed_word_spans(pair.speaker2.words)
        if not speaker1_spans or not speaker2_spans:
            raise ValueError(
                "Full-recording ASR transcript has no timed speech for "
                f"{external_id}/{pair.model_id.value}"
            )
        metrics = timeline_metrics(
            speaker1_spans=speaker1_spans,
            speaker2_spans=speaker2_spans,
            duration_seconds=duration_seconds,
        )
        sources.append(
            OffsetEvidence(
                source=source,
                estimated_shift_seconds=metrics.best_lag_seconds,
                bad_state_improvement=metrics.bad_state_improvement,
                overlap_reduction=metrics.overlap_reduction,
                silence_reduction=metrics.silence_reduction,
            )
        )
        windows.extend(
            _window_evidence(
                source=source,
                speaker1_spans=speaker1_spans,
                speaker2_spans=speaker2_spans,
                duration_seconds=duration_seconds,
            )
        )
    return OffsetEvidenceRecord(
        external_id=external_id,
        scope=EvidenceScope.FULL_RECORDING,
        sources=tuple(sources),
        windows=tuple(
            sorted(windows, key=lambda window: (window.source.value, window.start_seconds))
        ),
    )


def _window_evidence(
    source: SynchronizationEvidenceSource,
    speaker1_spans: tuple[SpeechSpan, ...],
    speaker2_spans: tuple[SpeechSpan, ...],
    duration_seconds: float,
) -> tuple[OffsetWindowEvidence, ...]:
    windows: list[OffsetWindowEvidence] = []
    start_seconds = 0.0
    while start_seconds < duration_seconds:
        end_seconds = min(duration_seconds, start_seconds + FULL_EVIDENCE_WINDOW_SECONDS)
        window_duration_seconds = end_seconds - start_seconds
        if window_duration_seconds >= MINIMUM_FULL_EVIDENCE_WINDOW_SECONDS:
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
                duration_seconds=window_duration_seconds,
            )
            windows.append(
                OffsetWindowEvidence(
                    source=source,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    estimated_shift_seconds=metrics.best_lag_seconds,
                    bad_state_improvement=metrics.bad_state_improvement,
                    overlap_reduction=metrics.overlap_reduction,
                    silence_reduction=metrics.silence_reduction,
                )
            )
        start_seconds += FULL_EVIDENCE_WINDOW_SECONDS
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


def _validate_model_ids(model_ids: tuple[AsrModelId, ...]) -> None:
    if not model_ids:
        raise ValueError("At least one full-recording ASR model is required")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Full-recording evidence model IDs must be unique")
    unsupported = tuple(
        model_id for model_id in model_ids if model_id not in SUPPORTED_FULL_EVIDENCE_MODELS
    )
    if unsupported:
        labels = ", ".join(model_id.value for model_id in unsupported)
        raise ValueError(f"Unsupported synchronization evidence models: {labels}")


def _evidence_source(model_id: AsrModelId) -> SynchronizationEvidenceSource:
    match model_id:
        case AsrModelId.PARAKEET_TDT:
            return SynchronizationEvidenceSource.PARAKEET
        case AsrModelId.CANARY:
            return SynchronizationEvidenceSource.CANARY
        case _:
            raise ValueError(f"Unsupported synchronization evidence model: {model_id.value}")
