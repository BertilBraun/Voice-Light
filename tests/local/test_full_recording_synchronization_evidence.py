from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.db.models import TrackSide
from app.local.synchronization_review.evaluation import EvidenceScope
from app.local.synchronization_review.full_recording_evidence import (
    full_recording_evidence_records,
)
from app.local.synchronization_review.models import SynchronizationEvidenceSource
from app.local.synchronization_review.repository import StoredConversationAnnotation
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.quality import AnnotationSpan, ConversationAnnotation, SpeakerConversationAnnotation


def test_full_recording_evidence_uses_whole_timeline_and_fixed_windows() -> None:
    transcripts = tuple(
        _transcript(
            model_id=model_id,
            side=side,
            shift_seconds=0.0 if side is TrackSide.SPEAKER1 else 8.0,
        )
        for model_id in (AsrModelId.PARAKEET_TDT, AsrModelId.CANARY)
        for side in TrackSide
    )

    records = full_recording_evidence_records(transcripts=transcripts)

    assert len(records) == 1
    record = records[0]
    assert record.external_id == "sample_001"
    assert record.scope is EvidenceScope.FULL_RECORDING
    assert {source.source for source in record.sources} == {
        SynchronizationEvidenceSource.PARAKEET,
        SynchronizationEvidenceSource.CANARY,
    }
    assert all(source.estimated_shift_seconds == pytest.approx(-3.0) for source in record.sources)
    assert len(record.windows) == 4
    assert {(window.start_seconds, window.end_seconds) for window in record.windows} == {
        (0.0, 180.0),
        (180.0, 360.0),
    }


def test_full_recording_evidence_rejects_incomplete_model_track_coverage() -> None:
    transcripts = tuple(
        _transcript(model_id=model_id, side=side, shift_seconds=0.0)
        for model_id, side in (
            (AsrModelId.PARAKEET_TDT, TrackSide.SPEAKER1),
            (AsrModelId.PARAKEET_TDT, TrackSide.SPEAKER2),
            (AsrModelId.CANARY, TrackSide.SPEAKER1),
        )
    )

    with pytest.raises(ValueError, match="Incomplete full-recording ASR model/track coverage"):
        full_recording_evidence_records(transcripts=transcripts)


def test_full_recording_windows_preserve_local_timing_and_expose_changed_lag() -> None:
    speaker1_spans = (
        (20.0, 26.0),
        (42.0, 50.0),
        (75.0, 80.0),
        (110.0, 117.0),
        (151.0, 158.0),
        (200.0, 207.0),
        (229.0, 235.0),
        (267.0, 276.0),
        (310.0, 315.0),
        (350.0, 357.0),
    )
    speaker2_spans = (
        (29.0, 45.0),
        (53.0, 78.0),
        (83.0, 113.0),
        (120.0, 154.0),
        (208.0, 230.0),
        (236.0, 268.0),
        (277.0, 311.0),
        (316.0, 351.0),
    )
    speaker1 = _transcript(
        model_id=AsrModelId.PARAKEET_TDT,
        side=TrackSide.SPEAKER1,
        shift_seconds=0.0,
    ).model_copy(update={"words": _words_from_spans(spans=speaker1_spans)})
    speaker2 = _transcript(
        model_id=AsrModelId.PARAKEET_TDT,
        side=TrackSide.SPEAKER2,
        shift_seconds=0.0,
    ).model_copy(update={"words": _words_from_spans(spans=speaker2_spans)})

    records = full_recording_evidence_records(
        transcripts=(speaker1, speaker2),
        model_ids=(AsrModelId.PARAKEET_TDT,),
    )

    assert [window.start_seconds for window in records[0].windows] == [0.0, 180.0]
    assert [window.estimated_shift_seconds for window in records[0].windows] == pytest.approx(
        [-3.0, -1.0]
    )


def test_full_recording_evidence_includes_full_annotation_windows() -> None:
    transcripts = tuple(
        _transcript(model_id=model_id, side=side, shift_seconds=0.0)
        for model_id in (AsrModelId.PARAKEET_TDT, AsrModelId.CANARY)
        for side in TrackSide
    )
    annotation = StoredConversationAnnotation(
        sample_id=transcripts[0].sample_id,
        external_id="sample_001",
        annotation=ConversationAnnotation(
            annotation_version="test-full-annotation",
            analyzed_duration_seconds=400.0,
            speaker1=_speaker_annotation(
                side="speaker1",
                starts=tuple(float(start_seconds) for start_seconds in range(0, 390, 10)),
            ),
            speaker2=_speaker_annotation(
                side="speaker2",
                starts=tuple(float(start_seconds + 8) for start_seconds in range(0, 380, 10)),
            ),
            speech_segment_count=78,
            turn_count=0,
            turn_taking_count=0,
            interaction_count=0,
            pause_count=0,
            backchannel_count=0,
            interruption_count=0,
            usable_event_count=0,
            events_per_hour=0.0,
            speaker_balance_score=1.0,
            quality_score=1.0,
        ),
        audio_quality=None,
    )

    records = full_recording_evidence_records(
        transcripts=transcripts,
        annotations=(annotation,),
    )

    assert {source.source for source in records[0].sources} == {
        SynchronizationEvidenceSource.CONVERSATION_ANNOTATION,
        SynchronizationEvidenceSource.PARAKEET,
        SynchronizationEvidenceSource.CANARY,
    }
    annotation_windows = tuple(
        window
        for window in records[0].windows
        if window.source is SynchronizationEvidenceSource.CONVERSATION_ANNOTATION
    )
    assert [(window.start_seconds, window.end_seconds) for window in annotation_windows] == [
        (0.0, 180.0),
        (180.0, 360.0),
    ]


def _speaker_annotation(
    side: str,
    starts: tuple[float, ...],
) -> SpeakerConversationAnnotation:
    return SpeakerConversationAnnotation(
        side=side,
        speech_segments=tuple(
            AnnotationSpan(
                start_seconds=start_seconds,
                end_seconds=start_seconds + 5.0,
                text=None,
            )
            for start_seconds in starts
        ),
        pauses=(),
        backchannels=(),
        turns=(),
        interruptions=(),
        segment_targets=(),
        connection_targets=(),
        speech_duration_seconds=len(starts) * 5.0,
        pause_duration_seconds=0.0,
        backchannel_duration_seconds=0.0,
    )


def _transcript(
    model_id: AsrModelId,
    side: TrackSide,
    shift_seconds: float,
) -> FullRecordingAsrTranscriptRecord:
    now = datetime.now(tz=UTC)
    return FullRecordingAsrTranscriptRecord(
        id=uuid4(),
        sample_track_id=uuid4(),
        sample_id=uuid4(),
        sample_external_id="sample_001",
        side=side,
        source_audio_sha256="a" * 64,
        prepared_audio_sha256="b" * 64,
        audio_filename=f"sample_001_{side.value}.flac",
        model_id=model_id,
        transcript_text="test words",
        words=tuple(
            TimestampedWord(
                text=f"word-{index}",
                start_seconds=start_seconds + shift_seconds,
                end_seconds=start_seconds + shift_seconds + 5.0,
            )
            for index, start_seconds in enumerate(range(0, 390, 10))
        ),
        language_estimate=None,
        source_duration_seconds=400.0,
        prepared_duration_seconds=400.0,
        processing_time_seconds=10.0,
        runtime=None,
        error=None,
        created_at=now,
        updated_at=now,
    )


def _words(starts: tuple[float, ...]) -> tuple[TimestampedWord, ...]:
    return tuple(
        TimestampedWord(
            text=f"word-{index}",
            start_seconds=start_seconds,
            end_seconds=start_seconds + 5.0,
        )
        for index, start_seconds in enumerate(starts)
    )


def _words_from_spans(spans: tuple[tuple[float, float], ...]) -> tuple[TimestampedWord, ...]:
    return tuple(
        TimestampedWord(
            text=f"word-{index}",
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        for index, (start_seconds, end_seconds) in enumerate(spans)
    )
