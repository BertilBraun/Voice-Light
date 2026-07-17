from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import numpy as np
import pytest

from app.local.asr.full_recording_models import (
    FullRecordingAsrTranscriptBundle,
    FullRecordingAsrTranscriptPair,
    FullRecordingAsrTranscriptRecord,
)
from app.local.db.models import TrackSide
from app.local.ingestion.alignment import SynchronizationAlignment, SynchronizationAlignmentOrigin
from app.local.ingestion.conversation import ANNOTATION_VERSION, analyze_conversation
from app.local.ingestion.discovery import DiscoveredSample
from app.local.ingestion.service import (
    RegisteredSample,
    ready_sample,
    track_durations_are_compatible,
)
from app.shared.asr import AsrModelId, TimestampedWord
from app.shared.audio import AudioTrack
from app.shared.quality import AudioMetadata


class ReadinessRepository:
    def __init__(self, reviewed_shift_seconds: float | None) -> None:
        self.reviewed_shift_seconds = reviewed_shift_seconds

    def reviewed_speaker2_shift(self, sample_id: UUID) -> float | None:
        del sample_id
        return self.reviewed_shift_seconds


class ReadinessFullAsrRepository:
    def __init__(self, transcripts: FullRecordingAsrTranscriptBundle | None) -> None:
        self.transcripts = transcripts
        self.requests: list[tuple[UUID, UUID, str, UUID, str, AsrModelId]] = []

    def get_exact_transcript_pair(
        self,
        sample_id: UUID,
        speaker1_track_id: UUID,
        speaker1_source_audio_sha256: str,
        speaker2_track_id: UUID,
        speaker2_source_audio_sha256: str,
        model_id: AsrModelId,
    ) -> FullRecordingAsrTranscriptPair | None:
        self.requests.append(
            (
                sample_id,
                speaker1_track_id,
                speaker1_source_audio_sha256,
                speaker2_track_id,
                speaker2_source_audio_sha256,
                model_id,
            )
        )
        if self.transcripts is None:
            return None
        return (
            self.transcripts.parakeet
            if model_id is AsrModelId.PARAKEET_TDT
            else self.transcripts.canary
        )


def test_unreviewed_sample_uses_explicit_auditable_zero_alignment() -> None:
    registered = registered_sample()
    transcripts = transcript_bundle(registered=registered)

    ready = ready_sample(
        repository=ReadinessRepository(reviewed_shift_seconds=None),
        full_asr_repository=ReadinessFullAsrRepository(transcripts=transcripts),
        registered=registered,
    )

    assert ready.alignment.speaker2_shift_seconds == 0.0
    assert ready.alignment.origin is SynchronizationAlignmentOrigin.UNREVIEWED_ZERO


def test_reviewed_sample_uses_stored_offset() -> None:
    registered = registered_sample()

    ready = ready_sample(
        repository=ReadinessRepository(reviewed_shift_seconds=-2.5),
        full_asr_repository=ReadinessFullAsrRepository(
            transcripts=transcript_bundle(registered=registered)
        ),
        registered=registered,
    )

    assert ready.alignment.speaker2_shift_seconds == -2.5
    assert ready.alignment.origin is SynchronizationAlignmentOrigin.REVIEWED


def test_full_conversation_annotation_uses_words_beyond_three_minutes() -> None:
    registered = registered_sample()
    transcripts = transcript_bundle(registered=registered)
    pair = transcripts.parakeet
    pair = pair.model_copy(
        update={
            "speaker1": pair.speaker1.model_copy(
                update={
                    "words": (
                        TimestampedWord(
                            text="late speaker one",
                            start_seconds=200.0,
                            end_seconds=202.0,
                        ),
                    )
                }
            ),
            "speaker2": pair.speaker2.model_copy(
                update={
                    "words": (
                        TimestampedWord(
                            text="late speaker two",
                            start_seconds=205.0,
                            end_seconds=207.0,
                        ),
                    )
                }
            ),
        }
    )
    audio = silent_audio_track(duration_seconds=240.0)

    annotation = analyze_conversation(
        speaker1_audio=audio,
        speaker2_audio=audio,
        transcripts=transcripts.model_copy(
            update={
                "parakeet": pair,
                "canary": pair.model_copy(update={"model_id": AsrModelId.CANARY}),
            }
        ),
        alignment=SynchronizationAlignment(
            speaker2_shift_seconds=0.0,
            origin=SynchronizationAlignmentOrigin.UNREVIEWED_ZERO,
        ),
    )

    assert annotation.annotation_version == ANNOTATION_VERSION
    assert annotation.analyzed_duration_seconds == 240.0
    assert any(
        target.start_seconds >= 200.0
        for speaker in (annotation.speaker1, annotation.speaker2)
        for target in speaker.segment_targets
    )


@pytest.mark.parametrize(
    ("speaker1_duration_seconds", "speaker2_duration_seconds", "expected"),
    (
        (1_000.0, 1_010.0, True),
        (1_000.0, 1_010.01, False),
        (2_000.0, 2_020.0, True),
        (2_001.0, 2_021.01, False),
        (10_000.0, 10_020.0, True),
        (10_000.0, 10_020.01, False),
    ),
)
def test_track_duration_compatibility_requires_absolute_and_relative_limits(
    speaker1_duration_seconds: float,
    speaker2_duration_seconds: float,
    expected: bool,
) -> None:
    assert (
        track_durations_are_compatible(
            speaker1_duration_seconds=speaker1_duration_seconds,
            speaker2_duration_seconds=speaker2_duration_seconds,
        )
        is expected
    )


def registered_sample() -> RegisteredSample:
    metadata = AudioMetadata(
        duration_seconds=10.0,
        sample_rate=16_000,
        channels=1,
        sample_count=160_000,
    )
    return RegisteredSample(
        discovered=DiscoveredSample(
            external_id="pmt_001",
            speaker1_path="speaker1.wav",
            speaker2_path="speaker2.wav",
        ),
        sample_id=uuid4(),
        speaker1_track_id=uuid4(),
        speaker2_track_id=uuid4(),
        speaker1_metadata=metadata,
        speaker2_metadata=metadata,
        speaker1_source_audio_sha256="a" * 64,
        speaker2_source_audio_sha256="b" * 64,
    )


def transcript_pair(registered: RegisteredSample) -> FullRecordingAsrTranscriptPair:
    return transcript_pair_for_model(
        registered=registered,
        model_id=AsrModelId.PARAKEET_TDT,
    )


def transcript_bundle(registered: RegisteredSample) -> FullRecordingAsrTranscriptBundle:
    return FullRecordingAsrTranscriptBundle(
        parakeet=transcript_pair_for_model(
            registered=registered,
            model_id=AsrModelId.PARAKEET_TDT,
        ),
        canary=transcript_pair_for_model(
            registered=registered,
            model_id=AsrModelId.CANARY,
        ),
    )


def transcript_pair_for_model(
    registered: RegisteredSample,
    model_id: AsrModelId,
) -> FullRecordingAsrTranscriptPair:
    return FullRecordingAsrTranscriptPair(
        sample_id=registered.sample_id,
        sample_external_id=registered.discovered.external_id,
        model_id=model_id,
        speaker1=transcript_record(
            registered=registered,
            side=TrackSide.SPEAKER1,
            track_id=registered.speaker1_track_id,
            model_id=model_id,
        ),
        speaker2=transcript_record(
            registered=registered,
            side=TrackSide.SPEAKER2,
            track_id=registered.speaker2_track_id,
            model_id=model_id,
        ),
    )


def transcript_record(
    registered: RegisteredSample,
    side: TrackSide,
    track_id: UUID,
    model_id: AsrModelId = AsrModelId.PARAKEET_TDT,
) -> FullRecordingAsrTranscriptRecord:
    now = datetime.now(tz=UTC)
    return FullRecordingAsrTranscriptRecord(
        id=uuid4(),
        sample_track_id=track_id,
        sample_id=registered.sample_id,
        sample_external_id=registered.discovered.external_id,
        side=side,
        source_audio_sha256=(
            registered.speaker1_source_audio_sha256
            if side is TrackSide.SPEAKER1
            else registered.speaker2_source_audio_sha256
        ),
        prepared_audio_sha256="c" * 64,
        audio_filename=f"{registered.discovered.external_id}_{side.value}.flac",
        model_id=model_id,
        transcript_text="hello",
        words=(
            TimestampedWord(
                text="hello",
                start_seconds=1.0,
                end_seconds=2.0,
            ),
        ),
        source_duration_seconds=10.0,
        prepared_duration_seconds=10.0,
        processing_time_seconds=1.0,
        runtime=None,
        error=None,
        created_at=now,
        updated_at=now,
    )


def silent_audio_track(duration_seconds: float) -> AudioTrack:
    sample_rate = 100
    sample_count = round(duration_seconds * sample_rate)
    return AudioTrack(
        samples=np.zeros((sample_count, 1), dtype=np.float32),
        metadata=AudioMetadata(
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=1,
            sample_count=sample_count,
        ),
    )
