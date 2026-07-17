from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.asr.full_recording_repository import FullRecordingAsrRepository
from app.local.db.models import TrackSide
from app.shared.asr import AsrModelId


class MemoryPairRepository(FullRecordingAsrRepository):
    def __init__(self, transcripts: tuple[FullRecordingAsrTranscriptRecord, ...]) -> None:
        self.transcripts = transcripts

    def get_cached_transcripts(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        model_ids: Sequence[AsrModelId],
    ) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
        return tuple(
            transcript
            for transcript in self.transcripts
            if transcript.sample_track_id == sample_track_id
            and transcript.source_audio_sha256 == source_audio_sha256
            and transcript.model_id in model_ids
        )


def test_exact_pair_requires_current_source_hash_for_both_tracks() -> None:
    sample_id = uuid4()
    speaker1 = transcript_record(sample_id=sample_id, side=TrackSide.SPEAKER1, source_hash="a" * 64)
    speaker2 = transcript_record(sample_id=sample_id, side=TrackSide.SPEAKER2, source_hash="b" * 64)
    repository = MemoryPairRepository((speaker1, speaker2))

    current_pair = repository.get_exact_transcript_pair(
        sample_id=sample_id,
        speaker1_track_id=speaker1.sample_track_id,
        speaker1_source_audio_sha256="a" * 64,
        speaker2_track_id=speaker2.sample_track_id,
        speaker2_source_audio_sha256="b" * 64,
        model_id=AsrModelId.PARAKEET_TDT,
    )
    stale_pair = repository.get_exact_transcript_pair(
        sample_id=sample_id,
        speaker1_track_id=speaker1.sample_track_id,
        speaker1_source_audio_sha256="c" * 64,
        speaker2_track_id=speaker2.sample_track_id,
        speaker2_source_audio_sha256="b" * 64,
        model_id=AsrModelId.PARAKEET_TDT,
    )

    assert current_pair is not None
    assert current_pair.speaker1.id == speaker1.id
    assert current_pair.speaker2.id == speaker2.id
    assert stale_pair is None


def test_exact_pair_rejects_cached_error_on_either_track() -> None:
    sample_id = uuid4()
    speaker1 = transcript_record(
        sample_id=sample_id,
        side=TrackSide.SPEAKER1,
        source_hash="a" * 64,
    )
    speaker2 = transcript_record(
        sample_id=sample_id,
        side=TrackSide.SPEAKER2,
        source_hash="b" * 64,
    ).model_copy(update={"error": "failed"})
    repository = MemoryPairRepository((speaker1, speaker2))

    pair = repository.get_exact_transcript_pair(
        sample_id=sample_id,
        speaker1_track_id=speaker1.sample_track_id,
        speaker1_source_audio_sha256="a" * 64,
        speaker2_track_id=speaker2.sample_track_id,
        speaker2_source_audio_sha256="b" * 64,
        model_id=AsrModelId.PARAKEET_TDT,
    )

    assert pair is None


def transcript_record(
    sample_id: UUID,
    side: TrackSide,
    source_hash: str,
) -> FullRecordingAsrTranscriptRecord:
    now = datetime.now(tz=UTC)
    return FullRecordingAsrTranscriptRecord(
        id=uuid4(),
        sample_track_id=uuid4(),
        sample_id=sample_id,
        sample_external_id="pmt_001",
        side=side,
        source_audio_sha256=source_hash,
        prepared_audio_sha256="d" * 64,
        audio_filename=f"pmt_001_{side.value}.flac",
        model_id=AsrModelId.PARAKEET_TDT,
        transcript_text="",
        words=(),
        source_duration_seconds=10.0,
        prepared_duration_seconds=10.0,
        processing_time_seconds=1.0,
        runtime=None,
        error=None,
        created_at=now,
        updated_at=now,
    )
