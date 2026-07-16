from __future__ import annotations

import wave
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from app.local.asr.full_recording_models import (
    FullRecordingAsrBatchItem,
    FullRecordingAsrBatchRecord,
    FullRecordingAsrBatchStatus,
    FullRecordingAsrSampleScope,
    FullRecordingAsrTranscriptRecord,
)
from app.local.asr.full_recording_repository import unique_models
from app.local.asr.full_recording_service import (
    FULL_ASR_SAMPLE_RATE,
    prepare_full_recording_audio,
    run_full_recording_asr_batch,
    transcribe_full_recording_item,
    validate_transcript_timestamps,
)
from app.local.db.models import TrackSide
from app.shared.asr import (
    AsrModelId,
    AsrTranscriptResult,
    RemoteAsrRequest,
    RemoteAsrResponse,
    TimestampedWord,
)
from app.shared.audio import probe_local_audio_metadata


class MemoryFullRecordingStore:
    def __init__(self, items: list[FullRecordingAsrBatchItem] | None = None) -> None:
        self.items = items or []
        self.transcripts: list[FullRecordingAsrTranscriptRecord] = []
        self.completed_item_ids: list[UUID] = []
        self.failed_item_ids: list[UUID] = []
        self.errors: list[str] = []
        self.recover_count = 0
        self.retry_count = 0

    def get_batch(self, batch_id: UUID) -> FullRecordingAsrBatchRecord:
        return batch_record(batch_id=batch_id)

    def recover_running_items(self, batch_id: UUID) -> int:
        self.recover_count += 1
        return 0

    def retry_failed_items(self, batch_id: UUID) -> int:
        self.retry_count += 1
        return 0

    def claim_next_item(self, batch_id: UUID) -> FullRecordingAsrBatchItem | None:
        return self.items.pop(0) if self.items else None

    def complete_item(self, item_id: UUID) -> None:
        self.completed_item_ids.append(item_id)

    def fail_item(self, item_id: UUID, error: str) -> None:
        self.failed_item_ids.append(item_id)
        self.errors.append(error)

    def finalize_batch(self, batch_id: UUID) -> FullRecordingAsrBatchRecord:
        return batch_record(batch_id=batch_id)

    def get_cached_transcripts(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        model_ids: tuple[AsrModelId, ...],
    ) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
        return tuple(
            transcript
            for transcript in self.transcripts
            if transcript.sample_track_id == sample_track_id
            and transcript.source_audio_sha256 == source_audio_sha256
            and transcript.model_id in model_ids
        )

    def upsert_transcript(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        prepared_audio_sha256: str,
        audio_filename: str,
        source_duration_seconds: float,
        prepared_duration_seconds: float,
        transcript: AsrTranscriptResult,
    ) -> FullRecordingAsrTranscriptRecord:
        record = transcript_record(
            sample_track_id=sample_track_id,
            source_audio_sha256=source_audio_sha256,
            prepared_audio_sha256=prepared_audio_sha256,
            audio_filename=audio_filename,
            source_duration_seconds=source_duration_seconds,
            prepared_duration_seconds=prepared_duration_seconds,
            transcript=transcript,
        )
        self.transcripts.append(record)
        return record


class RecordingRemoteAsrClient:
    def __init__(self, response: RemoteAsrResponse) -> None:
        self.response = response
        self.requests: list[RemoteAsrRequest] = []

    def transcribe(self, request: RemoteAsrRequest) -> RemoteAsrResponse:
        self.requests.append(request)
        return self.response


def test_prepare_full_recording_audio_streams_mono_16khz_flac(tmp_path: Path) -> None:
    source_path = tmp_path / "stereo.wav"
    write_wave(source_path=source_path, sample_rate=8_000, channel_count=2, duration_seconds=1.0)

    prepared = prepare_full_recording_audio(source_path)
    try:
        metadata = probe_local_audio_metadata(prepared.path)
        assert prepared.path.suffix == ".flac"
        assert metadata.sample_rate == FULL_ASR_SAMPLE_RATE
        assert metadata.channels == 1
        assert metadata.duration_seconds == pytest.approx(1.0, abs=0.01)
        assert prepared.source_duration_seconds == pytest.approx(1.0, abs=0.01)
        assert len(prepared.source_audio_sha256) == 64
        assert len(prepared.prepared_audio_sha256) == 64
    finally:
        prepared.path.unlink(missing_ok=True)


def test_full_recording_item_transcodes_calls_remote_and_persists(tmp_path: Path) -> None:
    source_path = tmp_path / "pmt_001_speaker1.wav"
    write_wave(source_path=source_path, sample_rate=16_000, channel_count=1, duration_seconds=1.0)
    item = batch_item(access_uri=str(source_path))
    store = MemoryFullRecordingStore()
    remote = RecordingRemoteAsrClient(
        RemoteAsrResponse(
            results=(
                AsrTranscriptResult(
                    model_id=AsrModelId.PARAKEET_TDT,
                    text="hello",
                    words=(TimestampedWord(text="hello", start_seconds=0.1, end_seconds=0.4),),
                ),
            )
        )
    )

    results = transcribe_full_recording_item(
        item=item,
        store=store,
        remote_client_factory=lambda: remote,
    )

    assert len(results) == 1
    assert results[0].sample_track_id == item.sample_track_id
    assert results[0].audio_filename == source_path.name
    assert len(remote.requests) == 1
    assert remote.requests[0].audio_filename.endswith(".flac")
    assert remote.requests[0].models == (AsrModelId.PARAKEET_TDT,)


def test_cached_full_recording_transcript_prevents_remote_call(tmp_path: Path) -> None:
    source_path = tmp_path / "pmt_001_speaker1.wav"
    write_wave(source_path=source_path, sample_rate=16_000, channel_count=1, duration_seconds=1.0)
    item = batch_item(access_uri=str(source_path))
    store = MemoryFullRecordingStore()
    prepared = prepare_full_recording_audio(source_path)
    try:
        store.upsert_transcript(
            sample_track_id=item.sample_track_id,
            source_audio_sha256=prepared.source_audio_sha256,
            prepared_audio_sha256=prepared.prepared_audio_sha256,
            audio_filename=source_path.name,
            source_duration_seconds=prepared.source_duration_seconds,
            prepared_duration_seconds=prepared.prepared_duration_seconds,
            transcript=AsrTranscriptResult(
                model_id=AsrModelId.PARAKEET_TDT,
                text="cached",
                words=(),
            ),
        )
    finally:
        prepared.path.unlink(missing_ok=True)

    results = transcribe_full_recording_item(
        item=item,
        store=store,
        remote_client_factory=unexpected_remote_client,
    )

    assert results[0].transcript_text == "cached"


def test_batch_recovery_and_failed_retry_are_explicit() -> None:
    batch_id = uuid4()
    store = MemoryFullRecordingStore(items=[batch_item(access_uri="missing.wav")])

    result = run_full_recording_asr_batch(
        batch_id=batch_id,
        store=store,
        remote_client_factory=unexpected_remote_client,
        maximum_tracks=None,
        recover_running=False,
        retry_failed=False,
    )

    assert result.id == batch_id
    assert store.recover_count == 0
    assert store.retry_count == 0
    assert len(store.failed_item_ids) == 1
    assert store.errors[0].startswith("ValueError: Full-recording ASR requires a local audio file")

    run_full_recording_asr_batch(
        batch_id=batch_id,
        store=store,
        remote_client_factory=unexpected_remote_client,
        maximum_tracks=1,
        recover_running=True,
        retry_failed=True,
    )
    assert store.recover_count == 1
    assert store.retry_count == 1


@pytest.mark.parametrize(
    "transcript,error_match",
    [
        (
            AsrTranscriptResult(
                model_id=AsrModelId.PARAKEET_TDT,
                text="",
                words=(),
                error="GPU out of memory",
            ),
            "GPU out of memory",
        ),
        (
            AsrTranscriptResult(
                model_id=AsrModelId.PARAKEET_TDT,
                text="late",
                words=(TimestampedWord(text="late", start_seconds=1.2, end_seconds=None),),
            ),
            "starts beyond",
        ),
        (
            AsrTranscriptResult(
                model_id=AsrModelId.PARAKEET_TDT,
                text="negative",
                words=(TimestampedWord(text="negative", start_seconds=None, end_seconds=-0.1),),
            ),
            "ends before",
        ),
    ],
)
def test_invalid_full_recording_transcripts_are_rejected(
    transcript: AsrTranscriptResult,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        validate_transcript_timestamps(transcript=transcript, duration_seconds=1.0)


def test_model_set_is_canonical_for_idempotency() -> None:
    assert unique_models((AsrModelId.PARAKEET_TDT, AsrModelId.CANARY, AsrModelId.PARAKEET_TDT)) == (
        AsrModelId.CANARY,
        AsrModelId.PARAKEET_TDT,
    )


def write_wave(
    source_path: Path,
    sample_rate: int,
    channel_count: int,
    duration_seconds: float,
) -> None:
    frame_count = round(sample_rate * duration_seconds)
    samples = np.zeros((frame_count, channel_count), dtype="<i2")
    with wave.open(str(source_path), "wb") as wave_writer:
        wave_writer.setnchannels(channel_count)
        wave_writer.setsampwidth(2)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(samples.tobytes())


def batch_item(access_uri: str) -> FullRecordingAsrBatchItem:
    return FullRecordingAsrBatchItem(
        id=uuid4(),
        batch_id=uuid4(),
        sample_track_id=uuid4(),
        sample_external_id="pmt_001",
        side=TrackSide.SPEAKER1,
        access_uri=access_uri,
        models=(AsrModelId.PARAKEET_TDT,),
        attempt_count=1,
    )


def batch_record(batch_id: UUID) -> FullRecordingAsrBatchRecord:
    now = datetime.now(UTC)
    return FullRecordingAsrBatchRecord(
        id=batch_id,
        idempotency_key="test-batch",
        dataset_id=uuid4(),
        sample_scope=FullRecordingAsrSampleScope.QUALITY_ANALYZED,
        models=(AsrModelId.PARAKEET_TDT,),
        status=FullRecordingAsrBatchStatus.RUNNING,
        total_track_count=1,
        pending_track_count=0,
        running_track_count=0,
        completed_track_count=1,
        failed_track_count=0,
        recent_errors=(),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=None,
    )


def transcript_record(
    sample_track_id: UUID,
    source_audio_sha256: str,
    prepared_audio_sha256: str,
    audio_filename: str,
    source_duration_seconds: float,
    prepared_duration_seconds: float,
    transcript: AsrTranscriptResult,
) -> FullRecordingAsrTranscriptRecord:
    now = datetime.now(UTC)
    return FullRecordingAsrTranscriptRecord(
        id=uuid4(),
        sample_track_id=sample_track_id,
        sample_id=uuid4(),
        sample_external_id="pmt_001",
        side=TrackSide.SPEAKER1,
        source_audio_sha256=source_audio_sha256,
        prepared_audio_sha256=prepared_audio_sha256,
        audio_filename=audio_filename,
        model_id=transcript.model_id,
        transcript_text=transcript.text,
        words=transcript.words,
        source_duration_seconds=source_duration_seconds,
        prepared_duration_seconds=prepared_duration_seconds,
        processing_time_seconds=transcript.processing_time_seconds,
        runtime=transcript.runtime,
        error=transcript.error,
        created_at=now,
        updated_at=now,
    )


def unexpected_remote_client() -> RecordingRemoteAsrClient:
    raise AssertionError("Remote ASR should not be called.")
