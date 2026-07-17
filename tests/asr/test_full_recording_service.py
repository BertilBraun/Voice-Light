from __future__ import annotations

import wave
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.asr.full_recording_service import (
    FULL_ASR_SAMPLE_RATE,
    FullRecordingAsrTrack,
    prepare_full_recording_audio,
    transcribe_full_recording_track,
    validate_transcript_timestamps,
)
from app.local.db.models import TrackSide
from app.shared.asr import (
    AsrModelId,
    AsrRequestStats,
    AsrRuntimeStats,
    AsrTranscriptResult,
    RemoteAsrResponse,
    RemoteAsrUploadRequest,
    TimestampedWord,
)
from app.shared.audio import probe_local_audio_metadata


class MemoryFullRecordingStore:
    def __init__(self) -> None:
        self.transcripts: list[FullRecordingAsrTranscriptRecord] = []

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
        self.requests: list[RemoteAsrUploadRequest] = []
        self.uploaded_audio: list[bytes] = []

    def transcribe_upload(
        self,
        request: RemoteAsrUploadRequest,
        audio_path: Path,
    ) -> RemoteAsrResponse:
        self.requests.append(request)
        self.uploaded_audio.append(audio_path.read_bytes())
        return self.response


def test_prepare_full_recording_audio_streams_mono_16khz_ogg_opus(tmp_path: Path) -> None:
    source_path = tmp_path / "stereo.wav"
    write_wave(source_path=source_path, sample_rate=8_000, channel_count=2, duration_seconds=1.0)

    prepared = prepare_full_recording_audio(source_path)
    try:
        metadata = probe_local_audio_metadata(prepared.path)
        assert prepared.path.suffix == ".ogg"
        assert prepared.transport_metadata.codec.value == "ogg_opus"
        assert prepared.transport_metadata.sample_rate == FULL_ASR_SAMPLE_RATE
        assert prepared.transport_metadata.size_bytes == prepared.path.stat().st_size
        assert metadata.channels == 1
        assert metadata.duration_seconds == pytest.approx(1.0, abs=0.01)
        assert prepared.source_duration_seconds == pytest.approx(1.0, abs=0.01)
        assert len(prepared.source_audio_sha256) == 64
        assert len(prepared.prepared_audio_sha256) == 64
    finally:
        prepared.path.unlink(missing_ok=True)


def test_full_recording_track_transcodes_calls_remote_and_persists(tmp_path: Path) -> None:
    source_path = tmp_path / "pmt_001_speaker1.wav"
    write_wave(source_path=source_path, sample_rate=16_000, channel_count=1, duration_seconds=1.0)
    track = full_recording_track(access_uri=str(source_path))
    store = MemoryFullRecordingStore()
    remote = RecordingRemoteAsrClient(
        RemoteAsrResponse(
            results=(
                AsrTranscriptResult(
                    model_id=AsrModelId.PARAKEET_TDT,
                    text="hello",
                    words=(TimestampedWord(text="hello", start_seconds=0.1, end_seconds=0.4),),
                    processing_time_seconds=0.25,
                    runtime=AsrRuntimeStats(
                        processing_time_seconds=0.25,
                        inference_time_seconds=0.25,
                    ),
                ),
            ),
            request_stats=AsrRequestStats(
                upload_stream_time_seconds=0.01,
                upload_validation_time_seconds=0.02,
                audio_preparation_time_seconds=0.03,
                transcription_time_seconds=0.25,
                request_time_seconds=0.32,
            ),
        )
    )

    results = transcribe_full_recording_track(
        track=track,
        store=store,
        remote_client_factory=lambda: remote,
    )

    assert len(results) == 1
    assert results[0].sample_track_id == track.sample_track_id
    assert results[0].audio_filename == source_path.name
    assert len(remote.requests) == 1
    assert remote.requests[0].audio.encoded_filename.endswith(".ogg")
    assert remote.requests[0].models == (AsrModelId.PARAKEET_TDT,)
    assert remote.uploaded_audio[0].startswith(b"OggS")
    assert results[0].runtime is not None
    assert results[0].runtime.client_audio_preparation_time_seconds is not None
    assert results[0].runtime.client_request_time_seconds is not None
    assert results[0].runtime.uploaded_audio_size_bytes == len(remote.uploaded_audio[0])
    assert results[0].runtime.server_audio_preparation_time_seconds == 0.03
    assert results[0].runtime.server_request_time_seconds == 0.32


def test_cached_full_recording_transcript_prevents_remote_call(tmp_path: Path) -> None:
    source_path = tmp_path / "pmt_001_speaker1.wav"
    write_wave(source_path=source_path, sample_rate=16_000, channel_count=1, duration_seconds=1.0)
    track = full_recording_track(access_uri=str(source_path))
    store = MemoryFullRecordingStore()
    prepared = prepare_full_recording_audio(source_path)
    try:
        store.upsert_transcript(
            sample_track_id=track.sample_track_id,
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

    results = transcribe_full_recording_track(
        track=track,
        store=store,
        remote_client_factory=unexpected_remote_client,
    )

    assert results[0].transcript_text == "cached"


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


def full_recording_track(access_uri: str) -> FullRecordingAsrTrack:
    return FullRecordingAsrTrack(
        sample_track_id=uuid4(),
        access_uri=access_uri,
        models=(AsrModelId.PARAKEET_TDT,),
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
