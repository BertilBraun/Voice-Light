from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from app.local.asr.full_recording_models import FullRecordingAsrTranscriptRecord
from app.local.asr.service import RemoteAsrClient
from app.shared.asr import (
    AsrModelId,
    AsrRequestStats,
    AsrTranscriptResult,
    RemoteAsrUploadRequest,
)
from app.shared.audio import probe_local_audio_metadata
from app.shared.audio.transport import (
    ASR_AUDIO_TRANSPORT_SPEC,
    AudioTransportMetadata,
    prepare_audio_transport,
    sha256_file,
)

FULL_ASR_SAMPLE_RATE = ASR_AUDIO_TRANSPORT_SPEC.sample_rate
DURATION_TOLERANCE_SECONDS = 0.1


@dataclass(frozen=True)
class PreparedFullRecordingAudio:
    path: Path
    transport_metadata: AudioTransportMetadata
    source_audio_sha256: str
    prepared_audio_sha256: str
    source_duration_seconds: float
    prepared_duration_seconds: float


@dataclass(frozen=True)
class FullRecordingAsrTrack:
    sample_track_id: UUID
    access_uri: str
    models: tuple[AsrModelId, ...]
    source_audio_sha256: str | None = None


@dataclass(frozen=True)
class FullRecordingAsrClientStats:
    audio_preparation_time_seconds: float
    request_time_seconds: float
    uploaded_audio_size_bytes: int


class FullRecordingTranscriptStore(Protocol):
    def get_cached_transcripts(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        model_ids: Sequence[AsrModelId],
    ) -> tuple[FullRecordingAsrTranscriptRecord, ...]: ...

    def upsert_transcript(
        self,
        sample_track_id: UUID,
        source_audio_sha256: str,
        prepared_audio_sha256: str,
        audio_filename: str,
        source_duration_seconds: float,
        prepared_duration_seconds: float,
        transcript: AsrTranscriptResult,
    ) -> FullRecordingAsrTranscriptRecord: ...


RemoteAsrClientFactory = Callable[[], RemoteAsrClient]


def transcribe_full_recording_track(
    track: FullRecordingAsrTrack,
    store: FullRecordingTranscriptStore,
    remote_client_factory: RemoteAsrClientFactory,
) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
    source_path = local_audio_path(track.access_uri)
    source_audio_sha256 = track.source_audio_sha256 or sha256_file(source_path)
    cached = store.get_cached_transcripts(
        sample_track_id=track.sample_track_id,
        source_audio_sha256=source_audio_sha256,
        model_ids=track.models,
    )
    missing_models = missing_model_ids(requested=track.models, transcripts=cached)
    if not missing_models:
        return cached

    audio_preparation_start = time.perf_counter()
    prepared = prepare_full_recording_audio(
        source_path=source_path,
        source_audio_sha256=source_audio_sha256,
    )
    audio_preparation_time_seconds = time.perf_counter() - audio_preparation_start
    try:
        request_start = time.perf_counter()
        response = remote_client_factory().transcribe_upload(
            request=RemoteAsrUploadRequest(
                audio=prepared.transport_metadata,
                models=missing_models,
            ),
            audio_path=prepared.path,
        )
        request_time_seconds = time.perf_counter() - request_start
        client_stats = FullRecordingAsrClientStats(
            audio_preparation_time_seconds=audio_preparation_time_seconds,
            request_time_seconds=request_time_seconds,
            uploaded_audio_size_bytes=prepared.transport_metadata.size_bytes,
        )
        results = requested_results(requested=missing_models, results=response.results)
        for result in results:
            result = transcript_with_pipeline_stats(
                transcript=result,
                client_stats=client_stats,
                server_stats=response.request_stats,
            )
            validate_transcript_timestamps(
                transcript=result,
                duration_seconds=prepared.prepared_duration_seconds,
            )
            store.upsert_transcript(
                sample_track_id=track.sample_track_id,
                source_audio_sha256=prepared.source_audio_sha256,
                prepared_audio_sha256=prepared.prepared_audio_sha256,
                audio_filename=source_path.name,
                source_duration_seconds=prepared.source_duration_seconds,
                prepared_duration_seconds=prepared.prepared_duration_seconds,
                transcript=result,
            )
    finally:
        prepared.path.unlink(missing_ok=True)

    final = store.get_cached_transcripts(
        sample_track_id=track.sample_track_id,
        source_audio_sha256=source_audio_sha256,
        model_ids=track.models,
    )
    remaining_models = missing_model_ids(requested=track.models, transcripts=final)
    if remaining_models:
        labels = ", ".join(model.value for model in remaining_models)
        raise ValueError(f"Full-recording ASR persistence is missing models: {labels}")
    return final


def transcript_with_pipeline_stats(
    transcript: AsrTranscriptResult,
    client_stats: FullRecordingAsrClientStats,
    server_stats: AsrRequestStats | None,
) -> AsrTranscriptResult:
    if transcript.runtime is None:
        return transcript
    runtime_update: dict[str, float | int] = {
        "client_audio_preparation_time_seconds": client_stats.audio_preparation_time_seconds,
        "client_request_time_seconds": client_stats.request_time_seconds,
        "uploaded_audio_size_bytes": client_stats.uploaded_audio_size_bytes,
    }
    if server_stats is not None:
        runtime_update.update(
            {
                "server_upload_stream_time_seconds": (server_stats.upload_stream_time_seconds),
                "server_upload_validation_time_seconds": (
                    server_stats.upload_validation_time_seconds
                ),
                "server_audio_preparation_time_seconds": (
                    server_stats.audio_preparation_time_seconds
                ),
                "server_transcription_time_seconds": (server_stats.transcription_time_seconds),
                "server_request_time_seconds": server_stats.request_time_seconds,
            }
        )
    return transcript.model_copy(
        update={
            "runtime": transcript.runtime.model_copy(update=runtime_update),
        }
    )


def prepare_full_recording_audio(
    source_path: Path,
    source_audio_sha256: str | None = None,
) -> PreparedFullRecordingAudio:
    if not source_path.is_file():
        raise ValueError(f"Full-recording ASR source file not found: {source_path}")
    source_duration_seconds = probe_local_audio_metadata(source_path).duration_seconds
    if source_duration_seconds <= 0.0:
        raise ValueError(f"Full-recording ASR source has no audio: {source_path}")
    output_file = tempfile.NamedTemporaryFile(
        prefix=f"{source_path.stem}_full_asr_",
        suffix=".ogg",
        delete=False,
    )
    output_path = Path(output_file.name)
    output_file.close()
    try:
        transport_metadata = prepare_audio_transport(
            source_path=source_path,
            output_path=output_path,
            spec=ASR_AUDIO_TRANSPORT_SPEC,
        )
        prepared_metadata = probe_local_audio_metadata(output_path)
        if prepared_metadata.channels != 1:
            raise ValueError(
                f"Prepared ASR audio has {prepared_metadata.channels} channels, expected one."
            )
        validate_duration_match(
            source_duration_seconds=source_duration_seconds,
            prepared_duration_seconds=prepared_metadata.duration_seconds,
        )
        return PreparedFullRecordingAudio(
            path=output_path,
            transport_metadata=transport_metadata,
            source_audio_sha256=source_audio_sha256 or sha256_file(source_path),
            prepared_audio_sha256=transport_metadata.sha256,
            source_duration_seconds=source_duration_seconds,
            prepared_duration_seconds=transport_metadata.duration_seconds,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def validate_duration_match(
    source_duration_seconds: float,
    prepared_duration_seconds: float,
) -> None:
    difference_seconds = abs(source_duration_seconds - prepared_duration_seconds)
    if difference_seconds > DURATION_TOLERANCE_SECONDS:
        raise ValueError(
            "Prepared ASR audio duration differs from its source by "
            f"{difference_seconds:.3f} seconds."
        )


def validate_transcript_timestamps(
    transcript: AsrTranscriptResult,
    duration_seconds: float,
) -> None:
    if transcript.error is not None:
        raise ValueError(
            f"Remote full-recording ASR failed for {transcript.model_id.value}: {transcript.error}"
        )
    for word in transcript.words:
        if word.start_seconds is not None and word.start_seconds < 0.0:
            raise ValueError(f"ASR word starts before the full recording: {word.text}")
        if (
            word.start_seconds is not None
            and word.start_seconds > duration_seconds + DURATION_TOLERANCE_SECONDS
        ):
            raise ValueError(f"ASR word starts beyond the full recording: {word.text}")
        if word.end_seconds is not None and word.end_seconds < 0.0:
            raise ValueError(f"ASR word ends before the full recording: {word.text}")
        if (
            word.start_seconds is not None
            and word.end_seconds is not None
            and word.end_seconds < word.start_seconds
        ):
            raise ValueError(f"ASR word ends before it starts: {word.text}")
        if (
            word.end_seconds is not None
            and word.end_seconds > duration_seconds + DURATION_TOLERANCE_SECONDS
        ):
            raise ValueError(f"ASR word ends beyond the full recording: {word.text}")


def requested_results(
    requested: Sequence[AsrModelId],
    results: Sequence[AsrTranscriptResult],
) -> tuple[AsrTranscriptResult, ...]:
    result_models = [result.model_id for result in results]
    if len(result_models) != len(set(result_models)):
        raise ValueError("Remote full-recording ASR response contains duplicate model results.")
    result_by_model = {result.model_id: result for result in results}
    missing = tuple(model for model in requested if model not in result_by_model)
    if missing:
        labels = ", ".join(model.value for model in missing)
        raise ValueError(f"Remote full-recording ASR response missing models: {labels}")
    return tuple(result_by_model[model] for model in requested)


def missing_model_ids(
    requested: Sequence[AsrModelId],
    transcripts: Sequence[FullRecordingAsrTranscriptRecord],
) -> tuple[AsrModelId, ...]:
    cached_models = {transcript.model_id for transcript in transcripts}
    return tuple(model for model in requested if model not in cached_models)


def local_audio_path(access_uri: str) -> Path:
    path = Path(access_uri)
    if not path.is_file():
        raise ValueError(f"Full-recording ASR requires a local audio file: {access_uri}")
    return path
