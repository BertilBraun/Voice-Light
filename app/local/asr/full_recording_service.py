from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from app.local.asr.full_recording_models import (
    FullRecordingAsrBatchItem,
    FullRecordingAsrBatchRecord,
    FullRecordingAsrTranscriptRecord,
)
from app.local.asr.service import RemoteAsrClient
from app.shared.asr import AsrModelId, AsrTranscriptResult, RemoteAsrUploadRequest
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


class FullRecordingAsrStore(Protocol):
    def get_batch(self, batch_id: UUID) -> FullRecordingAsrBatchRecord: ...

    def recover_running_items(self, batch_id: UUID) -> int: ...

    def retry_failed_items(self, batch_id: UUID) -> int: ...

    def claim_next_item(self, batch_id: UUID) -> FullRecordingAsrBatchItem | None: ...

    def complete_item(self, item_id: UUID) -> None: ...

    def fail_item(self, item_id: UUID, error: str) -> None: ...

    def finalize_batch(self, batch_id: UUID) -> FullRecordingAsrBatchRecord: ...

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


def run_full_recording_asr_batch(
    batch_id: UUID,
    store: FullRecordingAsrStore,
    remote_client_factory: RemoteAsrClientFactory,
    maximum_tracks: int | None,
    recover_running: bool,
    retry_failed: bool,
) -> FullRecordingAsrBatchRecord:
    if maximum_tracks is not None and maximum_tracks <= 0:
        raise ValueError("maximum_tracks must be positive when supplied.")
    store.get_batch(batch_id)
    if recover_running:
        store.recover_running_items(batch_id)
    if retry_failed:
        store.retry_failed_items(batch_id)
    processed_track_count = 0
    while maximum_tracks is None or processed_track_count < maximum_tracks:
        item = store.claim_next_item(batch_id)
        if item is None:
            break
        try:
            transcribe_full_recording_item(
                item=item,
                store=store,
                remote_client_factory=remote_client_factory,
            )
        except Exception as error:
            store.fail_item(item_id=item.id, error=f"{type(error).__name__}: {error}")
        else:
            store.complete_item(item.id)
        processed_track_count += 1
    return store.finalize_batch(batch_id)


def transcribe_full_recording_item(
    item: FullRecordingAsrBatchItem,
    store: FullRecordingAsrStore,
    remote_client_factory: RemoteAsrClientFactory,
) -> tuple[FullRecordingAsrTranscriptRecord, ...]:
    source_path = local_audio_path(item.access_uri)
    source_audio_sha256 = sha256_file(source_path)
    cached = store.get_cached_transcripts(
        sample_track_id=item.sample_track_id,
        source_audio_sha256=source_audio_sha256,
        model_ids=item.models,
    )
    missing_models = missing_model_ids(requested=item.models, transcripts=cached)
    if not missing_models:
        return cached

    prepared = prepare_full_recording_audio(
        source_path=source_path,
        source_audio_sha256=source_audio_sha256,
    )
    try:
        response = remote_client_factory().transcribe_upload(
            request=RemoteAsrUploadRequest(
                audio=prepared.transport_metadata,
                models=missing_models,
            ),
            audio_path=prepared.path,
        )
        results = requested_results(requested=missing_models, results=response.results)
        for result in results:
            validate_transcript_timestamps(
                transcript=result,
                duration_seconds=prepared.prepared_duration_seconds,
            )
            store.upsert_transcript(
                sample_track_id=item.sample_track_id,
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
        sample_track_id=item.sample_track_id,
        source_audio_sha256=source_audio_sha256,
        model_ids=item.models,
    )
    remaining_models = missing_model_ids(requested=item.models, transcripts=final)
    if remaining_models:
        labels = ", ".join(model.value for model in remaining_models)
        raise ValueError(f"Full-recording ASR persistence is missing models: {labels}")
    return final


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
