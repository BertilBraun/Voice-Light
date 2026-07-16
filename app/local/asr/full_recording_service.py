from __future__ import annotations

import base64
import hashlib
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

import av

from app.local.asr.full_recording_models import (
    FullRecordingAsrBatchItem,
    FullRecordingAsrBatchRecord,
    FullRecordingAsrTranscriptRecord,
)
from app.local.asr.service import RemoteAsrClient
from app.shared.asr import AsrModelId, AsrTranscriptResult, RemoteAsrRequest
from app.shared.audio import probe_local_audio_metadata

FULL_ASR_SAMPLE_RATE = 16_000
DURATION_TOLERANCE_SECONDS = 0.1
HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class PreparedFullRecordingAudio:
    path: Path
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
        prepared_bytes = prepared.path.read_bytes()
        response = remote_client_factory().transcribe(
            RemoteAsrRequest(
                audio_sha256=prepared.prepared_audio_sha256,
                audio_filename=prepared.path.name,
                audio_base64=base64.b64encode(prepared_bytes).decode("ascii"),
                models=missing_models,
            )
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
        suffix=".flac",
        delete=False,
    )
    output_path = Path(output_file.name)
    output_file.close()
    try:
        transcode_mono_flac(source_path=source_path, output_path=output_path)
        prepared_metadata = probe_local_audio_metadata(output_path)
        if prepared_metadata.sample_rate != FULL_ASR_SAMPLE_RATE:
            raise ValueError(
                f"Prepared ASR audio sample rate is {prepared_metadata.sample_rate}, "
                f"expected {FULL_ASR_SAMPLE_RATE}."
            )
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
            source_audio_sha256=source_audio_sha256 or sha256_file(source_path),
            prepared_audio_sha256=sha256_file(output_path),
            source_duration_seconds=source_duration_seconds,
            prepared_duration_seconds=prepared_metadata.duration_seconds,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def transcode_mono_flac(source_path: Path, output_path: Path) -> None:
    with av.open(str(source_path)) as input_container:
        if not input_container.streams.audio:
            raise ValueError(f"Audio stream not found: {source_path}")
        input_stream = input_container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=FULL_ASR_SAMPLE_RATE)
        with av.open(str(output_path), mode="w", format="flac") as output_container:
            output_stream = output_container.add_stream("flac", rate=FULL_ASR_SAMPLE_RATE)
            output_stream.layout = "mono"
            for decoded_frame in input_container.decode(input_stream):
                for resampled_frame in resampler.resample(decoded_frame):
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)
            for resampled_frame in resampler.resample(None):
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)
            for packet in output_stream.encode(None):
                output_container.mux(packet)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def local_audio_path(access_uri: str) -> Path:
    path = Path(access_uri)
    if not path.is_file():
        raise ValueError(f"Full-recording ASR requires a local audio file: {access_uri}")
    return path
