from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from app.shared.asr import (
    AsrModelId,
    AsrTranscriptResult,
    CachedAsrResponse,
    RemoteAsrResponse,
    RemoteAsrUploadRequest,
)
from app.shared.audio.transport import (
    ASR_AUDIO_TRANSPORT_SPEC,
    prepare_audio_transport,
    sha256_file,
)


class AsrTranscriptCache(Protocol):
    def get_cached_asr_transcripts(
        self,
        audio_sha256: str,
        model_ids: Sequence[AsrModelId],
    ) -> tuple[AsrTranscriptResult, ...]: ...

    def upsert_cached_asr_transcript(
        self,
        audio_sha256: str,
        audio_filename: str,
        transcript: AsrTranscriptResult,
    ) -> AsrTranscriptResult: ...


class RemoteAsrClient(Protocol):
    def transcribe_upload(
        self,
        request: RemoteAsrUploadRequest,
        audio_path: Path,
    ) -> RemoteAsrResponse: ...


RemoteAsrClientFactory = Callable[[], RemoteAsrClient]


def cached_asr_transcripts(
    audio_path: Path,
    requested_models: Sequence[AsrModelId],
    cache: AsrTranscriptCache,
    remote_client_factory: RemoteAsrClientFactory,
) -> CachedAsrResponse:
    model_ids = unique_model_ids(requested_models)
    if not model_ids:
        raise ValueError("At least one ASR model is required.")
    if not audio_path.is_file():
        raise ValueError(f"Audio file not found: {audio_path}")

    audio_sha256 = sha256_file(audio_path)
    cached_results = cache.get_cached_asr_transcripts(audio_sha256, model_ids)
    cached_model_ids = {result.model_id for result in cached_results}
    missing_model_ids = tuple(
        model_id for model_id in model_ids if model_id not in cached_model_ids
    )

    if missing_model_ids:
        with tempfile.TemporaryDirectory(prefix="voice-light-asr-transport-") as directory_name:
            transport_path = Path(directory_name) / f"{audio_path.stem}.ogg"
            transport_metadata = prepare_audio_transport(
                source_path=audio_path,
                output_path=transport_path,
                spec=ASR_AUDIO_TRANSPORT_SPEC,
            )
            remote_response = remote_client_factory().transcribe_upload(
                request=RemoteAsrUploadRequest(
                    audio=transport_metadata,
                    models=missing_model_ids,
                ),
                audio_path=transport_path,
            )
        persist_remote_results(
            audio_sha256=audio_sha256,
            audio_filename=audio_path.name,
            requested_model_ids=missing_model_ids,
            response=remote_response,
            cache=cache,
        )

    final_results = cache.get_cached_asr_transcripts(audio_sha256, model_ids)
    return CachedAsrResponse(
        audio_sha256=audio_sha256,
        results=ordered_results(requested_model_ids=model_ids, results=final_results),
    )


def unique_model_ids(requested_models: Sequence[AsrModelId]) -> tuple[AsrModelId, ...]:
    model_ids: list[AsrModelId] = []
    for model_id in requested_models:
        if model_id not in model_ids:
            model_ids.append(model_id)
    return tuple(model_ids)


def persist_remote_results(
    audio_sha256: str,
    audio_filename: str,
    requested_model_ids: Sequence[AsrModelId],
    response: RemoteAsrResponse,
    cache: AsrTranscriptCache,
) -> None:
    response_model_ids = {result.model_id for result in response.results}
    missing_response_model_ids = tuple(
        model_id for model_id in requested_model_ids if model_id not in response_model_ids
    )
    if missing_response_model_ids:
        labels = ", ".join(model_id.value for model_id in missing_response_model_ids)
        raise ValueError(f"Remote ASR response missing requested model results: {labels}")

    requested_model_id_set = set(requested_model_ids)
    for result in response.results:
        if result.model_id in requested_model_id_set:
            cache.upsert_cached_asr_transcript(
                audio_sha256=audio_sha256,
                audio_filename=audio_filename,
                transcript=result,
            )


def ordered_results(
    requested_model_ids: Sequence[AsrModelId],
    results: Sequence[AsrTranscriptResult],
) -> tuple[AsrTranscriptResult, ...]:
    ordered: list[AsrTranscriptResult] = []
    for model_id in requested_model_ids:
        matching_results = [result for result in results if result.model_id == model_id]
        if not matching_results:
            raise ValueError(
                f"Missing cached ASR result after inference for model: {model_id.value}"
            )
        ordered.append(matching_results[0])
    return tuple(ordered)
