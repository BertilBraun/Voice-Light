from __future__ import annotations

import json
import tempfile
from pathlib import Path
from queue import Queue
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import modal

from app.quality.remote_models import (
    QUALITY_INPUT_VOLUME_COUNT,
    AudioSource,
    LocalAudioSource,
    RemoteQualityRequest,
    RemoteQualityResponse,
    UriAudioSource,
    VolumeAudioSource,
    quality_input_volume_name,
)
from app.quality.transport import prepare_quality_transport_audio

quality_input_volume_slots: Queue[int] = Queue()
for quality_input_volume_index in range(QUALITY_INPUT_VOLUME_COUNT):
    quality_input_volume_slots.put(quality_input_volume_index)


class RemoteQualityClient(Protocol):
    def analyze(self, request: RemoteQualityRequest) -> RemoteQualityResponse: ...


class VolumeUpload(Protocol):
    def put_file(self, local_file: str, remote_path: str) -> None: ...


class HttpRemoteQualityClient:
    def __init__(self, endpoint_url: str, api_key: str) -> None:
        if not endpoint_url:
            raise ValueError("VOICE_LIGHT_REMOTE_QUALITY_ENDPOINT_URL is required for ingestion.")
        if not api_key:
            raise ValueError("VOICE_LIGHT_REMOTE_QUALITY_API_KEY is required for ingestion.")
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def analyze(self, request: RemoteQualityRequest) -> RemoteQualityResponse:
        if not has_local_source(request.speaker1) and not has_local_source(request.speaker2):
            return self._send(request)
        request_identifier = str(uuid4())
        volume_index = quality_input_volume_slots.get()
        volume = modal.Volume.from_name(
            quality_input_volume_name(volume_index),
            create_if_missing=True,
            version=2,
        )
        try:
            staged_request = stage_local_sources(
                request,
                volume,
                volume_index,
                request_identifier,
            )
            return self._send(staged_request)
        finally:
            volume.remove_file(request_identifier, recursive=True)
            quality_input_volume_slots.put(volume_index)

    def _send(self, request: RemoteQualityRequest) -> RemoteQualityResponse:
        request_body = json.dumps(request.model_dump(mode="json")).encode("utf-8")
        http_request = Request(
            self.endpoint_url,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=900) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ValueError(
                f"Remote quality request failed with HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise ValueError(f"Remote quality request failed: {error.reason}") from error
        return RemoteQualityResponse.model_validate_json(payload)


def stage_local_sources(
    request: RemoteQualityRequest,
    volume: modal.Volume,
    volume_index: int,
    request_identifier: str,
) -> RemoteQualityRequest:
    with tempfile.TemporaryDirectory(prefix="voice-light-quality-transport-") as directory_name:
        directory = Path(directory_name)
        prepared_speaker1 = prepare_staging_source(request.speaker1, directory, 1)
        prepared_speaker2 = prepare_staging_source(request.speaker2, directory, 2)
        with volume.batch_upload(force=True) as upload:
            speaker1 = stage_audio_source(
                prepared_speaker1, upload, volume_index, request_identifier, 1
            )
            speaker2 = stage_audio_source(
                prepared_speaker2, upload, volume_index, request_identifier, 2
            )
    return request.model_copy(update={"speaker1": speaker1, "speaker2": speaker2})


def prepare_staging_source(
    source: AudioSource,
    directory: Path,
    speaker_index: int,
) -> AudioSource:
    match source:
        case LocalAudioSource():
            output_path = directory / f"speaker{speaker_index}.flac"
            prepare_quality_transport_audio(Path(source.path), output_path)
            return source.model_copy(
                update={"filename": output_path.name, "path": str(output_path)}
            )
        case UriAudioSource() | VolumeAudioSource():
            return source


def has_local_source(source: AudioSource) -> bool:
    match source:
        case LocalAudioSource():
            return True
        case UriAudioSource() | VolumeAudioSource():
            return False


def stage_audio_source(
    source: AudioSource,
    upload: VolumeUpload,
    volume_index: int,
    request_identifier: str,
    speaker_index: int,
) -> VolumeAudioSource | UriAudioSource:
    match source:
        case LocalAudioSource():
            remote_path = (
                f"/{request_identifier}/speaker{speaker_index}{Path(source.filename).suffix}"
            )
            upload.put_file(source.path, remote_path)
            return VolumeAudioSource(
                volume_index=volume_index,
                path=remote_path.lstrip("/"),
                original_metadata=source.original_metadata,
            )
        case UriAudioSource() | VolumeAudioSource():
            return source
