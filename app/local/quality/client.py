from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from app.local.quality.remote_models import (
    AudioSource,
    LocalAudioSource,
    RemoteQualityRequest,
    UriAudioSource,
)
from app.local.quality.transport import prepare_quality_transport_audio
from app.shared.compute_api import QualityAnalysisResponse, QualityAnalysisUpload
from app.shared.quality import AudioMetadata


class RemoteQualityClient(Protocol):
    def analyze(self, request: RemoteQualityRequest) -> QualityAnalysisResponse: ...


@dataclass(frozen=True)
class PreparedQualityAudio:
    path: Path
    original_metadata: AudioMetadata | None


class HttpRemoteQualityClient:
    def __init__(self, endpoint_url: str, api_key: str) -> None:
        if not endpoint_url:
            raise ValueError("VOICE_LIGHT_COMPUTE_URL is required for ingestion.")
        if not api_key:
            raise ValueError("VOICE_LIGHT_COMPUTE_TOKEN is required for ingestion.")
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def analyze(self, request: RemoteQualityRequest) -> QualityAnalysisResponse:
        with tempfile.TemporaryDirectory(prefix="voice-light-quality-transport-") as directory_name:
            directory = Path(directory_name)
            speaker1 = prepare_audio_source(request.speaker1, directory, 1)
            speaker2 = prepare_audio_source(request.speaker2, directory, 2)
            upload = QualityAnalysisUpload(
                sample_id=request.sample_id,
                speaker1_original_metadata=speaker1.original_metadata,
                speaker2_original_metadata=speaker2.original_metadata,
            )
            with (
                speaker1.path.open("rb") as speaker1_file,
                speaker2.path.open("rb") as speaker2_file,
            ):
                response = httpx.post(
                    self.endpoint_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=[
                        (
                            "request_json",
                            (None, upload.model_dump_json(), "application/json"),
                        ),
                        ("speaker1", (speaker1.path.name, speaker1_file, "audio/flac")),
                        ("speaker2", (speaker2.path.name, speaker2_file, "audio/flac")),
                    ],
                    timeout=900.0,
                )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ValueError(
                f"Remote quality request failed with HTTP {response.status_code}: {response.text}"
            ) from error
        return QualityAnalysisResponse.model_validate_json(response.text)


def prepare_audio_source(
    source: AudioSource,
    directory: Path,
    speaker_index: int,
) -> PreparedQualityAudio:
    input_path = materialize_audio_source(source, directory, speaker_index)
    output_path = directory / f"speaker{speaker_index}.flac"
    prepare_quality_transport_audio(input_path, output_path)
    match source:
        case LocalAudioSource():
            original_metadata = source.original_metadata
        case UriAudioSource():
            original_metadata = None
    return PreparedQualityAudio(path=output_path, original_metadata=original_metadata)


def materialize_audio_source(
    source: AudioSource,
    directory: Path,
    speaker_index: int,
) -> Path:
    match source:
        case LocalAudioSource():
            return Path(source.path)
        case UriAudioSource():
            path = directory / f"speaker{speaker_index}.input"
            with httpx.stream("GET", source.uri, timeout=300.0) as response:
                response.raise_for_status()
                with path.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
            return path
