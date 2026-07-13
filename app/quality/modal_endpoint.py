from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse
from urllib.request import urlopen

import modal
from fastapi import Header, HTTPException

from app.audio import load_audio
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
from app.quality.service import score_two_track_sample
from app.storage.local import LocalStorageBackend

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "av>=17.1.0",
        "fastapi[standard]>=0.115.0",
        "numpy>=2.0.0",
        "pydantic>=2.0.0",
    )
    .add_local_python_source("app")
)

app = modal.App("VoiceLightQuality")
quality_input_mounts = tuple(
    Path(f"/quality-inputs/{volume_index}") for volume_index in range(QUALITY_INPUT_VOLUME_COUNT)
)
quality_input_volumes = tuple(
    modal.Volume.from_name(
        quality_input_volume_name(volume_index),
        create_if_missing=True,
        version=2,
    )
    for volume_index in range(QUALITY_INPUT_VOLUME_COUNT)
)


@app.function(
    image=modal_image,
    cpu=2.0,
    memory=4096,
    timeout=900,
    max_containers=20,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("voice-light-asr")],
    volumes={
        mount.as_posix(): volume
        for mount, volume in zip(quality_input_mounts, quality_input_volumes, strict=True)
    },
)
@modal.concurrent(max_inputs=1)
@modal.fastapi_endpoint(method="POST")
def analyze(
    request: RemoteQualityRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> RemoteQualityResponse:
    authorize_request(authorization)
    reload_request_volumes(request)
    with tempfile.TemporaryDirectory(prefix="voice-light-quality-") as temporary_directory:
        directory = Path(temporary_directory)
        speaker1_path = materialize_audio_source(request.speaker1, directory, "speaker1")
        speaker2_path = materialize_audio_source(request.speaker2, directory, "speaker2")
        storage = LocalStorageBackend()
        speaker1_audio = load_audio(storage, speaker1_path.as_posix())
        speaker2_audio = load_audio(storage, speaker2_path.as_posix())
        quality_result = score_two_track_sample(
            sample_id=request.sample_id,
            speaker1_audio=speaker1_audio,
            speaker2_audio=speaker2_audio,
            speaker1_uri="speaker1",
            speaker2_uri="speaker2",
        )
        return RemoteQualityResponse(
            speaker1_metadata=speaker1_audio.metadata,
            speaker2_metadata=speaker2_audio.metadata,
            quality_result=quality_result,
        )


def materialize_audio_source(source: AudioSource, directory: Path, stem: str) -> Path:
    match source:
        case VolumeAudioSource():
            relative_path = Path(source.path.lstrip("/\\"))
            if ".." in relative_path.parts:
                raise HTTPException(status_code=400, detail="Invalid quality input volume path.")
            path = quality_input_mounts[source.volume_index] / relative_path
            if not path.is_file():
                raise HTTPException(status_code=400, detail="Quality input file was not found.")
            return path
        case UriAudioSource():
            parsed_uri = urlparse(source.uri)
            if parsed_uri.scheme not in {"http", "https"}:
                raise HTTPException(
                    status_code=400,
                    detail="Audio URI must use HTTP or HTTPS, such as a presigned S3 URL.",
                )
            suffix = Path(parsed_uri.path).suffix or ".audio"
            path = directory / f"{stem}{suffix}"
            with urlopen(source.uri, timeout=300) as response, path.open("wb") as output:
                shutil.copyfileobj(response, output)
            return path
        case LocalAudioSource():
            raise HTTPException(
                status_code=400, detail="Local audio must be staged before request."
            )


def reload_request_volumes(request: RemoteQualityRequest) -> None:
    volume_indexes = {
        volume_index
        for source in (request.speaker1, request.speaker2)
        for volume_index in volume_indexes_for_source(source)
    }
    for volume_index in volume_indexes:
        quality_input_volumes[volume_index].reload()


def volume_indexes_for_source(source: AudioSource) -> tuple[int, ...]:
    match source:
        case VolumeAudioSource():
            return (source.volume_index,)
        case LocalAudioSource() | UriAudioSource():
            return ()


def authorize_request(authorization: str | None) -> None:
    expected_api_key = os.environ["VOICE_LIGHT_REMOTE_ASR_API_KEY"]
    if authorization != f"Bearer {expected_api_key}":
        raise HTTPException(status_code=401, detail="Invalid quality API key.")
