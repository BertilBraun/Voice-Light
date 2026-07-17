from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import httpx

from app.shared.asr import RemoteAsrRequest, RemoteAsrResponse, RemoteAsrUploadRequest
from app.shared.audio.transport import AudioTransportCodec


class HttpRemoteAsrClient:
    def __init__(self, endpoint_url: str, api_key: str) -> None:
        if not endpoint_url:
            raise ValueError("VOICE_LIGHT_COMPUTE_URL is required for uncached ASR.")
        if not api_key:
            raise ValueError("VOICE_LIGHT_COMPUTE_TOKEN is required for uncached ASR.")
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def transcribe(self, request: RemoteAsrRequest) -> RemoteAsrResponse:
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
                f"Remote ASR request failed with HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise ValueError(f"Remote ASR request failed: {error.reason}") from error
        return RemoteAsrResponse.model_validate_json(payload)

    def transcribe_upload(
        self,
        request: RemoteAsrUploadRequest,
        audio_path: Path,
    ) -> RemoteAsrResponse:
        if request.audio.encoded_filename != audio_path.name:
            raise ValueError("ASR upload filename does not match its transport metadata.")
        with audio_path.open("rb") as audio_file:
            response = httpx.post(
                f"{self.endpoint_url}-upload",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=[
                    ("request_json", (None, request.model_dump_json(), "application/json")),
                    (
                        "audio",
                        (
                            request.audio.encoded_filename,
                            audio_file,
                            content_type_for_codec(request.audio.codec),
                        ),
                    ),
                ],
                timeout=900.0,
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ValueError(
                f"Remote ASR upload failed with HTTP {response.status_code}: {response.text}"
            ) from error
        return RemoteAsrResponse.model_validate_json(response.text)


def content_type_for_codec(codec: AudioTransportCodec) -> str:
    match codec:
        case AudioTransportCodec.FLAC:
            return "audio/flac"
        case AudioTransportCodec.OGG_OPUS:
            return "audio/ogg"
