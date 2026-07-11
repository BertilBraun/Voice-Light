from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.asr.schemas import RemoteAsrRequest, RemoteAsrResponse


class HttpRemoteAsrClient:
    def __init__(self, endpoint_url: str, api_key: str) -> None:
        if not endpoint_url:
            raise ValueError("VOICE_LIGHT_REMOTE_ASR_ENDPOINT_URL is required for uncached ASR.")
        if not api_key:
            raise ValueError("VOICE_LIGHT_REMOTE_ASR_API_KEY is required for uncached ASR.")
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
