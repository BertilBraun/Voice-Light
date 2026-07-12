from __future__ import annotations

import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.quality.remote_models import RemoteQualityRequest, RemoteQualityResponse


class RemoteQualityClient(Protocol):
    def analyze(self, request: RemoteQualityRequest) -> RemoteQualityResponse: ...


class HttpRemoteQualityClient:
    def __init__(self, endpoint_url: str, api_key: str) -> None:
        if not endpoint_url:
            raise ValueError("VOICE_LIGHT_REMOTE_QUALITY_ENDPOINT_URL is required for ingestion.")
        if not api_key:
            raise ValueError("VOICE_LIGHT_REMOTE_QUALITY_API_KEY is required for ingestion.")
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def analyze(self, request: RemoteQualityRequest) -> RemoteQualityResponse:
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
