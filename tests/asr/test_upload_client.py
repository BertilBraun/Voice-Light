from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.local.asr.client import HttpRemoteAsrClient
from app.shared.asr import AsrModelId, RemoteAsrResponse, RemoteAsrUploadRequest
from app.shared.audio.transport import AudioTransportCodec, AudioTransportMetadata


def test_asr_upload_client_uses_binary_multipart_without_base64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_bytes = b"OggS-binary-opus"
    audio_path = tmp_path / "sample.ogg"
    audio_path.write_bytes(audio_bytes)
    request = RemoteAsrUploadRequest(
        audio=AudioTransportMetadata(
            original_filename="sample.wav",
            encoded_filename=audio_path.name,
            sha256="0" * 64,
            codec=AudioTransportCodec.OGG_OPUS,
            duration_seconds=1.0,
            sample_rate=16_000,
            channels=1,
            size_bytes=len(audio_bytes),
        ),
        models=(AsrModelId.PARAKEET_TDT,),
    )
    recorded_request_json: list[str] = []
    recorded_audio: list[bytes] = []

    def fake_post(
        url: str,
        headers: dict[str, str],
        files: list[tuple[str, tuple[str | None, object, str]]],
        timeout: float,
    ) -> httpx.Response:
        assert url == "https://compute.example/v1/asr:batch-upload"
        assert headers == {"Authorization": "Bearer token"}
        assert timeout == 900.0
        request_part = files[0][1]
        audio_part = files[1][1]
        assert isinstance(request_part[1], str)
        recorded_request_json.append(request_part[1])
        audio_file = audio_part[1]
        assert hasattr(audio_file, "read")
        recorded_audio.append(audio_file.read())
        return httpx.Response(
            status_code=200,
            json={"results": []},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = HttpRemoteAsrClient(
        endpoint_url="https://compute.example/v1/asr:batch",
        api_key="token",
    )

    response = client.transcribe_upload(request=request, audio_path=audio_path)

    assert response == RemoteAsrResponse(results=())
    assert recorded_audio == [audio_bytes]
    request_payload = json.loads(recorded_request_json[0])
    assert "audio_base64" not in request_payload
    assert request_payload["audio"]["codec"] == "ogg_opus"
