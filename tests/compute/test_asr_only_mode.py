from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

pytest.importorskip("silero_vad", reason="Compute runtime tests require compute dependencies.")

from app.compute.asr.models.registry import AsrModelCache
from app.compute.config import ComputeSettings
from app.compute.main import create_compute_app
from app.compute.runtime import ComputeRuntime
from app.shared.asr import RemoteAsrRequest, RemoteAsrResponse


def test_asr_only_runtime_skips_voice_model_loading(tmp_path: Path) -> None:
    runtime = ComputeRuntime(
        voice_stack_settings=None,
        dataset_audio_cache_directory=tmp_path / "dataset-audio",
    )

    runtime.start_loading()

    assert runtime.loading_task is None
    assert runtime.ready
    assert not runtime.voice_enabled
    assert not runtime.voice_ready
    assert runtime.stages() == ()
    assert runtime.speech_detector_factory is None
    assert runtime.speech_understanding_provider is None
    assert runtime.language_model is None
    assert runtime.speech_synthesizer is None


def test_asr_only_app_is_ready_and_serves_batch_asr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_response = RemoteAsrResponse(results=())

    def transcribe_without_loading_voice_models(
        model_cache: AsrModelCache,
        request: RemoteAsrRequest,
    ) -> RemoteAsrResponse:
        del model_cache, request
        return expected_response

    monkeypatch.setattr(
        "app.compute.main.transcribe_request",
        transcribe_without_loading_voice_models,
    )
    application = create_compute_app(
        ComputeSettings(
            token="secret-token",
            log_directory=tmp_path / "logs",
            dataset_audio_cache_directory=tmp_path / "dataset-audio",
            voice_stack=None,
        )
    )
    headers = {"Authorization": "Bearer secret-token"}

    with TestClient(application) as client:
        readiness = client.get("/health/ready", headers=headers)
        response = client.post(
            "/v1/asr:batch",
            headers=headers,
            json={
                "audio_sha256": "0" * 64,
                "audio_filename": "sample.wav",
                "audio_base64": "",
                "models": ["canary_1b_v2"],
            },
        )
        with pytest.raises(WebSocketDisconnect) as websocket_error:
            with client.websocket_connect("/v1/voice"):
                pass

    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"
    assert readiness.json()["stages"] == []
    assert response.status_code == 200
    assert response.json() == expected_response.model_dump(mode="json")
    assert websocket_error.value.code == 1013
    assert websocket_error.value.reason == "Voice stack is disabled."
