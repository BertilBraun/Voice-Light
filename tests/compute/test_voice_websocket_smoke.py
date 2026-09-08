from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient

import app.compute.main as compute_main
from app.compute.config import ComputeSettings, VoiceStackSettings
from app.compute.voice.admission import SingleVoiceSessionAdmission, VoiceSessionLease
from app.compute.voice.schemas import (
    AssistantAudioBoundaryEvent,
    AssistantTextDeltaEvent,
    SessionReadyEvent,
    SpeechStateEvent,
    TranscriptEvent,
    VoiceServerEventType,
)


class FakeComputeRuntime:
    def __init__(
        self,
        voice_stack_settings: VoiceStackSettings | None,
        dataset_audio_cache_directory: Path,
    ) -> None:
        del dataset_audio_cache_directory
        self.voice_enabled = voice_stack_settings is not None
        self.voice_ready = self.voice_enabled
        self.voice_session_admission = SingleVoiceSessionAdmission()

    def start_loading(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def try_admit_voice_session(self, request_id: str) -> VoiceSessionLease | None:
        return self.voice_session_admission.try_acquire(request_id)

    def release_voice_session(self, lease: VoiceSessionLease) -> None:
        self.voice_session_admission.release(lease)


async def run_mock_voice_session(
    websocket: WebSocket,
    runtime: FakeComputeRuntime,
    request_id: str,
) -> None:
    del runtime, request_id
    await websocket.accept()
    assert await websocket.receive_json() == {
        "type": "session.start",
        "input_sample_rate": 16_000,
        "local_time_zone": "Etc/UTC",
    }
    await websocket.send_text(
        SessionReadyEvent(
            session_id="smoke-session",
            input_sample_rate=16_000,
            output_sample_rate=24_000,
        ).model_dump_json()
    )
    assert await websocket.receive_bytes() == b"\x00\x00" * 1_280
    events = (
        SpeechStateEvent(type=VoiceServerEventType.VAD_STARTED, audio_time_ms=80),
        TranscriptEvent(type=VoiceServerEventType.TRANSCRIPT_FINAL, text="hello"),
        TranscriptEvent(type=VoiceServerEventType.TURN_COMMITTED, text="hello"),
        AssistantTextDeltaEvent(generation_id=1, text="Hi there."),
        AssistantAudioBoundaryEvent(
            type=VoiceServerEventType.ASSISTANT_AUDIO_START,
            generation_id=1,
        ),
    )
    for event in events:
        await websocket.send_text(event.model_dump_json())
    await websocket.send_bytes(b"\x00\x00" * 240)
    await websocket.send_text(
        AssistantAudioBoundaryEvent(
            type=VoiceServerEventType.ASSISTANT_AUDIO_END,
            generation_id=1,
        ).model_dump_json()
    )


def test_current_voice_websocket_route_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(compute_main, "ComputeRuntime", FakeComputeRuntime)
    monkeypatch.setattr(compute_main, "run_voice_session", run_mock_voice_session)
    settings = ComputeSettings.from_environment(
        {
            "VOICE_LIGHT_COMPUTE_TOKEN": "smoke-token",
            "VOICE_LIGHT_COMPUTE_LOG_DIR": str(tmp_path / "logs"),
            "VOICE_LIGHT_DATASET_AUDIO_CACHE_DIR": str(tmp_path / "dataset"),
        }
    )

    with TestClient(compute_main.create_compute_app(settings)) as client:
        with client.websocket_connect("/v1/voice") as websocket:
            websocket.send_json(
                {
                    "type": "session.start",
                    "input_sample_rate": 16_000,
                    "local_time_zone": "Etc/UTC",
                }
            )
            assert websocket.receive_json()["type"] == "session.ready"
            websocket.send_bytes(b"\x00\x00" * 1_280)
            assert [websocket.receive_json()["type"] for _ in range(5)] == [
                "vad.started",
                "transcript.final",
                "turn.committed",
                "assistant.text.delta",
                "assistant.audio.start",
            ]
            assert websocket.receive_bytes() == b"\x00\x00" * 240
            assert websocket.receive_json()["type"] == "assistant.audio.end"
