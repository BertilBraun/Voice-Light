from __future__ import annotations

import os

from fastapi import WebSocket

from app.compute.runtime import ComputeRuntime
from app.compute.telemetry import RequestIdScope
from app.compute.voice.session import SessionPolicy, VoiceSession


async def run_voice_session(
    websocket: WebSocket,
    runtime: ComputeRuntime,
    request_id: str,
) -> None:
    with RequestIdScope(request_id):
        session = VoiceSession(
            websocket=websocket,
            speech_detector=runtime.require_speech_detector_factory().create(),
            transcriber=runtime.require_streaming_asr(),
            language_model=runtime.require_language_model(),
            speech_synthesizer=runtime.require_speech_synthesizer(),
            policy=SessionPolicy.from_environment(os.environ),
        )
        await session.run()
