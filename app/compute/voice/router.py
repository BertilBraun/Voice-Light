from __future__ import annotations

from fastapi import WebSocket

from app.compute.runtime import ComputeRuntime
from app.compute.telemetry import RequestIdScope
from app.compute.voice.session import SessionPolicy, VoiceSession
from app.compute.voice.speech_detection import SileroSpeechDetector


async def run_voice_session(
    websocket: WebSocket,
    runtime: ComputeRuntime,
    request_id: str,
) -> None:
    with RequestIdScope(request_id):
        session = VoiceSession(
            websocket=websocket,
            speech_detector=SileroSpeechDetector(),
            transcriber=runtime.require_streaming_asr(),
            language_model=runtime.require_language_model(),
            speech_synthesizer=runtime.require_speech_synthesizer(),
            policy=SessionPolicy(),
        )
        await session.run()
