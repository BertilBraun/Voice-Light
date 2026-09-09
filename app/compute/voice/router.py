from __future__ import annotations

import os

from fastapi import WebSocket

from app.compute.runtime import ComputeRuntime
from app.compute.telemetry import RequestIdScope
from app.compute.voice.playback import PlaybackPolicyConfig
from app.compute.voice.search import (
    ConfiguredTavilySearchSettings,
    QwenSearchResultSummarizer,
    SearchPipeline,
)
from app.compute.voice.session import SessionPolicy, VoiceSession
from app.compute.voice.tools import StandardSearchHandler, create_runtime_tool_registry


async def run_voice_session(
    websocket: WebSocket,
    runtime: ComputeRuntime,
    request_id: str,
) -> None:
    with RequestIdScope(request_id):
        language_model = runtime.require_language_model()
        voice_stack_settings = runtime.voice_stack_settings
        assert voice_stack_settings is not None
        match voice_stack_settings.search:
            case ConfiguredTavilySearchSettings():
                search_handler = StandardSearchHandler(
                    SearchPipeline(
                        provider=runtime.require_search_provider(),
                        summarizer=QwenSearchResultSummarizer(
                            runtime.require_search_text_generator()
                        ),
                    )
                )
            case _:
                search_handler = None
        session = VoiceSession(
            websocket=websocket,
            speech_detector=runtime.require_speech_detector_factory().create(),
            speech_understanding_provider=runtime.require_speech_understanding_provider(),
            language_model=language_model,
            speech_synthesizer=runtime.require_speech_synthesizer(),
            policy=SessionPolicy.from_environment(os.environ),
            playback_policy=PlaybackPolicyConfig.from_environment(os.environ),
            tool_executor=create_runtime_tool_registry(search_handler=search_handler),
        )
        await session.run()
