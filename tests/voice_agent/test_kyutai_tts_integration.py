from __future__ import annotations

import asyncio
import os

import pytest
import torch

pytest.importorskip("moshi", reason="Kyutai TTS integration requires the compute extra.")

from app.compute.voice.interfaces import (
    SynthesisWord,
    SynthesizedAudioChunk,
    SynthesizedWordBoundary,
)
from app.compute.voice.kyutai_tts import KyutaiSpeechSynthesizer


@pytest.mark.integration
def test_kyutai_streams_pcm_and_original_text_boundaries() -> None:
    if os.environ.get("VOICE_LIGHT_RUN_KYUTAI_INTEGRATION") != "1":
        pytest.skip("Set VOICE_LIGHT_RUN_KYUTAI_INTEGRATION=1 to run the model test.")
    if not torch.cuda.is_available():
        pytest.skip("Kyutai TTS integration requires CUDA.")

    asyncio.run(_stream_short_utterance())


async def _stream_short_utterance() -> None:
    synthesizer = await asyncio.to_thread(KyutaiSpeechSynthesizer)
    session = synthesizer.start_session()
    await session.add_word(SynthesisWord(text="Hello,", text_start=0, text_end=6))
    await session.add_word(SynthesisWord(text="world!", text_start=7, text_end=13))
    await session.finish_input()
    audio_sample_count = 0
    boundaries: list[SynthesizedWordBoundary] = []
    async for event in session.stream_events():
        match event:
            case SynthesizedAudioChunk():
                audio_sample_count += len(event.pcm_bytes) // 2
            case SynthesizedWordBoundary():
                boundaries.append(event)
    await session.cancel()

    assert audio_sample_count > 0
    assert [boundary.text_offset for boundary in boundaries] == [6, 13]
    assert all(boundary.start_sample >= 0 for boundary in boundaries)
