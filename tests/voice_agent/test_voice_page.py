from __future__ import annotations

from fastapi.testclient import TestClient

from app.local.main import app


def test_voice_page_exposes_streaming_conversation_history() -> None:
    with TestClient(app) as client:
        page_response = client.get("/voice-agent")
        script_response = client.get("/pages/voice-agent/app.js")
        worklet_response = client.get("/pages/voice-agent/playback-worklet.js")

    assert page_response.status_code == 200
    assert 'id="conversation-history"' in page_response.text
    assert 'id="conversation-empty"' in page_response.text
    assert 'id="user-transcript"' not in page_response.text
    assert 'id="assistant-transcript"' not in page_response.text
    assert 'message.type === "turn.committed"' in script_response.text
    assert 'message.type === "llm.history"' in script_response.text
    assert "console.table(message.messages)" in script_response.text
    assert 'message.type === "assistant.text.delta"' in script_response.text
    assert 'message.type === "assistant.cancel"' in script_response.text
    assert 'message.type === "assistant.audio.sentence"' in script_response.text
    assert 'type: "playback.progress"' in script_response.text
    assert "turn-unspoken" in script_response.text
    assert "characterCount: message.text_end - message.text_start" in script_response.text
    assert "playback.characterCount * playedSamples" in worklet_response.text
    assert "characterOffset === playback.reportedCharacterOffset" in worklet_response.text
    assert 'data.type === "sentence" && data.generationId > this.cancelledGenerationId' in (
        worklet_response.text
    )
