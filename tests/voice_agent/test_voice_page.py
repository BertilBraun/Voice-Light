from __future__ import annotations

from fastapi.testclient import TestClient

from app.local.main import app


def test_voice_page_exposes_streaming_conversation_history() -> None:
    with TestClient(app) as client:
        page_response = client.get("/voice-agent")
        script_response = client.get("/pages/voice-agent/app.js")
        capture_worklet_response = client.get("/pages/voice-agent/capture-worklet.js")
        worklet_response = client.get("/pages/voice-agent/playback-worklet.js")

    assert page_response.status_code == 200
    assert 'id="conversation-history"' in page_response.text
    assert 'id="conversation-empty"' in page_response.text
    assert 'id="recording-player"' in page_response.text
    assert 'id="recording-download"' in page_response.text
    assert 'id="user-transcript"' not in page_response.text
    assert 'id="assistant-transcript"' not in page_response.text
    assert 'message.type === "turn.committed"' in script_response.text
    assert 'message.type === "llm.history"' in script_response.text
    assert "console.table(message.messages)" in script_response.text
    assert "JSON.stringify(message.messages, null, 2)" in script_response.text
    assert "recordedInputChunks.push(data.slice(0))" in script_response.text
    assert "createPcmWav(recordedInputChunks, INPUT_SAMPLE_RATE)" in script_response.text
    assert "new AudioContext({ sampleRate: INPUT_SAMPLE_RATE })" in script_response.text
    assert "capture-worklet.js?v=2" in script_response.text
    assert "processorOptions: { targetSampleRate: INPUT_SAMPLE_RATE }" in script_response.text
    assert 'console.info("Voice input capture"' in script_response.text
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
    assert "new Int16Array(input.length)" in capture_worklet_response.text
    assert "input[index]" in capture_worklet_response.text
    assert "sourcePosition" not in capture_worklet_response.text
