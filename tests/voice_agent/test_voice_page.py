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
    assert 'message.type === "llm.model_request"' in script_response.text
    assert "console.table(message.messages)" in script_response.text
    assert "messages: message.messages, tools: message.tools" in script_response.text
    assert "recordedInputChunks.push(data.slice(0))" in script_response.text
    assert "createPcmWav(recordedInputChunks, INPUT_SAMPLE_RATE)" in script_response.text
    assert "new AudioContext({ sampleRate: INPUT_SAMPLE_RATE })" in script_response.text
    assert "capture-worklet.js?v=2" in script_response.text
    assert "processorOptions: { targetSampleRate: INPUT_SAMPLE_RATE }" in script_response.text
    assert 'console.info("Voice input capture"' in script_response.text
    assert 'message.type === "assistant.text.delta"' in script_response.text
    assert 'message.type === "assistant.cancel"' in script_response.text
    assert 'message.type === "assistant.audio.text_boundary"' in script_response.text
    assert 'message.type === "playback.command"' in script_response.text
    assert 'data.type === "playback.started"' in script_response.text
    assert 'data.type === "playback.acknowledgement"' in script_response.text
    assert 'type: "playback.started"' in script_response.text
    assert 'type: "playback.progress"' in script_response.text
    assert 'type: "playback.acknowledgement"' in script_response.text
    assert "turn-unspoken" in script_response.text
    assert "textOffset: message.text_offset" in script_response.text
    assert "this.sourceSamplePosition >= this.boundaries[0].startSample" in worklet_response.text
    assert 'type: "playback.started"' in worklet_response.text
    assert 'case "playback.command":' in worklet_response.text
    assert "PlaybackState.PAUSED_BUFFERED" in worklet_response.text
    assert "new Int16Array(input.length)" in capture_worklet_response.text
    assert "input[index]" in capture_worklet_response.text
    assert "sourcePosition" not in capture_worklet_response.text
