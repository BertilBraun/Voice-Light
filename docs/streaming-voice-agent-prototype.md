# Streaming Voice Agent Prototype

The prototype page is available at:

```text
http://127.0.0.1:8000/voice-agent
```

Its deployed Modal WebSocket endpoint is:

```text
wss://bertil-braun-private--voicelightagent-voiceagentserver-web-app.modal.run/session
```

Paste that URL into the page and select **Start microphone**. The browser sends 16 kHz mono
PCM16 frames and plays the 24 kHz PCM16 chunks returned by CosyVoice.

## Pipeline

- Silero VAD processes microphone audio continuously.
- Nemotron Speech Streaming English 0.6B produces accumulated-audio partial transcripts.
- A 400 ms VAD silence window commits the turn.
- Qwen3-1.7B generates a non-thinking response through a token streamer.
- Text is sent to CosyVoice at eight words, or at six or seven words when the chunk ends a
  sentence. Only the last chunk may contain fewer than six words.
- CosyVoice 3 0.5B accepts the text generator and streams 24 kHz PCM audio back over the same
  WebSocket.

Binary server messages begin with two little-endian unsigned 32-bit integers: generation ID and
sequence number. The remaining bytes are mono PCM16 audio. Generation IDs let the browser discard
audio from a cancelled response.

## Deploy

The Modal image is large because CosyVoice includes compiled and CUDA dependencies. Build layers
are cached after the first deployment.

```powershell
$env:PYTHONUTF8='1'
uv run modal deploy .\app\voice_agent\modal_endpoint.py
```

The deployed class scales to zero, admits one session at a time, and keeps the L40S container
available for five minutes after the last session ends. A new session after scale-down incurs the
model-loading cold start; an active WebSocket session is not interrupted by the idle window.

## Current Limitation

The `BufferedNemotronTranscriber` implements the streaming interface but transcribes accumulated
audio every 800 ms. It does not yet retain Nemotron's encoder/RNNT cache. Replace this adapter with
the Transformers input-feature generator described by the Nemotron model card before treating ASR
latency or GPU utilization as representative of the intended architecture.

The session orchestration, VAD, LLM token stream, CosyVoice text generator, audio stream, and browser
playback do not depend on that replacement.
