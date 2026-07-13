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
- Whisper Base English transcribes the complete committed turn.
- Silero uses a 0.4 speech threshold with a 0.25 release threshold, retains 300 ms of pre-roll,
  and commits about 500 ms after Silero reports the end of speech.
- Qwen3-1.7B generates a short, deterministic, non-thinking response with the full in-session
  conversation history.
- Each completed sentence is sent to CosyVoice 2 0.5B immediately. The reference-speaker features
  are computed once at startup and reused across synthesis calls. CosyVoice streams 24 kHz PCM audio
  over the WebSocket.
- Browser capture pauses during assistant playback to prevent acoustic echo from creating false
  user turns. This prototype therefore does not support barge-in once playback has begun.

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

The deployed class scales to zero, admits one session at a time, and releases the L40S container 30
seconds after the last session ends. A new session after scale-down incurs the model-loading cold
start; an open WebSocket remains active and therefore does not begin the scale-down window. The
Hugging Face, ModelScope, and Torch caches share the persistent `voice-light-agent-model-cache`
volume, so cold starts reload cached files instead of downloading them again.

## Current Limitation

The `BufferedWhisperTranscriber` runs one accumulated-audio transcription after VAD commits the
turn. VAD ingestion continues concurrently while finalized turns are transcribed, but Whisper does
not produce partial transcripts. It is a reliability baseline that keeps CosyVoice on its supported
Transformers and Torch versions. A cache-aware streaming ASR should run in a separately pinned
service before treating ASR latency as representative of the intended architecture.

The session orchestration, VAD, LLM token stream, CosyVoice synthesis, audio stream, and browser
playback do not depend on that replacement.
