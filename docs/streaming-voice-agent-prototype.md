# Streaming Voice Agent Prototype

The prototype page is available at `http://127.0.0.1:8000/voice-agent`. The browser connects to
the local `/api/voice/session` WebSocket; users never enter or receive the compute URL or token.

## Pipeline

- The local application keeps the microphone active, runs Silero VAD continuously, owns the
  ordered conversation history, and cancels playback immediately on barge-in.
- One authenticated local-to-compute WebSocket multiplexes typed streaming operations.
- NVIDIA Nemotron Speech Streaming English 0.6B receives 80 ms PCM16 chunks and retains its real
  cache-aware RNNT state until the utterance is finalized.
- Qwen3-1.7B receives the complete ordered conversation and streams non-thinking text deltas. There
  are no artificial word-count or sentence-count prompt restrictions.
- Completed sentence chunks enter Pocket TTS immediately. Its official iterator returns real 24 kHz
  PCM chunks rather than slicing a completed waveform after synthesis.
- Browser capture continues during playback. A new speech-start event cancels local generation,
  sends cancellation upstream, and clears queued browser playback by generation identifier.

Readiness is distinct from connectivity. The compute process becomes live before model loading is
complete; `/health/ready` returns 503 with per-model stages until Nemotron, Qwen, and Pocket TTS are
loaded. The voice WebSocket rejects sessions while the backend is unready.

Binary local-to-browser audio messages begin with two little-endian unsigned 32-bit integers: the
generation ID and sequence number. Remaining bytes are mono PCM16 audio.

## Run

On the rented Ubuntu machine, follow the commands in the README. On the local machine, set:

```text
VOICE_LIGHT_COMPUTE_URL=http://<vast-ip>:8000
VOICE_LIGHT_COMPUTE_TOKEN=<matching bearer token>
```

Then run `uv run python -m app.local.server` and open the prototype page. TLS, DNS, instance lifecycle,
and storage provisioning remain deliberately outside this prototype.
