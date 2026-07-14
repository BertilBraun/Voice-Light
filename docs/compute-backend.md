# Focused compute backend

Voice Light is split into two deployable applications:

- The local application (`app.local.main`) owns PostgreSQL, persistent metadata, the website, and
  non-voice browser-facing APIs.
- The compute application (`app.compute.main`) owns disposable model processes and caches. It has
  no database dependency. Its voice WebSocket owns the complete ephemeral live session.

The Python package mirrors that deployment boundary:

- `app/local/` contains analyses, batch-ASR orchestration, database access, ingestion, the dataset
  dashboard, and web assets.
- `app/compute/` contains batch-ASR model execution, quality scoring, and streaming ASR/LLM/TTS
  execution.
- `app/shared/` contains only typed compute API models and reusable audio, quality, and storage
  primitives needed on both sides.
- `app/training/` remains an offline application and does not participate in either runtime.

Compute and shared modules must never import local modules. Local modules call compute HTTP APIs
through the schemas in `app.shared` and their authenticated clients.

The browser connects directly to `/v1/voice`. One compute-side `VoiceSession` owns server Silero
VAD, genuine streaming Nemotron state, turn decisions, conversation history, Qwen/Pocket work, and
cancellation for the life of that WebSocket. The protocol is optimized for this specific cascade,
not for provider-neutral ASR/LLM/TTS operations. No voice state survives disconnection.

## Compute API

Compute HTTP endpoints except liveness require `Authorization: Bearer <token>`. The research voice
WebSocket is deliberately public for direct use by the browser.

- `GET /health/live`: process liveness; does not imply that models are ready.
- `GET /health/ready`: ready only after the streaming ASR, LLM, and TTS startup stages complete.
- `POST /v1/asr:batch`: batch ASR with request-contained base64 audio.
- `POST /v1/quality:analyze`: two multipart audio uploads plus a sample identifier.
- `WS /v1/voice`: focused, ephemeral cascaded voice session with binary PCM in both directions.

The compute server never accepts filesystem paths or shell commands from HTTP clients. Quality
uploads are decoded in request-scoped temporary storage and deleted after the response.

## TTS decision

The prototype uses Kyutai Pocket TTS 2.1 with its `generate_audio_stream` iterator. It is a current,
small English model with a documented roughly 200 ms first chunk, faster-than-real-time CPU
generation, reusable voice state, and rapid abandonment when a local generation is cancelled. Its
CPU execution leaves the RTX 4090 VRAM available to Nemotron and Qwen and it uses the same current
Torch dependency rather than forcing the old Transformers environment required by CosyVoice.

Qwen3-TTS 0.6B is the strongest future quality challenger. It advertises a 97 ms streaming
architecture, but its current official Python distribution pins Transformers 4.x while Nemotron's
current implementation needs Transformers 5.x, and its supported Python generation methods return
complete waveforms. Chatterbox Turbo also exposes whole-utterance generation in its official API.
Kyutai's larger delayed-stream model is genuinely streaming but adds substantially more model and
runtime complexity than this prototype needs.

`deployment/compute/benchmark_tts.py` measures cold/warm first-chunk latency, total generation time,
real-time factor, and output duration on the rented machine so the choice can be revisited with
hardware evidence.
