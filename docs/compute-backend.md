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
VAD, genuine streaming Nemotron state, turn decisions, conversation history, Qwen/Kyutai work, and
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

The prototype uses `kyutai/tts-1.6b-en_fr` through Moshi's direct PyTorch runtime. The checkpoint,
Mimi codec, and fixed precomputed voice are loaded once. Each assistant turn creates fresh delayed-
stream LM and codec state, accepts complete words incrementally, and emits PCM frames and model-
predicted word-start timestamps incrementally. Batch size is always one. Cancellation signals the
generation worker immediately; a lock prevents a canceled worker and its successor from using the
shared model concurrently. Startup performs one discarded streaming utterance so the first user
turn does not pay CUDA kernel warm-up costs.

The Moshi package's conservative dependency ceilings lag this repository's Torch, Hugging Face Hub,
and Safetensors versions. `tool.uv.override-dependencies` explicitly selects the repository's newer
shared stack. This combination and the 32-codebook quality setting must be validated on the rented
GPU before deployment. If memory or throughput is insufficient, codebook count is the first
benchmark variable; lowering it trades audio quality for speed and does not change the protocol.

`deployment/compute/benchmark_tts.py` measures cold/warm first-chunk latency, total generation time,
real-time factor, output duration, and emitted boundary count on the rented machine.
