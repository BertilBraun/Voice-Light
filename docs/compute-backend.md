# Provider-neutral compute backend

Voice Light is split into two deployable applications:

- The local application (`app.local.main`) owns PostgreSQL, persistent metadata, the website,
  browser-facing APIs, voice sessions, ordered conversation history, and orchestration.
- The compute application (`app.compute.main`) owns disposable model processes and caches. It has
  no database dependency and exposes authenticated, typed compute APIs.

The Python package mirrors that deployment boundary:

- `app/local/` contains analyses, batch-ASR orchestration, database access, ingestion, the dataset
  dashboard, browser voice sessions, and web assets.
- `app/compute/` contains batch-ASR model execution, quality scoring, and streaming ASR/LLM/TTS
  execution.
- `app/shared/` contains only typed compute API models and reusable audio, quality, and storage
  primitives needed on both sides.
- `app/training/` remains an offline application and does not participate in either runtime.

Compute and shared modules must never import local modules. Local modules may call compute only
through the schemas in `app.shared` and the authenticated HTTP/WebSocket clients.

The browser WebSocket terminates at the local application. For each browser voice session, the
local application opens one authenticated WebSocket to the compute backend. That connection
multiplexes stateful Nemotron ASR audio, LLM token deltas, and Pocket TTS audio chunks. Keeping one
connection avoids reconnecting for each sentence while keeping turn-taking, history, cancellation,
and user-visible protocol state local.

Silero VAD runs in the local session because speech start is an orchestration decision: it must
cancel playback immediately without waiting for a remote round trip. Nemotron retains its genuine
streaming RNNT state on the compute server for the duration of each utterance. LLM requests contain
the complete ordered conversation on every turn; the compute server does not persist conversations.

## Compute API

All compute endpoints except liveness require `Authorization: Bearer <token>`. The token comes from
`VOICE_LIGHT_COMPUTE_TOKEN` on both applications.

- `GET /health/live`: process liveness; does not imply that models are ready.
- `GET /health/ready`: ready only after the streaming ASR, LLM, and TTS startup stages complete.
- `POST /v1/asr:batch`: batch ASR with request-contained base64 audio.
- `POST /v1/quality:analyze`: two multipart audio uploads plus a sample identifier.
- `WS /v1/voice`: typed multiplexed streaming ASR, LLM, TTS, and cancellation commands.

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
