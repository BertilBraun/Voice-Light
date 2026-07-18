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
per-session VAD state, genuine streaming Nemotron state, turn decisions, conversation history,
Qwen/TTS work, and cancellation for the life of that WebSocket. The runtime admits one live voice
WebSocket at a time. Silero is loaded during readiness; persistent Nemotron, Qwen, and selected TTS
child processes own their CUDA models. No voice state survives disconnection.

## Compute API

Compute HTTP endpoints except liveness require `Authorization: Bearer <token>`. The research voice
WebSocket is deliberately public for direct use by the browser.

- `GET /health/live`: process liveness; does not imply that models are ready.
- `GET /health/ready`: ready after the enabled startup stages complete.
- `POST /v1/asr:batch`: batch ASR with request-contained base64 audio.
- `POST /v1/quality:analyze`: two multipart audio uploads plus a sample identifier.
- `WS /v1/voice`: focused, ephemeral cascaded voice session with binary PCM in both directions.

The compute server never accepts filesystem paths or shell commands from HTTP clients. Quality
uploads are decoded in request-scoped temporary storage and deleted after the response.

## Voice web search

The voice agent's `search(query)` tool uses the Tavily Search API. Set
`VOICE_LIGHT_TAVILY_API_KEY` in `.env.compute` to a Tavily API key. The
voice stack still starts when this setting is absent, but an attempted search returns a clear tool
failure until a key is configured.

Each invocation requests at most three web results over a three-second HTTP timeout. The compute
process normalizes and bounds each title, URL, and snippet, then submits only that bounded context
to a separate deterministic pass through the already-loaded Qwen model with tools disabled and a
64-token output cap. Raw provider payloads and the summarization prompt never enter the session
conversation; only Qwen's bounded plain-text summary is committed through the normal tool-result
message. The speech-oriented summary targets one or two sentences and 40 words, omits unsolicited
comparisons and tangents, and excludes source names, URLs, and citations.

This design adds one external network round trip and one serialized Qwen generation before the
main post-tool response. It avoids an additional model copy and GPU concurrency hazards, but search
turns will therefore be materially slower than calculator or local-time turns. The worker lease
ensures the summarization pass cannot overlap another Qwen inference. The surrounding tool timeout
defaults to 30 seconds so the bounded summary can finish on the supported GPU; deployments can
override it with `VOICE_LIGHT_TOOL_TIMEOUT_SECONDS`.

Provider and summarizer durations are logged separately without the query or result contents. This
makes it possible to distinguish Tavily network latency from Qwen prefill/generation time before
changing result bounds or summary quality. A browser-only `search.debug` event additionally logs
the bounded normalized results, exact isolated Qwen request, summary, and stage timings to the
developer console. This event is never added to the main model context or audible conversation.

## Browser-local time

The argument-free `get_time()` tool returns the current time in the browser user's IANA time zone,
not the compute server's operating-system time zone. The browser includes the zone reported by
`Intl.DateTimeFormat` in `session.start`; the compute boundary validates it with the IANA zone
database. Tool results include the local ISO 8601 timestamp, UTC offset, and zone name so the model
does not have to infer which clock it received. Non-browser clients that omit this session field
retain an explicit `Etc/UTC` compatibility default.

The model speaks one short bridge before starting search. That bridge is finalized as its own TTS
utterance so it cannot stall mid-word while the search and summary run. The post-result answer uses
a successor TTS utterance whose PCM and word boundaries are rebased into the same browser playback
generation, preserving one audio start/end pair while allowing an intentional silent wait.

## ASR-only mode

Set `VOICE_LIGHT_VOICE_STACK_ENABLED=false` to run the compute server only for batch ASR and
quality requests. This skips the Silero speech detector and the streaming ASR, language-model, and
TTS workers. Readiness succeeds immediately with no model stages, batch ASR models remain
request-loaded and cached, and `/v1/voice` rejects connections while the mode is active. The setting
defaults to `true`. Existing deployments can switch modes by editing `.env.compute` and restarting
the compute service.

## TTS selection

`VOICE_LIGHT_TTS_BACKEND` selects `kyutai` or `voxtream` at process startup. Both implement the same
typed word-input/audio-event interface and use the same restartable subprocess lifecycle,
cancellation deadline, progress watchdog, and exclusive worker lease.

- `kyutai` uses `kyutai/tts-1.6b-en_fr` through Moshi's direct PyTorch runtime. It has the stronger
  current voice quality and model-predicted word alignment, but approximately 469 ms isolated
  first-word-to-PCM latency.
- `voxtream` uses VoXtream2 at pinned revision
  `8ec2d62159dae4716ae7058827244a962d40603c`. It consumes words while emitting audio and uses
  phoneme progress for conservative word-completion boundaries. The fixed prompt is cached.

VoXtream requires older Torch, Transformers, Hugging Face Hub, and related dependencies. Bootstrap
therefore installs it in `.cache/compute/voxtream/.venv`; those packages never enter the main
Nemotron/Qwen environment.

To select VoXtream:

```bash
sed -i 's/^VOICE_LIGHT_TTS_BACKEND=.*/VOICE_LIGHT_TTS_BACKEND=voxtream/' .env.compute
bash deployment/compute/bootstrap.sh
bash deployment/compute/start.sh
```

Switch the value back to `kyutai` and restart to restore the original synthesizer.

`uv run python -m deployment.compute.benchmark_tts --runs 5` measures cold/warm first-chunk
latency, first-word-to-audio latency, delayed-stream LM and Mimi decode time, model-step count,
total generation time, real-time factor, output duration, and emitted boundary count on the rented
machine.
