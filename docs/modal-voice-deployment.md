# Modal Voice Deployment Runbook

## Architecture

`deployment.modal.voice_light` is a thin Modal ASGI wrapper around the current
`app.compute` runtime and application factories. It does not restore the deleted
`app.voice_agent` runtime. The current `/v1/voice` route, `ComputeRuntime`, `VoiceSession`, tool
registry, search integration, predictive generation, playback controller, Nemotron ASR, Qwen
workers, and Kyutai TTS remain authoritative.

The GPU container admits one Modal input and the compute route separately enforces one live voice
session. Modal requests a co-located A10 pair first and falls back only to an L40S pair when A10
capacity is unavailable. Qwen, Nemotron, and the search summarizer are pinned to physical GPU 0;
Kyutai is pinned to physical GPU 1, so concurrent text and first-frame speech generation cannot
contend for the same CUDA device.
A100 and H100 are deliberately excluded from the bounded fallback list because they are
unnecessarily expensive for this stack. The endpoint is
scheduled in Modal's broad `eu` compute region and routed through `eu-west`; this avoids
latency-dominated global placements while retaining the larger European GPU pool. Modal may scale
to zero, has one maximum container, and keeps an idle container for 120 seconds. Modal sets
`VOICE_LIGHT_EAGER_MODEL_LOADING=true`, so the ASGI lifespan awaits model
initialization before Modal marks a cold container ready or admits the first request. Nemotron,
Qwen, Kyutai, and the independent search summarizer load concurrently. The 1,800-second Modal
startup timeout bounds that work. Other provider-neutral deployments retain
background loading: `/health/live` can succeed before `/health/ready`, and `/v1/voice` closes with
retryable code `1013` until every required model is ready.

One persistent Nemotron subprocess owns both streaming RNNT decoding and turn-adapter inference.
The adapter consumes layer 6/12/18/24 features from the RNNT encoder call and retains incremental
causal convolution and GRU state. A second Nemotron backbone and rolling waveform re-encoding are
not used.

Modal uses pinned `Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554` with the direct Transformers backend. A production-
backend canary replaced the final 16-pass 1.7B tool LoRA after fixed prompts reproduced its poor
factual reasoning and irrelevant tool choices. The official 4B checkpoint retained structured
weather, calculation, and multi-zone time calls in the same canary. This is an integration check,
not a general model-quality claim. Conversational sampling uses the model-card recommendations of
temperature 0.7, top-p 0.8, and top-k 20. Modal uses a separate pinned Qwen3-0.6B worker for search
summarization. It loads concurrently with the conversation model and prevents search summarization
from blocking the 4B conversation worker or contending with Kyutai during a tool turn. The typed
streaming, tool-call, cancellation, and stale-event protocols remain unchanged.

Qwen remains the primary tool selector. A model may emit a correct tool call without audible text;
in that case the session supplies a short tool-specific bridge while execution is already in
flight. If Qwen omits a structured search call for an explicit
current-information/search request or a confirmation of an immediately preceding lookup offer, a
narrow typed router supplies the missing `search` call. The call still passes through the normal
schema validator, tool journal, and configured provider; post-tool rounds cannot route again.
Grounded search summaries are already speech-ready and go directly to Kyutai instead of requiring
a redundant second conversational-model round. The short bridge is finalized at its semantic
boundary while the tool continues in the background; the result then starts a new acoustic session
inside the same browser playback generation. Kyutai's two-word lookahead otherwise strands the end
of an open bridge until result text arrives, producing a deterministic mid-preamble stall.
Successful tool messages contain the raw result, matching both the fine-tuning renderer and Qwen's
native tool-response format rather than an internal execution envelope. Tavily uses its `fast`
search depth, and results below the configured 0.5 relevance threshold are excluded before the
summarizer. The summary prompt also rejects evidence about a different named place.

Temperature conversion is deliberately narrower and deterministic. Explicit or contextual
Celsius/Fahrenheit/Kelvin requests are converted into bounded calculator expressions, including
adjacent weather ranges such as a Fahrenheit high and low. Once Qwen emits the tool-call start
marker, the router reuses that call identity and dispatches without waiting for the remaining JSON
syntax. The exact labeled calculator result is spoken without a second Qwen interpretation round.
This both prevents unit hallucinations and removes the measured calculator JSON-tail and
post-result contention from that path. Other arithmetic continues through the ordinary structured
calculator flow.

## Account configuration

Create the named secret without putting credentials in the repository:

```powershell
modal secret create voice-light-compute `
  VOICE_LIGHT_COMPUTE_TOKEN='<random bearer token>'

modal secret create voice-light-search `
  VOICE_LIGHT_TAVILY_API_KEY='<Tavily API key>'
```

`VOICE_LIGHT_COMPUTE_TOKEN` protects the HTTP APIs. The browser voice WebSocket is intentionally
unauthenticated because the page connects directly. Public model downloads work without
`HF_TOKEN`, but authenticated downloads have higher Hub rate limits. Add `HF_TOKEN` to
`voice-light-compute` when needed. Tavily is isolated in `voice-light-search`, so updating the
search credential cannot replace the compute token. Without the Tavily key, the search tool reports
its typed unavailable result. Updating a Modal secret restarts dependent containers.

The deployment creates or reuses `voice-light-agent-model-cache` for Hugging Face and Torch data
and `voice-light-runtime-cache` for runtime logs and dataset-audio cache. The adapter checkpoint is
copied into the immutable image at `/opt/voice-light-artifacts/adapter-best.pt`; the source remains
the untracked `.cache/rtx3090-node-final-backup-20260829/voice-light-human-finetune-v1` backup.
Run `cache_models` before deployment to populate the Volume with the exact pinned Nemotron, Qwen,
Kyutai, and selected voice artifacts. Serving sets `HF_HUB_OFFLINE=1`, so scale-from-zero
containers read these files from Modal storage and do not redownload or query the Hub.

## Deploy

Use Python 3.12 and run Modal by module so the local `app.py` filename cannot shadow the Modal
package. On a Windows console that does not default to UTF-8, set `PYTHONUTF8` for the CLI output:

```powershell
$env:PYTHONUTF8 = '1'
modal run -m deployment.modal.voice_light::cache_models
modal run -m deployment.modal.voice_light::smoke_tool_use
modal run -m deployment.modal.voice_light::smoke_search_provider
modal run -m deployment.modal.qwen_quality_canary::evaluate `
  --model-name Qwen/Qwen3-4B-Instruct-2507 `
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554
modal deploy -m deployment.modal.voice_light
python -m deployment.modal.smoke_websocket
```

Run the tool-use smoke after prompt, schema, tokenizer, or Qwen changes. It loads the exact
production checkpoint on one L40S and requires ordinary speech, calculation, current search,
explicit search, confirmed-search follow-up, and post-tool continuation cases to emit the expected
spoken text and structured Hermes calls. It validates model behavior without invoking Tavily.

Run the search-provider smoke after creating or updating the `voice-light-search` secret. It fails
before making a request unless that secret contains `VOICE_LIGHT_TAVILY_API_KEY`, then performs a
real bounded Tavily query. Its JSON output contains only `configured`, `result_count`, and
`provider_latency_ms`; it never prints the credential or result content.

The quality canary runs fixed factual-correction, comparison, arithmetic, tool-selection, multi-
zone time, and short-story prompts on the production Transformers backend. Review its individual
JSON observations; passing the protocol cases is not evidence of general research superiority.

### GPU memory snapshot canary

`deployment.modal.voice_light_snapshot_canary` is a separate fixed-A10 deployment for testing
Modal's alpha GPU memory snapshots without changing the production `VoiceLightAgent` application,
autoscaler, or endpoint. It reuses the production image, secrets, cache Volumes, environment, model
configuration, and ASGI application. Unlike production, it loads and warms the complete voice stack
inside `@modal.enter(snap=True)` and enables both CPU and GPU snapshot capture. A post-snapshot enter
hook refuses admission when the restored runtime does not report every voice stage ready.

Deploying this module creates or updates only the canary application:

```powershell
modal deploy -m deployment.modal.voice_light_snapshot_canary
python -m deployment.modal.smoke_websocket `
  --url wss://bertil-braun-private--voicelightagent-voice-light-snapsh-7572b5.modal.run/v1/voice `
  --open-timeout-seconds 300
```

The first few cold invocations may create snapshots rather than restore them. In the Modal
Containers view, distinguish snapshot-creation starts from restored starts and compare multiple
restored `session.ready` measurements with the production fixed-A10 baseline. Do not promote the
canary unless all three CUDA subprocesses restore coherently, a complete WebSocket turn produces
valid PCM, restored readiness improves by at least 30%, and first-turn and p95 latency do not
regress. Redeploying code or changing GPU configuration invalidates prior snapshots; changing a
mounted Volume does not, so model-cache changes require an explicit canary redeploy.

The 2026-09-10 canary did not produce a restorable snapshot. The first capture loaded and warmed
all models in 24.432 seconds, then Modal reported `Failed to create memory snapshot`. Its automatic
retry loaded in 49.059 seconds and exceeded the default memory request during capture. A bounded
retry with 64 GiB of container memory again failed snapshot creation, retried model initialization,
and never completed the WebSocket handshake within 300 seconds. The canary application was stopped
after each attempt. Production therefore keeps snapshots disabled, and further snapshot work is
not planned for this project completion pass.

The deployed endpoints are:

- public demo: `https://bertilbraun.github.io/Voice-Light/`
- HTTPS base: `https://bertil-braun-private--voicelightagent-voice-light.eu-west.modal.run`
- voice WebSocket: `wss://bertil-braun-private--voicelightagent-voice-light.eu-west.modal.run/v1/voice`
- Modal dashboard: `https://modal.com/apps/bertil-braun-private/main/deployed/VoiceLightAgent`

GitHub Pages serves the static browser client from
`app/local/web/pages/voice-agent`; its production WebSocket URL is an explicit checked-in constant.
The Pages deployment therefore has no Python runtime, model artifacts, or credentials and loading
it does not start or hold an inference container. Modal hosts only the GPU WebSocket service. GPU
scale-from-zero remains the longer 32--55 second initialization path described above.

Serve the browser locally, then put the WebSocket URL in the endpoint field or query string:

```powershell
$env:VOICE_LIGHT_RELOAD = 'false'
python -m app.local.server
```

```text
http://127.0.0.1:8000/voice-agent?compute=wss%3A%2F%2Fbertil-braun-private--voicelightagent-voice-light.eu-west.modal.run%2Fv1%2Fvoice
```

## Current completion record

The final runtime keeps the current provider-neutral compute application and uses Modal only for
packaging, placement, secrets, persistent cache Volumes, admission, and scale-to-zero. Production
requests a co-located two-GPU allocation (`A10:2`, with `L40S:2` as the only fallback), keeps Qwen,
Nemotron, and the search summarizer on GPU 0, and reserves GPU 1 for Kyutai. Four persistent model
workers load concurrently. The shared Nemotron worker supplies both streaming ASR and causal encoder
features to the incremental turn adapter; it does not load a second speech backbone or repeatedly
re-encode a rolling waveform window. The 120-second idle window is the intentional compromise
between scale-to-zero cost and avoiding a cold start between immediately adjacent demonstrations.

The current final-topology scaled-to-zero readiness observations span approximately 32.0--54.9
seconds, including one 32.736-second two-GPU deployment sample. These are individual observations,
not a percentile or availability guarantee. Modal placement, image startup, framework imports, and
host performance remain variable even though all model weights are read from the persistent Volume.

Two consecutive human production traces reported 11 complete turns with the following browser
telemetry. `end→PCM` ranged from 760 to 1,631 ms (1,288 ms median). Five turns that displayed a
prepared candidate ranged from 760 to 1,014 ms (1,007 ms median); six turns without one ranged from
1,288 to 1,631 ms (1,413 ms median). Across the same turns, endpoint was 487--502 ms, ASR final was
42--147 ms, LLM was 101--164 ms, TTS was 584--665 ms, and browser play acknowledgement was
122--139 ms. These UI spans have different origins and overlap; they must not be added. They include
the user's network and browser path and are human observations rather than a controlled latency
benchmark. The traces exercised current London/New York weather, current time, ordinary follow-up,
and prepared and non-prepared turns. Earlier human runs also exercised backchannel resume,
interruption, stories, and sequential tool use, but no percentile claim is made from those sessions.

At a valid structured tool boundary, the current session flushes the bridge words and finishes that
TTS utterance before executing the call. Tool execution is asynchronous with respect to already
buffered browser playback. Search summaries that are already safe for speech are appended directly;
other successful results return through a subsequent Qwen invocation. Either path starts another
TTS utterance inside the same assistant generation and monotonically rebases its PCM and text
offsets. This prevents Kyutai's lookahead from holding the final bridge words across a tool wait,
but a short pause or acoustic seam at the utterance boundary remains possible and should be judged
in a human microphone smoke. Reusing one uninterrupted Kyutai utterance across the tool wait was
implemented and tested, then reverted because its two-word lookahead could withhold the end of the
bridge until result text arrived; semantic utterance boundaries are the final production behavior.

The following commands reproduce the completion validation without embedding credentials:

```powershell
.\.venv\Scripts\ruff.exe format --exclude .cache .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.\.venv\Scripts\ruff.exe check --fix --exclude .cache .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.\.venv\Scripts\python.exe -m pytest -m "not integration" `
  --ignore=tests/compute/test_merge_qwen_lora.py `
  tests/voice_agent tests/training/turn_taking tests/deployment/modal tests/compute -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
node --test tests\browser\*.test.mjs
```

The final 2026-09-12 run passed 708 Python tests, with three environment-dependent skips and one
live integration test deselected, in 34.71 seconds. The ignored offline LoRA merge test requires the
optional `peft` package, which was unavailable in the Windows validation environment and is not used
by the deployed unadapted 4B runtime. The browser playback/debug suite passed all 35 tests. Ruff
formatting and linting passed. The deployed Qwen tool-routing smoke passed all eight cases, the real
Tavily provider smoke returned two bounded results in 338.89 ms, and the final WebSocket readiness
probe completed in 54.858 seconds. A previous probe of the same final topology completed in 32.042
seconds, illustrating the cold-start variability rather than a code-path change.

After creating the secrets shown above, prepare and validate the pinned artifacts, then deploy and
probe the exact public route:

```powershell
$env:PYTHONUTF8 = '1'
modal run -m deployment.modal.voice_light::cache_models
modal run -m deployment.modal.voice_light::smoke_tool_use
modal run -m deployment.modal.voice_light::smoke_search_provider
modal run -m deployment.modal.qwen_quality_canary::evaluate `
  --model-name Qwen/Qwen3-4B-Instruct-2507 `
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554
modal deploy -m deployment.modal.voice_light
python -m deployment.modal.smoke_websocket `
  --url wss://bertil-braun-private--voicelightagent-voice-light.eu-west.modal.run/v1/voice
```

The automated WebSocket smoke proves admission, protocol readiness, PCM ingestion, and response
framing. Final acceptance still requires a browser microphone run covering audible response,
current search, calculation, current time, normal completion, a short `mm-hmm` backchannel, and a
clear interruption. Record the browser's `end→PCM`, endpoint, ASR, LLM, TTS, play, onset-to-duck,
onset-to-cancel, and onset-to-resume fields from that same run.

## Turn-policy configuration

The deployed starting values are:

| Environment variable | Value | Meaning |
| --- | ---: | --- |
| `VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS` | `1` | Nemotron streaming lookahead used in training and inference |
| `VOICE_LIGHT_FLOOR_TAKE_THRESHOLD` | `0.82` | predicted floor take that commits interruption |
| `VOICE_LIGHT_NON_FLOOR_FEEDBACK_THRESHOLD` | `0.82` | predicted feedback that resumes the same generation |
| `VOICE_LIGHT_OVERLAP_CLASSIFICATION_DEADLINE_MS` | `500` | conservative unresolved-overlap deadline |
| `VOICE_LIGHT_OVERLAP_FINALIZATION_GRACE_MS` | `120` | maximum wait for a prompt final transcript before an empty provisional overlap resumes |
| `VOICE_LIGHT_OVERLAP_PREDICTION_SETTLE_MS` | `80` | bounded wait at speech end for adapter evidence already queued or in flight |
| `VOICE_LIGHT_TRANSCRIPT_FREE_FLOOR_TAKE_DEADLINE_MS` | `900` | hard deadline for sustained overlap without transcript evidence |
| `VOICE_LIGHT_MINIMUM_NORMAL_TURN_COMMIT_SILENCE_MS` | `240` | minimum post-endpoint silence before adapter evidence may commit a normal turn |
| `VOICE_LIGHT_OVERLAP_REARM_SILENCE_MS` | `160` | clean Silero silence required before a resolved backchannel can open another overlap |
| `VOICE_LIGHT_MAXIMUM_PREDICTION_LAG_MS` | `240` | maximum age of causal adapter evidence during active overlap |
| `VOICE_LIGHT_MAXIMUM_TRANSPORT_AHEAD_MS` | `1200` | maximum PCM duration released ahead of browser playback credit; this absorbs intermittent streaming-TTS cadence without delaying worklet-side cancellation |
| `VOICE_LIGHT_SPECULATIVE_YIELD_THRESHOLD` | `0.55` | adapter yield probability that may start a private candidate |
| `VOICE_LIGHT_SPECULATIVE_TURN_COMPLETION_THRESHOLD` | `0.55` | adapter completion probability that may independently start a private candidate |
| `VOICE_LIGHT_SPECULATIVE_MINIMUM_CONFIDENCE` | `0.60` | minimum adapter confidence for either speculative trigger |
| `VOICE_LIGHT_PENDING_SILENCE_SPECULATION_MS` | `80` | low-probability Silero silence that opens the speculative decision window |
| `VOICE_LIGHT_ADAPTER_FIRST_SPECULATION_WINDOW_MS` | `80` | additional window in which causal adapter evidence may start a candidate before the Silero fallback starts at 160 ms total silence |
| `VOICE_LIGHT_VAD_SPECULATION_DEBOUNCE_MS` | `0` | additional silence after Silero's causal endpoint before the VAD fallback starts |
| `VOICE_LIGHT_VAD_ENDPOINT_YIELD_PROBABILITY` | `0.70` | synthetic yield evidence assigned to the causal VAD endpoint |
| `VOICE_LIGHT_VAD_ENDPOINT_CONFIDENCE` | `0.70` | confidence assigned to the causal VAD endpoint evidence |
| `VOICE_LIGHT_QWEN_CUDA_DEVICE` | `0` | physical CUDA device used by the primary Qwen worker |
| `VOICE_LIGHT_SEARCH_CUDA_DEVICE` | `0` | physical CUDA device used by the bounded search summarizer |
| `VOICE_LIGHT_NEMOTRON_CUDA_DEVICE` | `0` | physical CUDA device used by Nemotron ASR and the shared turn adapter |
| `VOICE_LIGHT_TTS_CUDA_DEVICE` | `1` | physical CUDA device reserved for Kyutai TTS |
| `VOICE_LIGHT_QWEN_FIRST_AUDIO_YIELD_ENABLED` | `false` | shared-GPU Qwen yielding is disabled for the isolated two-GPU deployment |
| `VOICE_LIGHT_QWEN_FIRST_AUDIO_YIELD_WORD_COUNT` | `11` | conservative English-word runway before yielding shared-GPU time to Kyutai |
| `VOICE_LIGHT_QWEN_FIRST_AUDIO_YIELD_TIMEOUT_MS` | `400` | safety deadline that resumes Qwen even if first PCM has not arrived |

The threshold is the evaluated Voice-Light starting point, not a universal calibration. Silero
onset always causes the immediate reversible duck/pause. Strong floor-take evidence commits
cancellation; strong non-floor-feedback evidence resumes the same generation without a user turn;
the deadline preserves the conservative fallback. The browser debug panel shows Silero state,
turn completion, floor take, non-floor feedback, policy decision, and decision latency. These are
ephemeral events and never enter durable audible-only conversation history. Its rolling 20-second
timeline advances every 80 ms during user speech, silence, and assistant playback, and merges causal
adapter evidence at Nemotron's approximately 169 ms encoder cadence. Probabilities render only as
timestamped model-sample dots: solid dots were eligible for policy decisions, while hollow dots
were rejected as stale or superseded. Gaps mean no inference, and the panel reports observed median
cadence and latest-sample age instead of inventing interpolated predictions. The current shared
encoder is activated only for a Silero speech turn and its bounded pre-roll; it does not produce
probabilities throughout assistant-only playback or session silence. Continuous encoder-only
interaction inference remains a known limitation.

The latency-first speculative values deliberately spend more hidden Qwen/TTS compute. Candidate
text and PCM remain private until the final transcript passes the causal promotion checks. For an
A/B baseline, set `VOICE_LIGHT_VAD_SPECULATION_ENABLED=false`. To retain speculation with the prior
conservative adapter gate, set both speculative probability thresholds to `0.65` and speculative
minimum confidence to `0.70`.

The browser sends authoritative playback-clock credit every 80 ms. PCM release waits outside the
WebSocket send lock once the configured transport window is full, so duck, pause, resume, and cancel
commands are not queued behind an entire synthesized response. Cancellation wakes blocked senders
and rejects stale-generation PCM; it does not delete valid audio or advance audible-only history.

## Validation and measured deployment results

The chronological engineering record below includes measurements from superseded configurations;
the current completion record above is authoritative for the final topology. Automated route-level
WebSocket coverage uses the
real `/v1/voice` route with mocked model providers and verifies session readiness, PCM ingestion,
final transcript, and audible response framing. The voice-agent and turn-taking suites cover
checkpoint validation, protocol serialization, recurrent reset, incremental causality, optional
adapter degradation, hybrid overlap decisions, cancellation barriers, backchannel resume, stale
generation rejection, structured sequential tool rounds, configured search behavior, playback
acknowledgements, and audible-only history.

| Command | Result |
| --- | --- |
| `ruff format app deployment tests` | 544 files already formatted |
| `ruff check --fix app deployment tests` | all checks passed |
| `pytest tests\voice_agent -q` | 421 passed, 2 skipped |
| `pytest tests\training\turn_taking -q` | 97 passed |
| compute suite excluding the optional LoRA-merge module | 67 passed, 2 skipped |
| Modal deployment tests | 8 passed |
| `node --test tests\browser\*.test.mjs` | 28 passed |

The Windows integration environment does not install the optional `peft` package, so collecting the
entire compute directory fails at `test_merge_qwen_lora.py`. The reported compute run excludes only
that offline model-merge test; deployed merged-model behavior is covered by the Modal tool-use
smoke and is not bypassed in production.

An early deployment that loaded in a background lifespan exposed a Modal-specific idle-suspension
problem: rejected readiness probes released the function input and stretched observed
startup-to-ready to approximately 922 seconds. The eager-startup fix removed that deadlock. The
original current-stack deployment then measured approximately 173.1 seconds to ready, dominated
by 112.270 seconds for conversational vLLM and 30.871 seconds for its separate search model.

The direct merged-Qwen deployment measured 38.647 seconds on its best sampled cold start: 11.551
seconds for Nemotron plus the adapter, 8.973 seconds for Qwen, and 17.023 seconds for Kyutai.
Additional scale-from-zero samples varied up to 80.462 seconds as the three isolated Python/CUDA
workers initialized; a warm health request completed in 0.473 seconds. Loading all three workers
concurrently and an eight-core reservation were both measured and reverted because they increased
sampled startup to 44.924 and 80.462 seconds respectively. These are single observations, not
percentiles.
After cache preparation and the final no-reservation deployment, the verification cold health
request completed in 44.792 seconds and the immediately following warm request in 0.448 seconds.
A live `session.start` WebSocket smoke against the deployed `/v1/voice` route received a validated
`session.ready` event in 1.022 seconds while warm.

CPU and GPU snapshot attempts were reverted. GPU snapshots consistently failed to capture the
current multi-process GPU stack, including after both vLLM engines entered sleep mode and after
vLLM was replaced by direct Transformers. A later import-only CPU snapshot added approximately 30
seconds of capture-and-restore work before model loading on each newly sampled Modal worker type.
The endpoint therefore favors reliable cached loading over snapshot paths that prevented or delayed
admission.

An instrumented deployment after the shared-adapter revision fix measured 46.823 seconds from
WebSocket connection attempt to session readiness on a cold container. Two immediately following
warm sessions were ready in 0.763 and 0.816 seconds. The adapter stayed `active` in all three runs:
the cold run delivered four causal predictions with 23.34 ms median inference latency, and the warm
runs delivered six predictions each with 21.94 and 23.08 ms medians. No adapter degradation event
was observed. The checkpoint revision mismatch and raw-input-frame queue overflow found in earlier
runs are therefore fixed in the deployed endpoint.

A later deployed trace attributed 33.102 seconds of initialization to 0.087 seconds for Silero,
8.878 seconds for Nemotron, 8.843 seconds for Qwen, and 15.294 seconds for Kyutai. Nemotron and Qwen
each spent less than one second in the visible safetensors deserialization loop; most of their stage
time was isolated Python, framework, and CUDA worker bootstrap. The staged startup overlaps those
two similarly sized bootstrap periods without making Kyutai a third concurrent CPU/GPU contender.
The historical pre-removal demo is not a like-for-like ten-second baseline: it used main-process
Transformers Qwen, CosyVoice2 0.5B, and only isolated Nemotron. The current deployment uses isolated
workers for merged Qwen 1.7B, Nemotron 0.6B plus the adapter, and Kyutai 1.6B.

With staged startup deployed, the first model-loading critical path was 28.847 seconds: 0.171
seconds for Silero, 12.071/12.587 seconds for concurrent Nemotron/Qwen, then 15.930 seconds for
Kyutai. A second, slower worker completed the same phases in 42.715 seconds: 0.233 seconds,
18.840/19.088 seconds concurrently, then 23.145 seconds. Sequential loading at those measured
second-run stage rates would have taken about 61.5 seconds. The import-only CPU snapshot confounded
the end-to-end samples by adding about 30 seconds before these phases and was removed.

On 2026-09-10, loading and warming Nemotron, Qwen, and Kyutai concurrently produced fixed-A10
runtime readiness samples of 35.235, 22.357, and 32.489 seconds. The corresponding cold WebSocket
`session.ready` samples were 66.312, 35.124, and 55.836 seconds. This improves materially on the
immediately preceding 51.812-second runtime and 99.257-second end-to-end sample, but does not meet a
reliable 30-second cold-start target. After enabling the bounded A10-to-L40S fallback, one additional
cold production smoke was ready in 35.662 seconds with 22.997 seconds inside `ComputeRuntime`.

The first placement-instrumented global cold start landed in `ap-northeast-2`: independent
load-then-warm chains reached runtime readiness in 39.049 seconds and the client received
`session.ready` in 67.346 seconds. The production endpoint was then constrained to broad European
compute and `eu-west` routing. Its first cold worker landed in `eu-frankfurt-1`, reached runtime
readiness in 26.773 seconds, and delivered `session.ready` in 37.115 seconds. The independent chains
allowed Nemotron's 3.651-second warmup to overlap the remaining Qwen and Kyutai loading. These are
single cold observations, not percentiles; end-to-end readiness still misses the sub-30-second goal.

That production smoke also exercised pending-silence speculation with recorded microphone audio.
Two early candidates were correctly invalidated when Nemotron revised the transcript. The promoted
candidate started only 6.421 ms before the authoritative Silero endpoint, reached its first Qwen word
in 125.373 ms and first Kyutai PCM in 701.631 ms, then released PCM 270.147 ms after commitment. This
proves the early trigger is operational, but transcript churn prevented a material lead on this
utterance; the theoretical 160--220 ms gain is not claimed as an achieved result.

The 2026-09-10 European user session immediately before deterministic conversion routing landed on
AWS `eu-north-1`. All three model loads began within two milliseconds, confirming that startup was
parallel: Qwen loaded/warmed in 23.422 seconds, Nemotron plus the adapter in 23.876 seconds, and
Kyutai loaded/warmed in 30.204 seconds. Runtime readiness took 30.389 seconds. Modal scheduling,
image/container startup, and imports added about 14 seconds before the first placement log, for an
observed click-to-session-ready cold path of about 45.4 seconds. The first cold readiness smoke
after deploying deterministic conversion routing measured 47.795 seconds; the final numeric-evidence
prompt deployment measured 56.765 seconds. Kyutai deserialization, not sequential model loading,
is now the application critical path; variable Modal scheduling plus the failed snapshot canary and
scale-to-zero requirement leave the reliable sub-30-second cold-start target unmet.

That user session produced nine played responses with approximately 1.04-second median
Silero-endpoint-to-PCM and 1.16-second median endpoint-to-playback. Individual PCM-send-to-browser
playback acknowledgements were healthy at 94--174 ms. Ordinary Kyutai first PCM took 0.52--0.66
seconds, while two calculator turns rose to 1.36 and 1.63 seconds under shared-GPU worker
contention. All seven calculator executions themselves succeeded in under one millisecond, but the
old path spent 0.74--1.37 seconds completing model-authored tool JSON and 2.42--4.42 seconds reaching
post-tool PCM. This is the failure addressed by deterministic temperature routing; it is not a
general Kyutai first-frame speedup.

Predictive generation in the same session created 24 candidates and promoted six. A Qwen word was
ready at commitment for 60% of promoted candidates, but TTS PCM was ready for none; median candidate
start was only 11.6 ms before the final Silero endpoint and median candidate-to-PCM was 855 ms.
Consequently, the current speculative policy usually hides part of Qwen but not Kyutai. The UI's
20--24 ms endpoint readings on two turns do not bypass the 240 ms media-sample guard: queued 8 ms
capture frames can be processed faster than real time, while the displayed endpoint uses a server
processing timestamp. A browser-origin true acoustic-end timestamp remains a known telemetry
limitation.

A subsequent five-turn European browser session measured end-of-speech-to-first-PCM at 681--1,658
ms (1,219 ms median and 1,121 ms mean) and end-of-speech-to-browser-playback acknowledgement at
852--1,808 ms (1,377 ms median and 1,284 ms mean). The model path itself remained stable: Qwen first
text took 93--143 ms and ordinary Kyutai first PCM took 547--600 ms. Speculation was promoted on two
of the five turns; those two reached playback in 852 and 905 ms. The other turns spent 800--865 ms
from commitment to first PCM because transcript revisions invalidated private work or the adapter
returned to hold. Pending-silence candidates now compare normalized lexical words during revisions,
so capitalization, punctuation, and spacing corrections preserve work while changed or appended
intent still invalidates it. The deployment smoke for that revision reached `session.ready` from a
cold connection in 45.564 seconds.

One L40S-only cold probe remained queued for more than 120 seconds without Modal creating a
container. That delay occurred entirely before application or model initialization. The ordered GPU
fallbacks address this capacity-dependent scheduling component while retaining L40S as the preferred
cost/performance choice.

The first fallback smoke selected A100-40GB and reached session readiness in 105.341 seconds. Modal
scheduling and container/global startup consumed approximately 24.3 seconds; model initialization
consumed 81.086 seconds: 0.341 seconds for Silero, 35.431/36.352 seconds for concurrent
Nemotron/Qwen, then 44.137 seconds for Kyutai. Because this was much slower than sampled L40S
workers, H100 precedes A100 in the fallback order. The immediately following warm A100 session was
ready in 1.131 seconds.

An isolated full-stack A10 canary subsequently proved that the last-resort GPU fits without changing
the runtime: all five model stages reached ready in 31.122 seconds and `nvidia-smi` reported 9,545
MiB used out of 23,028 MiB. Its stages were 0.111 seconds for Silero, 11.569/12.130 seconds for
concurrent Nemotron/Qwen, and 18.880 seconds for Kyutai. A10 is therefore the final availability
fallback ahead of the substantially slower measured A100, not an assumed fit based on checkpoint
size.

After deploying the A10 fallback and interaction fixes, the final cold WebSocket smoke received a
validated `session.ready` event in 44.895 seconds. This includes Modal scheduling as well as model
initialization and is a single observation, not a latency percentile.

After placing H100 ahead of A100, the final deployed cold container completed model initialization
in 27.940 seconds: 0.091 seconds for Silero, 10.773/11.608 seconds for concurrent Nemotron/Qwen,
then 16.056 seconds for Kyutai. Modal reported 35.7 seconds for the first queued WebSocket request
including scheduling, container, and application startup. A request that joined the already-warming
container completed in 19.321 seconds locally; it is not reported as warm latency. The final worker's
GPU-name probe produced no output, so its exact selected fallback type is unknown. The last
unambiguously warm probe remains the validated 1.131-second A100 `session.ready` result.

The same fixed 7-second WAV smoke measured commit-to-first-PCM at 247.09 ms cold and 183.22/224.72
ms warm. Separate live traces around Kyutai's first word measured 436.6 and 443.0 ms from first word
to worker PCM after skipping redundant depformer computation during known delay steps, down from
963.5--1,035.5 ms in the preceding deployment. The WAV's true-end-to-commit remained 2.56--2.98
seconds because Nemotron/VAD processed the synthetic clip behind real time; this is not presented
as TTS latency. The automated client does not render audio, so audible quality remains a human check.

The final deployment command completed in 45.74 seconds with cached dependency layers. Its browser
smoke reached `Ready to talk`, activated the microphone, streamed 16 kHz PCM, produced Nemotron
partial/final transcripts, exercised speculative Qwen generations and invalidation/promotion, and
received four playback command acknowledgements with a 162.76 ms p95 in that single session. The
stale-candidate escape rate was zero. Continuous ambient speech caused repeated new turns before
Kyutai emitted PCM, so the session recorded zero generated audio samples and no valid
speech-end-to-first-audio, duck, cancellation, or backchannel-resume latency. Speaker audibility
and the configured-search path were therefore not claimed as live passes.

A later human session exposed three correctness and measurement issues. One backchannel produced
12 cooperative overlap resolutions and 36 playback commands because the same Silero-positive tail
could immediately rearm; resume p95 was 832 ms and the browser held as many as 557,568 source
samples (about 23.2 seconds). The deployed trace also showed that the old endpoint label measured
from the first pause inside an utterance, so displayed values of 5,516 and 2,491 ms were not final
speech-end latency. The repaired implementation requires 160 ms of clean silence before overlap
rearming, bounds transport-ahead PCM to 500 ms, and reports first pause, final endpoint, ASR
finalization, candidate resolution, server PCM release, and browser playback separately. These are
pre-deployment findings and test-validated corrections, not post-fix production latency claims.

The real Modal search-provider smoke failed safely before making an HTTP request because the
`voice-light-compute` secret did not contain `VOICE_LIGHT_TAVILY_API_KEY`. The tool loop now stops
after one truthful provider failure and rejects an identical repeated successful call, but live
weather remains unavailable until the account owner supplies that secret and reruns the smoke.

The deployment containing these corrections completed in 56.737 seconds. Its first WebSocket
probe reached `session.ready` in 64.688 seconds from a scaled-to-zero state; the immediately
following warm probe was ready in 1.011 seconds. The deployed merged-Qwen smoke passed all six
ordinary, calculation, search, explicit-search, confirmed-search, and post-tool-continuation cases.
The search-provider smoke independently reproduced the missing-secret failure above. No post-fix
human microphone run was available during this deployment, so the transport-window and rearm
latencies remain regression-tested rather than claimed as measured production improvements.

The next deployment made readiness include bounded production-path probes instead of stopping at
worker construction. Nemotron (including the shared turn adapter) and Qwen warm concurrently after
their concurrent load; Kyutai then performs one complete synthesis and drain before the WebSocket
can emit `session.ready`. On the measured cold worker, Nemotron/Qwen loaded in 21.862/22.256 seconds,
their concurrent probes occupied approximately 7.51 seconds, Kyutai loaded in 27.449 seconds, and
its final probe took approximately 0.68 seconds. Total application model readiness was 58.048 seconds.
The scaled-to-zero WebSocket probe reached `session.ready` in 72.858 seconds including scheduling;
the immediate warm probe took 1.180 seconds. This intentionally moves the previously hidden first
ASR/LLM/TTS execution cost into readiness. It prevents a misleading ready state, but does not make
the cold path fast.

Three-way model loading remains disabled because the measured attempt regressed the same startup
work from 38.647 to 44.924 seconds; eight reserved CPU cores regressed it to 80.462 seconds. Weight
deserialization itself remained near one second per checkpoint in the latest logs, so the dominant
cost is isolated Python/framework/CUDA bootstrap rather than redownloading models. The safe current
parallelism is Nemotron plus Qwen loading, followed by Kyutai, with the Nemotron and Qwen probes also
running concurrently.

The same deployment no longer exposes the search tool when
`VOICE_LIGHT_TAVILY_API_KEY` is absent. A current-weather request now produces one immediate truthful
spoken failure without a generated preamble, failed tool round, or search summarizer invocation.
Configured search still uses the structured sequential tool path. Tool timing logs now separate
Qwen tool-syntax buffering, handler execution, and post-tool first PCM; the browser worklet also
publishes an exact playback clock when a tool preamble drains so later same-generation PCM cannot
wait on a stale sub-80-ms credit edge. The deployed structured-tool smoke passed all six cases. The
real-provider smoke still failed before HTTP with the exact missing-secret error, so live weather was
not claimed. The UTF-8 deployment retry completed in 14.337 seconds with cached image layers.

After separating Tavily into the dedicated `voice-light-search` secret, the real provider smoke
succeeded with `configured=true`, two bounded results, and 1,978.38 ms provider latency. The secret
value was neither logged nor committed. The deployment containing the search secret and speculative
audio deadlock fix completed in 15.763 seconds with cached layers.

A subsequent human run recorded three empty-transcript overlaps resolved as non-floor-taking in
216--248 ms, but onset-to-resume reached 811 ms because the session synchronously awaited final ASR
before sending `RESUME`. The policy now gives final ASR a typed 120 ms grace: prompt meaningful
lexical material cancels without a false resume, while a slow or empty finalization resumes the same
generation after the bound and completes classification afterward. Sustained speech and strong
floor-take evidence retain their existing cancellation paths.

That run also reported zero skipped and zero replayed source samples, while the browser queue
repeatedly approached the old 500 ms transport ceiling (11,264 of 12,000 samples at 24 kHz). This
supports playback underrun rather than PCM ordering corruption as the likely periodic crackle cause.
The deployed transport headroom is now 1,200 ms (57.6 KB PCM); browser-side cancel still immediately
discards queued audio, and playback still starts with the first frame rather than waiting for a full
prebuffer. Human listening remains required to confirm that the crackle is gone. This deployment
completed in 52.165 seconds; its cold and immediately warm readiness probes measured 63.038 and
1.194 seconds respectively.

The next deployment adds direct evidence for that hypothesis. Each 80 ms browser playback clock
now carries a cumulative underrun count. An underrun is counted only when active, nonterminal audio
drains before `assistant.audio.end` and later PCM for the same generation resumes; final drains,
policy pauses, cancellation, replacement, and idle time are excluded. The live browser status and
server session report expose the count without adding it to durable conversation history.

The deployment completed in 52.965 seconds. A corrected WebSocket benchmark validated every binary
audio generation, sequence, and source position, returned playback credit for each drained chunk,
and reached a complete `assistant.audio.end` instead of stopping at the transport window. Its cold
`session.ready` was 56.317 seconds and the immediately warm result was 1.235 seconds. Cold and warm
commit-to-first-PCM were 834.78 and 951.84 ms. The fixed WAV endpoint itself lagged true audio end by
4.751 and 4.586 seconds, so its 5.585 and 5.538 second true-end-to-first-PCM values characterize the
synthetic benchmark, not natural microphone latency. The shared adapter stayed active with 32 and
31 predictions; median reported inference latency was 77.72 and 91.94 ms. A real Tavily smoke also
returned two bounded results in 2,286.85 ms. The benchmark simulates immediate playback drain, so
only a browser microphone run can establish the deployed audible underrun count and confirm whether
the larger transport window removed the reported crackle.

A subsequent human session exposed ten confirmed playback underruns, concentrated around spoken
tool preambles, with zero skipped, replayed, or discarded samples. Same-generation PCM now applies
a five-millisecond tail ramp when an active queue truly drains and a five-millisecond attack only
when audio resumes after a confirmed underrun; contiguous chunks and immediate cancellation are
unchanged. The same trace proved a separate terminal-playback race: two overlaps started after the
server generation had completed and then vanished without either resolution or promotion. An
overlap that outlives resumable assistant playback now finalizes and commits through the normal
user-turn path, including when ASR had no partial before producing a multiword final.

Four searches in that session completed successfully, while a fifth failed after the provider's
three-second HTTP timeout without receiving a response. The canonical typed Tavily timeout is now
five seconds and failure logs include only the safe exception type. The deployment containing all
three fixes completed in 42.393 seconds. Its scaled-to-zero WebSocket readiness smoke completed in
55.942 seconds, and a real configured Tavily smoke returned two results in 290.89 ms. Audible
tool-boundary quality and the terminal-overlap repair still require a refreshed human browser run.

The following human session showed that five of six overlaps were prematurely accepted as
non-floor feedback after only 256--432 ms while Silero still reported active speech. Their empty
early ASR finals reset the turn and discarded the remaining words. Non-floor model evidence is now
provisional until VAD speech end; strong floor-take and meaningful partial lexical evidence remain
immediate. At speech end the session waits at most 80 ms for adapter work already queued or in
flight before using the acoustic fallback, preserving causal attribution without depending on task
scheduling. Required search requests now dispatch when Qwen emits the typed tool-call-start event,
instead of waiting another measured 1.22--1.51 seconds for redundant search JSON to finish. Other
tools and ambiguous requests still require complete structured arguments.

The next 12-turn human trace measured the user-perceived speech-end-to-first-rendered-audio path as
`endpoint + total`, equivalently `end-to-PCM + play`. Its median was 1,801 ms, p95 was 2,007 ms,
and range was 961--2,007 ms. Median component times were 469 ms for endpoint commitment, 162 ms for
the first complete Qwen word, 635 ms from that word to Kyutai PCM, and 279 ms from server PCM send
to browser rendering. Four speculative hits had a 1,152 ms median versus 1,877 ms for misses, but
none had TTS PCM buffered at commit. The best observed component combination was still 898 ms, so a
consistent sub-800-ms result requires improvements in at least two stages rather than relabeling or
one queue adjustment.

Overlap playback now uses asymmetric gain envelopes. Unresolved overlap fades toward -15 dB over
450 ms and returns to full gain over 450 ms when classified as a backchannel. The fallback pause is
500 ms so it cannot truncate the reversible fade. A committed interruption cancels server
generation and rejects new PCM immediately, while at most 100 ms of already buffered browser audio
fades to silence before its exact played position is acknowledged and the remainder is discarded.
Paused or idle cancellation remains immediate.

A detected onset before the first assistant sample is audible holds the pending generation rather
than cancelling it immediately. If ASR finalizes that onset without text, the same generation is
released; lexical speech still commits an interruption. This avoids leaving the session idle after
a pre-playback false start.

### 2026-09-11 Qwen and tool-path validation

The final 1.7B tool LoRA and the official 4B Instruct checkpoint were run through the same fixed
production-backend canary. The 1.7B output reproduced the reported incoherent funny fact and an
irrelevant correction path. The 4B checkpoint returned coherent comparison, arithmetic, weather-
search, two-zone time, and story structures. One factual-correction answer still underestimated
lifetime food consumption, so the result supports replacing the clearly regressed checkpoint but
does not establish broad factual reliability.

After promotion, the deployed L40S tool smoke passed all six protocol cases: ordinary speech,
calculation, current search, explicit search, confirmed-search follow-up, and post-tool
continuation. The configured real Tavily smoke returned two results in 233.996 ms. A scaled-to-zero
WebSocket reached `session.ready` in 32.735 seconds. Runtime logs place the container in Frankfurt
and show all required models ready in 21.008 seconds: Nemotron plus adapter load/warm took
12.089/2.770 seconds, Qwen 4B took 15.080/2.218 seconds, and Kyutai took 19.975/0.918 seconds, with
all three loading concurrently. The difference between 21.008-second runtime readiness and the
32.735-second client observation is Modal scheduling/container/import/routing overhead.

The deployed endpoint remains
`wss://bertil-braun-private--voicelightagent-voice-light.eu-west.modal.run/v1/voice`.
A browser microphone run is still required to judge the continuous tool-boundary audio and the
new model's conversational behavior under real interruptions.

The following human run exposed two tool-path regressions. The open Kyutai bridge session retained
its two-word lookahead across the tool await, so spoken preambles stalled until result text arrived.
The search pipeline used Tavily's lowest-relevance `ultra-fast` mode, discarded its relevance score,
and admitted a Split, Croatia result into London and New York prompts. On that worker Tavily itself
took 359--457 ms, while shared 4B search summarization took 1,246--2,016 ms and post-tool first PCM
took another 926--1,969 ms. Tool execution began within 36 ms once a complete call was available;
the provider was not being scheduled after playback.

The deployed repair finalizes the bridge at the semantic boundary, uses Tavily `fast` search with a
0.5 minimum relevance score and named-place mismatch instruction, and isolates search summarization
on the pinned Qwen3-0.6B worker. A configured post-deploy provider smoke returned two accepted
results in 407.77 ms. A scaled-to-zero WebSocket reached `session.ready` in 32.294 seconds. Runtime
readiness was 21.439 seconds: the search summarizer loaded in 12.234 seconds, Nemotron loaded/warmed
in 12.976/2.773 seconds, Qwen 4B in 15.428/2.105 seconds, and Kyutai in 20.430/0.893 seconds. All
four workers started concurrently, and Kyutai remains the cold-start critical path. Audible bridge
continuity and grounded result quality still require a refreshed human microphone run.

The next human transcript showed that current-weather searches worked, but the base 4B model then
answered an exact speed conversion and a disputed windsurfing claim from memory. The runtime prompt
and typed tool descriptions now require `calculate` for exact arithmetic, comparisons, totals, and
unit conversions, and require `search` for uncertain factual claims challenged by the user. The
production-backend canary passed both a kilometers-per-hour-to-knots calculation call and the
challenged-fact search call, in addition to its existing six cases. This strengthens tool selection;
it does not make ungrounded base-model statements authoritative.

All generated speech now passes through one TTS-boundary normalizer after structured tool parsing.
Markdown emphasis, code, strike markers, and double quotation marks are removed from synthesis
words while apostrophes, numeric punctuation, the browser transcript, durable audible history, and
source text offsets remain unchanged. This prevents Kyutai from vocalizing formatting artifacts
without allowing normalization to alter tool JSON.

The public-demo follow-up keeps the 4B Instruct model and does not restore the regressed 1.7B
fine-tune. Predictive transcript revisions are compared as semantic word sequences: capitalization,
punctuation, whitespace, and apostrophe-only changes preserve already prepared Qwen/TTS work, while
lexical additions or substitutions invalidate it. The original ASR text, not the comparison form,
is always sent to Qwen and retained in conversation history. During assistant playback, incomplete
generic partials remain reversible until speech ends or the 900 ms hard deadline; explicit repair,
question, and stop language plus strong adapter floor-take evidence remain immediate.

The adapter now gets an 80 ms scheduling opportunity after the initial 80 ms pending-silence
signal. If usable causal evidence has not arrived, the existing Silero fallback starts speculation
at 160 ms. Every committed turn reports its causal source in the typed latency event, and overlap
logs include decision, reason, source, and latency. This makes the adapter's actual contribution
measurable instead of inferring it from probability plots. Deterministic search routing also bounds
queries to the typed 240-character provider limit, preferring the final complete question; the
reported 304-character windsurfing utterance therefore routes as `What's the actual max speed of a
wind surfer ever?` instead of failing Pydantic validation. These changes were validated with 114
session tests, 10 observability and Modal configuration tests, 33 overlap tests, 19 predictive
tests, and 19 deterministic-routing tests before deployment. The Modal deployment completed in
44.886 seconds, and the immediately following scaled-to-zero WebSocket smoke reached
`session.ready` in 32.968 seconds. The smoke validates deployment and readiness, not microphone
quality or post-change adapter contribution; those still require a fresh human interaction trace.

An isolated L40S benchmark confirmed that Kyutai's first PCM is structurally delayed until model
step 19: the checkpoint uses a 16-frame text/audio shift, a two-frame acoustic delay, and then the
first complete decodable frame. Five warm 32-codebook runs reached first PCM in 313--342 ms
(314 ms median after the first run), compared with 294--354 ms at 16 codebooks and 292--339 ms at
8 codebooks. Reducing codebooks therefore saved only about 10--20 ms of first-frame latency while
improving full-utterance real-time factor from approximately 0.28 to 0.20 and 0.16 respectively.
Production remains at 32 codebooks because the small first-frame gain does not justify unmeasured
speech-quality loss. The 570--650 ms Kyutai latency in the following full voice trace is instead
consistent with GPU contention from concurrent Qwen inference; controlled Qwen/TTS scheduling is
the next latency experiment.

The controlled scheduling experiment is now deployed on the Transformers production backend.
After 11 complete synthesis words have been submitted without first PCM, the session sends a typed
pause for the current Qwen worker invocation. The logits processor blocks between tokens so Kyutai
can use the shared GPU; first PCM resumes Qwen immediately, while a 400 ms timeout and generation
cleanup prevent deadlock. Short responses and tool preambles that produce PCM before the threshold
are unchanged. The static 11-word runway is conservative for ordinary English text, but source
words are not identical to Kyutai tokenizer entries, so the timeout remains required.

Deployment of commit `67e1026e` completed successfully. A scaled-to-zero WebSocket reached
`session.ready` in 35.590 seconds. In a recorded multi-turn microphone trace, the yield fired at 11
words and first PCM resumed Qwen after 104.2 ms rather than reaching the timeout. A separate warm
three-trial clipped-microphone smoke measured commit-to-first-PCM at 607.68, 614.09, and 663.75 ms
(614.09 ms median, 663.75 ms p90). Those trials used a short tool preamble and reached PCM before
the yield threshold, so they validate the unchanged fast path rather than proving an A/B latency
gain. A fresh human run with long non-tool responses is still needed to judge audible quality and
the shared-GPU benefit across varied generations.

The next deployment replaced shared-GPU scheduling with two co-located GPUs. It requests
`A10:2` with `L40S:2` as the only fallback, retains scale-to-zero and the 120-second idle window,
and excludes A100/H100. Deployment commit `d78115d2` reached cold `session.ready` in 32.736 seconds;
model initialization itself took 21.238 seconds. Three warm recorded-microphone trials measured
Kyutai worker first-word-to-PCM at 353.4, 346.6, and 346.2 ms. Its LM portion was 183.4--184.4 ms
and Mimi decoding was 160.0--168.6 ms. This restores the isolated Kyutai profile and removes the
previous human trace's 621.6--711.8 ms shared-GPU TTS times. Qwen first delta remained 231.5--237.6
ms. Commit-to-first-PCM was 632.14--655.18 ms (649.59 ms median), so GPU isolation fixes measured
TTS contention but does not remove endpoint, final-transcript, candidate-resolution, or network
latency. A human interaction run is still required to measure end-to-PCM and perceived pacing.

An A/B placement experiment moved Nemotron beside Kyutai on physical GPU 1, leaving Qwen alone on
GPU 0. Cold model readiness remained effectively unchanged at 21.151 seconds and Qwen first delta
improved from roughly 232--238 ms to 204--207 ms. During five warm recorded turns, however, Kyutai
worker first-word-to-PCM regressed to 458.9--485.6 ms from the isolated 346.2--353.4 ms baseline.
The roughly one-third TTS regression outweighed the small Qwen gain, so production retains
Nemotron and Qwen on GPU 0 and reserves GPU 1 for Kyutai.

## Final human acceptance run

The final microphone run on 2026-09-12 completed eight of eight turns, including live weather and
time tool use and ordinary conversational follow-ups. The tester reported that background speech
and interaction actions behaved correctly and considered the result ready to wrap. The trace's
`end→PCM` values were 593, 597, 616, 660, 694, 705, 1,318, and 1,451 ms: median 677 ms, mean
829 ms, and six of eight turns below 800 ms. Median endpoint decision, LLM first-word, TTS
first-PCM, and browser playback measurements were 508, 268, 453, and 109 ms respectively.

The two latency outliers were the turns that did not promote prepared speculative work; both still
completed normally. No turn was dropped and no playback stall was reported. This is one human
session rather than a controlled population estimate, and the supplied transcript does not contain
separately timestamped duck, cancellation, or backchannel-resume actions.

## Known limitations

- The current final-topology cold readiness observations range from approximately 32.0 to 54.9
  seconds. Earlier architectures ranged as high as 66.312 seconds, and an obsolete A100 fallback
  required 105.341 seconds end to end. Modal scheduling and host performance remain variable.
  Restoring the old approximately
  ten-second behavior requires consolidating repeated Python/CUDA worker bootstrap, keeping a warm
  container (which conflicts with scale-to-zero), or replacing the larger current model stack;
  cached weights alone cannot remove library initialization. GPU memory snapshots remain an alpha
  experiment, not a production setting: the fixed-A10 canary failed capture even with a 64 GiB
  memory request. Further GPU-snapshot work is not planned.
- A live human must provide microphone speech and judge audible output. Automated and agent-run
  checks cannot honestly certify microphone capture, speaker audibility, natural backchannel, or
  interruption perception.
- The final eight-turn human acceptance trace measured `end→PCM` at 593--1,451 ms (677 ms median,
  829 ms mean), with six turns below 800 ms. It remains one small acceptance trace rather than a
  latency distribution, and controlled live duck, cancellation, and resume measurements remain to
  be recorded on the final deployment.
- The Transformers conversation worker now logs the exact input prompt-token count, but conversation
  history is not yet compacted or token-budgeted. A long session can therefore increase prefill
  latency, dilute attention to the latest topic, and eventually exceed the model context window.
- Tool bridges and results are separate TTS utterances within one assistant generation. The semantic
  boundary prevents a lookahead stall, but an audible pause or voice seam can remain at that point.
- The adapter currently catches up from a bounded 300 ms pre-roll after Silero onset. Continuous
  assistant-playback context requires replacing the high-level RNNT generation loop with one
  scheduler that owns persistent encoder, decoder, and adapter state; a side encoder loop cannot
  safely share the private Hugging Face streaming caches and was not added.
- Real web search requires `VOICE_LIGHT_TAVILY_API_KEY` in the separately managed
  `voice-light-search` Modal secret.
- Cache preparation works without `HF_TOKEN` but may be rate-limited; serving itself is offline.
