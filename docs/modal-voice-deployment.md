# Modal Voice Deployment Runbook

## Architecture

`deployment.modal.voice_light` is a thin Modal ASGI wrapper around the current
`app.compute` runtime and application factories. It does not restore the deleted
`app.voice_agent` runtime. The current `/v1/voice` route, `ComputeRuntime`, `VoiceSession`, tool
registry, search integration, predictive generation, playback controller, Nemotron ASR, Qwen
workers, and Kyutai TTS remain authoritative.

The GPU container admits one Modal input and the compute route separately enforces one live voice
session. Modal requests L40S first, then allows H100, A10, or A100 when preferred capacity is
unavailable. All four have sufficient memory and CUDA compatibility for the measured stack;
fallbacks reduce scheduling stalls but have different cost and performance. Modal may scale to
zero, has one maximum container, and keeps an idle container for 1,200 seconds. Modal sets
`VOICE_LIGHT_EAGER_MODEL_LOADING=true`, so the ASGI lifespan awaits model
initialization before Modal marks a cold container ready or admits the first request. Nemotron and
Qwen start concurrently, then the shared search generator and Kyutai start in dependency order. The
1,800-second Modal startup timeout bounds that work. Other provider-neutral deployments retain
background loading: `/health/live` can succeed before `/health/ready`, and `/v1/voice` closes with
retryable code `1013` until every required model is ready.

One persistent Nemotron subprocess owns both streaming RNNT decoding and turn-adapter inference.
The adapter consumes layer 6/12/18/24 features from the RNNT encoder call and retains incremental
causal convolution and GRU state. A second Nemotron backbone and rolling waveform re-encoding are
not used.

Modal uses the pinned merged 1.7B Qwen checkpoint with the direct Transformers backend. The same
worker handles conversation generation and search summarization because tool rounds are
sequential. This avoids vLLM's approximately 90-second engine profiling path and a second Qwen
backbone while preserving the existing typed streaming, tool-call, cancellation, and stale-event
protocols.

Qwen remains the primary tool selector. If it omits a structured search call for an explicit
current-information/search request or a confirmation of an immediately preceding lookup offer, a
narrow typed router supplies the missing `search` call. The call still passes through the normal
schema validator, tool journal, configured provider, and sequential Qwen continuation; post-tool
rounds cannot route again.

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
Run `cache_models` before deployment to populate the Volume with the exact pinned Nemotron, merged
Qwen, Kyutai, and selected voice artifacts. Serving sets `HF_HUB_OFFLINE=1`, so scale-from-zero
containers read these files from Modal storage and do not redownload or query the Hub.

## Deploy

Use Python 3.12 and run Modal by module so the local `app.py` filename cannot shadow the Modal
package. On a Windows console that does not default to UTF-8, set `PYTHONUTF8` for the CLI output:

```powershell
$env:PYTHONUTF8 = '1'
modal run -m deployment.modal.voice_light::cache_models
modal run -m deployment.modal.voice_light::smoke_tool_use
modal run -m deployment.modal.voice_light::smoke_search_provider
modal deploy -m deployment.modal.voice_light
python -m deployment.modal.smoke_websocket
```

Run the tool-use smoke after prompt, schema, tokenizer, or merged-Qwen changes. It loads the exact
production checkpoint on one L40S and requires ordinary speech, calculation, current search,
explicit search, confirmed-search follow-up, and post-tool continuation cases to emit the expected
spoken text and structured Hermes calls. It validates model behavior without invoking Tavily.

Run the search-provider smoke after creating or updating the `voice-light-search` secret. It fails
before making a request unless that secret contains `VOICE_LIGHT_TAVILY_API_KEY`, then performs a
real bounded Tavily query. Its JSON output contains only `configured`, `result_count`, and
`provider_latency_ms`; it never prints the credential or result content.

The deployed endpoints are:

- HTTPS base: `https://bertil-braun-private--voicelightagent-voice-light.modal.run`
- voice WebSocket: `wss://bertil-braun-private--voicelightagent-voice-light.modal.run/v1/voice`
- Modal dashboard: `https://modal.com/apps/bertil-braun-private/main/deployed/VoiceLightAgent`

Serve the browser locally, then put the WebSocket URL in the endpoint field or query string:

```powershell
$env:VOICE_LIGHT_RELOAD = 'false'
python -m app.local.server
```

```text
http://127.0.0.1:8000/voice-agent?compute=wss%3A%2F%2Fbertil-braun-private--voicelightagent-voice-light.modal.run%2Fv1%2Fvoice
```

## Turn-policy configuration

The deployed starting values are:

| Environment variable | Value | Meaning |
| --- | ---: | --- |
| `VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS` | `1` | Nemotron streaming lookahead used in training and inference |
| `VOICE_LIGHT_FLOOR_TAKE_THRESHOLD` | `0.82` | predicted floor take that commits interruption |
| `VOICE_LIGHT_NON_FLOOR_FEEDBACK_THRESHOLD` | `0.82` | predicted feedback that resumes the same generation |
| `VOICE_LIGHT_OVERLAP_CLASSIFICATION_DEADLINE_MS` | `500` | conservative unresolved-overlap deadline |
| `VOICE_LIGHT_OVERLAP_FINALIZATION_GRACE_MS` | `120` | maximum wait for a prompt final transcript before an empty provisional overlap resumes |
| `VOICE_LIGHT_TRANSCRIPT_FREE_FLOOR_TAKE_DEADLINE_MS` | `1200` | hard deadline for sustained overlap without transcript evidence |
| `VOICE_LIGHT_OVERLAP_REARM_SILENCE_MS` | `160` | clean Silero silence required before a resolved backchannel can open another overlap |
| `VOICE_LIGHT_MAXIMUM_PREDICTION_LAG_MS` | `240` | maximum age of causal adapter evidence during active overlap |
| `VOICE_LIGHT_MAXIMUM_TRANSPORT_AHEAD_MS` | `1200` | maximum PCM duration released ahead of browser playback credit; this absorbs intermittent streaming-TTS cadence without delaying worklet-side cancellation |

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

The browser sends authoritative playback-clock credit every 80 ms. PCM release waits outside the
WebSocket send lock once the configured transport window is full, so duck, pause, resume, and cancel
commands are not queued behind an entire synthesized response. Cancellation wakes blocked senders
and rejects stale-generation PCM; it does not delete valid audio or advance audible-only history.

## Validation and measured deployment results

The final validation record is dated 2026-09-09. Automated route-level WebSocket coverage uses the
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

## Known limitations

- The latest truthful cold readiness sample is 72.858 seconds, of which 58.048 seconds was model
  load plus first-inference warmup. Modal scheduling and fallback GPU selection remain variable; an
  earlier A100 fallback required 105.341 seconds end to end. Restoring the old approximately
  ten-second behavior requires consolidating repeated Python/CUDA worker bootstrap, keeping a warm
  container (which conflicts with scale-to-zero), or replacing the larger current model stack;
  cached weights alone cannot remove library initialization. GPU memory
  snapshots remain an alpha, fixed-GPU canary candidate, not a production setting: each fallback
  GPU must prove coherent restoration of all three CUDA subprocesses and at least a 30% readiness
  improvement without first-turn or p95 regression.
- A live human must provide microphone speech and judge audible output. Automated and agent-run
  checks cannot honestly certify microphone capture, speaker audibility, natural backchannel, or
  interruption perception.
- Speech-end-to-first-text/audio and live duck, cancellation, and resume latency require a ready
  GPU session plus timestamped real audio; no synthetic result is presented as production proof.
- The adapter currently catches up from a bounded 300 ms pre-roll after Silero onset. Continuous
  assistant-playback context requires replacing the high-level RNNT generation loop with one
  scheduler that owns persistent encoder, decoder, and adapter state; a side encoder loop cannot
  safely share the private Hugging Face streaming caches and was not added.
- Real web search requires `VOICE_LIGHT_TAVILY_API_KEY` in the separately managed
  `voice-light-search` Modal secret.
- Cache preparation works without `HF_TOKEN` but may be rate-limited; serving itself is offline.
