# Modal Voice Deployment Runbook

## Architecture

`deployment.modal.voice_light` is a thin Modal ASGI wrapper around the current
`app.compute` runtime and application factories. It does not restore the deleted
`app.voice_agent` runtime. The current `/v1/voice` route, `ComputeRuntime`, `VoiceSession`, tool
registry, search integration, predictive generation, playback controller, Nemotron ASR, Qwen
workers, and Kyutai TTS remain authoritative.

The GPU container admits one Modal input and the compute route separately enforces one live voice
session. Modal requests L40S first, then allows H100 or A100 when L40S capacity is unavailable. All
three have sufficient memory and CUDA compatibility for the measured stack; fallbacks reduce
scheduling stalls but can cost more per second. Modal may scale to zero, has one maximum container,
and keeps an idle container for 1,200 seconds. Modal sets
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

## Account configuration

Create the named secret without putting credentials in the repository:

```powershell
modal secret create voice-light-compute `
  VOICE_LIGHT_COMPUTE_TOKEN='<random bearer token>'
```

`VOICE_LIGHT_COMPUTE_TOKEN` protects the HTTP APIs. The browser voice WebSocket is intentionally
unauthenticated because the page connects directly. Public model downloads work without
`HF_TOKEN`, but authenticated downloads have higher Hub rate limits. Add `HF_TOKEN` and
`VOICE_LIGHT_TAVILY_API_KEY` to the same secret in the Modal dashboard when needed. Without the
Tavily key, the search tool reports its typed unavailable result. Updating a Modal secret restarts
dependent containers.

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
modal deploy -m deployment.modal.voice_light
python -m deployment.modal.smoke_websocket
```

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

The threshold is the evaluated Voice-Light starting point, not a universal calibration. Silero
onset always causes the immediate reversible duck/pause. Strong floor-take evidence commits
cancellation; strong non-floor-feedback evidence resumes the same generation without a user turn;
the deadline preserves the conservative fallback. The browser debug panel shows Silero state,
turn completion, floor take, non-floor feedback, policy decision, and decision latency. These are
ephemeral events and never enter durable audible-only conversation history. Its rolling 20-second
timeline advances every 80 ms during user speech, silence, and assistant playback, and merges causal
adapter evidence at Nemotron's approximately 169 ms encoder cadence; it does not invent
interpolated model predictions.

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
| `ruff format app deployment tests` | 537 files formatted or already formatted |
| `ruff check --fix app deployment tests` | all checks passed |
| voice-agent, turn-taking, compute route smoke, configuration, and boundary Pytest suites | 537 passed, 4 skipped |
| Modal deployment tests (Modal-enabled Python environment) | 4 passed |
| `node --test tests\browser\*.test.mjs` | 23 passed |

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

## Known limitations

- Sampled sequential cold startup varies from approximately 39 to 80 seconds; the latest
  instrumented sample was 46.823 seconds. Restoring the old
  approximately ten-second behavior requires consolidating repeated Python/CUDA worker bootstrap
  or separating the workers into independently snapshot-compatible services; cached weights alone
  cannot remove library initialization.
- A live human must provide microphone speech and judge audible output. Automated and agent-run
  checks cannot honestly certify microphone capture, speaker audibility, natural backchannel, or
  interruption perception.
- Speech-end-to-first-text/audio and live duck, cancellation, and resume latency require a ready
  GPU session plus timestamped real audio; no synthetic result is presented as production proof.
- Real web search requires the account owner to add `VOICE_LIGHT_TAVILY_API_KEY`.
- Cache preparation works without `HF_TOKEN` but may be rate-limited; serving itself is offline.
