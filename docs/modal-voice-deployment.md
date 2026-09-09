# Modal Voice Deployment Runbook

## Architecture

`deployment.modal.app` is a thin Modal ASGI wrapper around
`app.compute.main:create_app_from_environment`. It does not restore the deleted
`app.voice_agent` runtime. The current `/v1/voice` route, `ComputeRuntime`, `VoiceSession`, tool
registry, search integration, predictive generation, playback controller, Nemotron ASR, Qwen
workers, and Kyutai TTS remain authoritative.

The L40S container admits one Modal input and the compute route separately enforces one live voice
session. Modal may scale to zero, has one maximum container, and keeps an idle container for 1,200
seconds. Modal sets `VOICE_LIGHT_EAGER_MODEL_LOADING=true`, so the ASGI lifespan awaits sequential
model initialization before Modal marks a cold container ready or admits the first request. The
1,800-second Modal startup timeout bounds that work. Other provider-neutral deployments retain
background loading: `/health/live` can succeed before `/health/ready`, and `/v1/voice` closes with
retryable code `1013` until every required model is ready.

One persistent Nemotron subprocess owns both streaming RNNT decoding and turn-adapter inference.
The adapter consumes layer 6/12/18/24 features from the RNNT encoder call and retains incremental
causal convolution and GRU state. A second Nemotron backbone and rolling waveform re-encoding are
not used.

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

## Deploy

Use Python 3.12 and run Modal by module so the local `app.py` filename cannot shadow the Modal
package. On a Windows console that does not default to UTF-8, set `PYTHONUTF8` for the CLI output:

```powershell
$env:PYTHONUTF8 = '1'
modal deploy -m deployment.modal.app
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
ephemeral events and never enter durable audible-only conversation history.

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
| `ruff format app deployment tests` | 531 files already formatted |
| `ruff check --fix app deployment tests` | all checks passed |
| combined voice, turn-taking, route smoke, configuration, Modal, and boundary Pytest suites | 483 passed; Kyutai-extra and live-Tavily tests skipped |
| `node --test tests\browser\*.test.mjs` | 23 passed |

An early deployment that loaded in a background lifespan exposed a Modal-specific idle-suspension
problem: rejected readiness probes released the function input and stretched observed
startup-to-ready to approximately 922 seconds. The eager-startup fix removed that deadlock. On the
final cached-volume deployment, Silero loaded in 0.092 seconds, Nemotron ASR in 11.819 seconds,
conversational Qwen in 112.270 seconds, the search summarizer in 30.871 seconds, and Kyutai TTS in
18.066 seconds. Startup-to-ready and cold WebSocket admission were approximately 173.1 seconds.
These are single observations, not percentiles.

The final deployment command completed in 45.74 seconds with cached dependency layers. Its browser
smoke reached `Ready to talk`, activated the microphone, streamed 16 kHz PCM, produced Nemotron
partial/final transcripts, exercised speculative Qwen generations and invalidation/promotion, and
received four playback command acknowledgements with a 162.76 ms p95 in that single session. The
stale-candidate escape rate was zero. Continuous ambient speech caused repeated new turns before
Kyutai emitted PCM, so the session recorded zero generated audio samples and no valid
speech-end-to-first-audio, duck, cancellation, or backchannel-resume latency. Speaker audibility
and the configured-search path were therefore not claimed as live passes.

## Known limitations

- Sequential cold startup is about 173 seconds on the single measured L40S run. Parallel or
  image-snapshot model initialization is the primary deployment optimization opportunity.
- A live human must provide microphone speech and judge audible output. Automated and agent-run
  checks cannot honestly certify microphone capture, speaker audibility, natural backchannel, or
  interruption perception.
- Speech-end-to-first-text/audio and live duck, cancellation, and resume latency require a ready
  GPU session plus timestamped real audio; no synthetic result is presented as production proof.
- Real web search requires the account owner to add `VOICE_LIGHT_TAVILY_API_KEY`.
- Public Hugging Face downloads are functional but rate-limited without `HF_TOKEN`.
