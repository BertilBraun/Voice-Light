# Modal and Streaming Turn-Taking Restoration Design

## Audit

The live browser connects directly to the compute application's `/v1/voice` WebSocket. The
authoritative path is `app.compute.main` -> `ComputeRuntime` -> `VoiceSession`; it already owns
Silero VAD, streaming Nemotron ASR, Qwen generation and sequential tool rounds, search, Kyutai or
VoxStream TTS, predictive generation, browser playback acknowledgements, audible-only history,
and cancellation barriers.

Commit `52c51bd081bcca1d29abb58b628d2efd2bc075ab` deployed an older `app.voice_agent` application
at `/session`. It used one L40S container, one concurrent input, a persistent model-cache volume,
and scale-to-zero. Commit `bd22a86f` intentionally removed that runtime. Restoration therefore
adds only a deployment wrapper under `deployment/modal`; it does not restore the deleted package
or its orchestration.

`CompositeSpeechUnderstandingProvider` already accepts optional turn predictions, while
`ComputeRuntime` currently passes `None`. The final adapter was trained against hidden states from
Nemotron encoder layers 6, 12, 18, and 24 with one lookahead token and an 80 ms encoder frame.
The current persistent Nemotron worker owns the identical model and revision.

## Runtime design

Nemotron remains a single subprocess and a single 0.6B backbone. The worker will capture the four
hidden-state taps from each cache-aware encoder call used by RNNT generation and immediately run
the adapter. It will not perform a second encoder call and will not retain or repeatedly re-encode
a rolling waveform.

The adapter will expose an incremental path whose state contains the GRU state plus the exact left
context required by each causal convolution. Chunked inference must match a whole-sequence forward
pass in evaluation mode. The state resets at each finalized user turn and session close.

The typed worker protocol will carry causal observation metadata with audio and return predictions
associated with the latest fully consumed observation. The integrated speech-understanding
provider will translate worker output into the existing yield, future-activity, turn-event, and
overlap-disposition evidence. ASR events stay independent: adapter load or inference failure emits
one degradation event and disables adapter inference for that session without terminating ASR.

The five event-head outputs retain their training order: turn completion, continuation pause,
other, non-floor feedback, and floor take. The four future-activity logits map to consecutive
training horizons. Model evidence is sent only through the debug stream and is never appended to
conversation history.

## Interaction policy

Silero onset remains the immediate acoustic trigger. During audible assistant playback it issues a
reversible duck followed by a bounded pause. Adapter floor-take evidence at or above the configured
threshold commits interruption and cancels generation/playback. Non-floor-feedback evidence
resumes the same generation without creating a user turn. Unresolved overlap uses the existing
conservative timeout fallback.

The initial floor-take threshold is 0.82, based on the Voice-Light evaluation rather than a claim
of universal calibration. Thresholds, prediction lag, and overlap deadlines remain validated,
typed configuration loaded from environment variables.

## Modal deployment

The new Modal module will wrap `app.compute.main:app`, mount persistent Hugging Face, Torch, and
runtime cache volumes, package the adapter checkpoint at a stable container path, inject named
Modal secrets, and preserve one-session admission with one concurrent ASGI input, one maximum
container, and scale-to-zero. The browser continues to accept the deployed WebSocket URL directly.
Deployment and smoke-test commands, measured results, and any account-owned secret setup belong in
the streaming voice-agent runbook.
