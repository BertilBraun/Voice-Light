# Streaming Voice Agent Prototype

The prototype page is served by the local application at `http://127.0.0.1:8000/voice-agent`.
The endpoint field on that page must point directly at the compute WebSocket, for example
`ws://<compute-host>:8000/v1/voice`. The local application only serves the page; it does not proxy
or orchestrate voice sessions.

## Session ownership

The compute runtime admits exactly one live voice WebSocket. A second connection is closed with
retryable WebSocket code `1013` until the active session releases its lease. Admission is released
in the route's `finally` block on normal stop, disconnect, or failure.

The admitted WebSocket creates one ephemeral `VoiceSession` and one conversation-scoped
`SpeechUnderstandingSession`. `VoiceSession` owns the per-session Silero iterator, pre-roll buffer,
turn policy, ordered conversation, Qwen generation, Kyutai TTS generation, generation identifiers,
playback estimate, and cancellation. It does not own a Nemotron decoder process or a provisional
turn detector directly. The speech-understanding session owns streaming epochs, logical-turn
epochs, transcript-revision state, the current transcription lease, optional-detector queues, and
the normalized evidence stream. Closing the WebSocket destroys both conversation-scoped objects.

The application-scoped `SpeechUnderstandingProvider` owns the persistent backend resources.
Today `CompositeSpeechUnderstandingProvider` owns the existing `NemotronStreamingTranscriber`,
whose `RestartingNemotronWorkerManager` owns the persistent Nemotron subprocess. An optional
application-scoped `TurnPredictionProvider` may similarly create a standalone detector source for
each conversation. Provider shutdown, not WebSocket shutdown, closes those managers and their
processes. A speech-understanding session lazily acquires the provider-owned Nemotron worker lease
when audio first reaches a logical turn, releases it when that turn is finalized or the session
fails, and uses the same manager for the next logical turn. Models are never spawned or reloaded per
WebSocket or per turn.

The compute parent remains CUDA-free for live voice orchestration. Persistent Nemotron, Qwen, and
Kyutai subprocesses own their respective CUDA models and outlive a connection. The loaded Silero
model also outlives a connection, while a fresh iterator carries each session's VAD state. There is
no persistence, session re-entry, or reconnection protocol.

The browser captures microphone PCM, renders server events, plays server PCM, and reports playback
progress. The server remains authoritative for VAD, turn boundaries, cancellation, and which
validated playback offsets enter model history.

## Pipeline and turn policy

- The browser continuously sends mono 16 kHz PCM16 as binary WebSocket messages.
- Server-side Silero uses threshold `0.4`, 250 ms minimum silence, and 100 ms speech padding.
- The session retains 300 ms of pre-roll and waits another 500 ms after Silero becomes inactive.
  Thus the configured acoustic silence before commitment is approximately 750 ms, plus frame
  scheduling, rather than 500 ms total.
- NVIDIA Nemotron Speech Streaming English 0.6B uses its 160 ms lookahead configuration until the
  session finalizes the utterance. Its persistent child process has one strictly owned active ASR
  stream. The worker's cumulative text streamer decodes the complete token cache on every update,
  including a trailing word that Hugging Face's whitespace-oriented `TextIteratorStreamer` would
  otherwise retain until more text or stream completion. Single-session admission makes
  cross-WebSocket serialization unnecessary. The browser uses a native 16 kHz audio context so
  its resampler converts microphone audio before PCM16 quantization. Set
  `VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS` to `0`, `1`, `6`, or `13` to compare the model's 80, 160, 560,
  and 1120 ms streaming configurations without changing code.
- Qwen3-1.7B runs in a persistent child process. A typed generation-ID protocol streams
  non-thinking text deltas for the complete ordered conversation.
- Each committed user turn emits `llm.history` with the immutable conversation snapshot supplied to
  that generation. The browser logs both a table and the complete formatted JSON snapshot to its
  developer console for prompt inspection.
- Each complete whitespace-delimited word enters Kyutai TTS while Qwen generates later text. The
  final trailing word is flushed when Qwen finishes.
- A new authoritative server speech-start still immediately cancels private or inaudible queued
  work. When browser-authoritative state says a committed response is audibly speaking, the start
  instead opens unresolved overlap: the server sends an immediate duck and boundary-pause command
  while the same ASR turn receives preserved pre-roll and subsequent user audio. Only a
  response-requiring or floor-taking resolution invokes the existing Qwen/TTS teardown barrier.
  Before starting a successor generation, the session still awaits that barrier, so old and new
  Qwen/TTS work never overlap accidentally.

### Natural user overlap and reversible playback

Playback, generation, speech understanding, and durable history remain separate:

- Speech understanding continues to receive immutable `CapturedAudioChunk` observations with the
  playback condition known at capture time. ASR remains mandatory. Optional detector degradation
  leaves ASR, Silero, and the fallback overlap policy active.
- `ProvisionalVadTranscriptOverlapPolicy` is the temporary replaceable classifier. Its focused
  configuration owns acknowledgement, laughter, question, stop/repair, and continuation
  vocabulary plus the 500 ms deadline. `VoiceSession` supplies timestamped VAD, transcript, and
  optional interruption evidence; it does not embed the vocabulary.
- `PlaybackController` creates typed generation-scoped commands, maintains causal server estimates,
  reconciles browser acknowledgements, rejects duplicate or stale acknowledgements, and records
  command latency and rendered-position estimate error.
- The AudioWorklet owns rendered output position, source-sample position, queued PCM, word
  boundaries, gain ramps, pause/resume state, completion, cancellation, and command idempotency.
- Generation retains Qwen/TTS ownership and its cancellation barrier. At 350 ms unresolved overlap
  it stops at the next complete generated word. TTS cannot advance more than 500 ms of source audio
  beyond the latest playback position. Resume releases both holds; cancellation uses the existing
  task and private-release barriers.
- Conversation history receives no user entry for non-floor-taking feedback. A promoted
  interruption retains the original ASR turn, pre-roll, transcript revision lineage, and onset
  sample. Assistant history advances only for words proven complete by reaching the next word
  boundary, or for a fully completed generation.

The temporary policy behaves as follows:

- the first positive Silero observation during audible playback issues a 25 ms ramp to -18 dB and a
  pause request for the next known word boundary, capped by a rendered-output deadline 120 ms after
  the latest browser checkpoint;
- a transcript-free burst, configured closed acknowledgement, or short laughter that has ended is
  ephemeral and resumes the exact first unplayed sample;
- `how`, `what`, explicit stop/repetition language, `no`, `wait`, `actually`, and other meaningful
  non-acknowledgement lexical material take the response-required fast path;
- an acknowledgement followed by more material, including `yeah, but`, takes the floor;
- speech still active at 500 ms takes the floor even without transcript evidence.

The browser states are `IDLE`, `QUEUED`, `SPEAKING`, `DUCKING`, `PAUSED_BUFFERED`, `RESUMING`,
`DRAINING_TO_BOUNDARY`, `CANCELLED`, and `COMPLETED`. A paused generation does not advance either
playback cursor. Resume preserves the resampler fraction and begins at the exact first unplayed
source sample. Cancellation clears queued PCM and permanently raises the rejected-generation
watermark. Terminal generations cannot reactivate, older-generation commands cannot affect a
replacement, and repeated command IDs return the cached acknowledgement without applying the
operation again.

The initial budgets are 500 ms target paused-buffer age, 800 ms absolute resume age, 500 ms maximum
synthesized-ahead audio, a generation hold at 350 ms, and mandatory resume or cancellation at
500 ms. A resume rejected by either the server or worklet becomes cancellation.

Deterministic validation on 2026-07-17 produced the following development measurements:

| Scenario | Controller/transport observation | Sample engine observation |
| --- | ---: | --- |
| `mm-hm` resume | onset-to-duck acknowledgement 6.36 ms; onset-to-pause 7.85 ms; onset-to-resume 10.58 ms | no duplicated or skipped source samples |
| `How?` fast repair | onset-to-cancel acknowledgement 5.76 ms | cancellation stops before the next render quantum |
| Forced pause | command deadline reached at the exact requested rendered sample | partial word remains unacknowledged and resumes exactly |
| Gain duck | command-to-acknowledgement 7.67 ms in the in-process trace | configured ramp consumes exactly 20 ms in the harness; product default is 25 ms |

Each in-process value above came from a deterministic single-scenario trace, so it is not a
statistical production p95. The harness excludes a real browser event loop, device scheduling, and
network latency. The server now records those complete live paths from the original Silero
observation; production p95 acceptance for 50 ms duck, 120 ms pause, and 120 ms explicit stop
therefore requires a browser/GPU deployment run. The implementation does not shift the timing
origin to hide that unmeasured path.

### Predictive generation

`VoiceSession` accepts a stable `SpeechUnderstandingProvider`. It consumes normalized transcript
and interaction-evidence events and does not know whether they came from separate processes or one
integrated inference step. No turn-taking checkpoint or canned predictor is required. By default,
Silero's first inactive decision arms a causal
`InteractionPrediction`. After a sample-counted 100 ms debounce, the session starts speculation
while ASR remains open and the existing additional 500 ms silence guard continues toward
commitment. The debounce lets Nemotron publish trailing text without blocking audio ingestion and
is configurable through `VOICE_LIGHT_VAD_SPECULATION_DEBOUNCE_MS`.
`SessionPolicy.vad_speculation_enabled` or `VOICE_LIGHT_VAD_SPECULATION_ENABLED=false` disables
this early start to retain the original non-speculative path as a measurable baseline.

Incremental ASR snapshots are tracked as a monotonic revision lineage with a stable prefix and
volatile suffix. Crossing the configurable speculative yield threshold starts at most one Qwen/TTS
candidate, anchored to the immutable conversation snapshot, revision ID, prediction, input sample
position, and monotonic creation time. Trained predictions use the stable prefix. The VAD endpoint
uses the complete current `stable_prefix + volatile_suffix` text. A later change to how that same
text is divided between the two fields does not invalidate the VAD candidate; a change to their
normalized concatenation does. Speculation does not call `finish()`—Nemotron remains open until the
high-confidence prediction rule or guarded silence commitment fires.

Candidate output passes through a private release gate. Text deltas, word boundaries, PCM bytes,
and their original offsets are retained in production order but are not sent to the browser and do
not enter durable history. At commitment the server finalizes ASR and conservatively validates the
final text without asking Qwen to judge its own response. An unchanged stable anchor with the same
normalized lexical request is promoted; otherwise Qwen and TTS are cancelled, their teardown
barrier is awaited, and a new authoritative generation receives the next monotonic generation ID.
Promotion releases buffered output in order and turns the gate into a pass-through for a candidate
that is still streaming.

Candidates move through explicit created, prefilling, streaming, ready, committed, invalidated,
cancellation-requested, cancelled, and failed states. Resumed user activity, a changed VAD-anchored
transcript, a revised trained-predictor stable prefix, a decisive return to hold/continuation,
floor-taking overlap, shutdown, or model failure invalidates speculative work. Backchannel
probability alone does not invalidate committed output. The overlap controller now owns audible
duck, pause, resume, and yield behavior without moving those decisions into model inference.

### Speech-understanding integration contract

`SpeechUnderstandingSession.add_audio()` accepts one immutable `CapturedAudioChunk`. Its PCM16
payload must exactly match `[start_input_sample, end_input_sample)`. It also carries a monotonic
sequence number and observation timestamp, `stream_epoch`, `turn_epoch`, the current typed Silero
evidence, and the current `PlaybackCondition`.

`PlaybackCondition` is a causal input, not a detector output. It contains its own event ID, optional
generation ID, `PlaybackState`, whether the assistant is believed audible, the latest rendered
output and source sample positions, the browser output rate when known, its monotonic observation
time, and an authority value. Commands immediately create causal `SERVER_ESTIMATED` state.
Playback start/progress/completion messages and matching command acknowledgements move the latest
condition to `BROWSER_AUTHORITATIVE`. Later server actions return it to an estimate until the next
acknowledgement. Past audio observations retain their original condition. The compatibility
`InteractionPrediction.assistant_playback_state` field is deprecated and, where still populated,
mirrors this input instead of being hardcoded to `IDLE`.

The session exposes an asynchronous `SpeechUnderstandingEvent` stream. `TranscriptRevision`,
`YieldEvidence`, `FutureActivityEvidence`, `TurnEventEvidence`, and
`OverlapDispositionEvidence` are sibling events. Status, degraded, and abstained results are
explicit events. Evidence models are partial by design: a backend emits only the probability types
it supports. The current policy-side `InteractionPredictionReducer` creates the legacy convenient
snapshot only when the provisional standalone source supplied all fields required by the existing
threshold policy. Product thresholds are unchanged.

Every event stamp records stream and turn epochs, an inference step, an observation ID, exact input
sample bounds, optional encoder-frame bounds, the input sample observed through including
lookahead, observation and emission monotonic times, source/model revision, optional conditioned
transcript revision, and optional conditioned playback event. A transcript parent is required only
when a standalone lexical detector actually consumed that revision. Merely having a latest
revision does not create a causal parent. This permits integrated ASR and turn-head outputs from one
encoder step to be siblings. A speculative candidate separately records the prediction-evidence
event ID and the transcript revision used to build its prompt.

The existing `TurnPredictionSource` survives only as a provisional internal adapter behind
`CompositeSpeechUnderstandingSession`. It receives the captured chunk plus the latest available
transcript revision. Its legacy full prediction is normalized into sibling evidence events. The
first speech-bearing pre-roll observation remains ASR-only to preserve the validated predictive
generation timing from the earlier direct integration.

The orchestration policy interprets that output as follows:

- sufficient confidence plus `p_user_yield >= speculative_yield_threshold` may create one private
  Qwen/TTS candidate;
- sufficient confidence plus `p_user_yield >= commitment_yield_threshold` commits the turn and
  finalizes ASR;
- sufficient confidence plus `p_user_speech >= decisive_hold_threshold` blocks silence commitment
  and invalidates an uncommitted candidate;
- `p_user_interruption >= floor_taking_overlap_threshold` invalidates an uncommitted candidate.

Speech understanding predicts and transcribes only. It does not start Qwen, release output, mutate
conversation history, or manage playback. It finalizes the backend's logical speech turn only when
`VoiceSession` asks it to do so. `VoiceSession` retains policy decisions, the single-Qwen teardown
barrier, and the Silero endpoint fallback.

### Integrated Nemotron seam

`SpeechUnderstandingProvider`, `SpeechUnderstandingSession`, and their explicit
`IntegratedNemotronSpeechUnderstandingProvider/Session` protocols are the factory seam for the
future same-pass backend. No integrated runtime is claimed in this prototype, and no second
Nemotron encoder pass has been added.

The eventual integrated worker must keep FastConformer attention/convolution caches, RNNT decoder
state, tapped encoder features, and the turn adapter's recurrent state inside the persistent
Nemotron process. One cache-aware encoder step will fan out to RNNT decoding and the adapter, then
emit transcript and interaction siblings with the same inference/observation identity. Process
ownership, lazy restart after a failed lease, and application shutdown remain provider and worker
manager responsibilities; conversation and epoch state remain session responsibilities.

### Initial GPU evaluation

The cumulative streamer was first evaluated with no additional VAD debounce, then with the
sample-counted 100 ms debounce. Node-local warm trials used three fixed utterances and excluded
USA-to-browser network latency:

| Utterance | No debounce | 100 ms debounce |
| --- | ---: | ---: |
| “What is the capital of France?” | 2/5 promoted | 5/5 promoted |
| “What is two plus two?” | 0/5 promoted | 5/5 promoted |
| “Set a timer for ten minutes.” | 5/5 promoted | 5/5 promoted |
| **Total** | **7/15 (46.7%)** | **15/15 (100%)** |

The capital-question commit-to-first-PCM p50 improved from approximately 968 ms to 191 ms. The
first promoted arithmetic trace reached PCM in 331 ms; repeated server playback acknowledgements
had a median of approximately 619 ms, versus approximately 999 ms without a usable candidate. The
timer p50 moved from approximately 404 ms to 538 ms, reflecting current TTS/CPU variance rather
than a consistent improvement. Promoted candidates hid approximately 0.48–0.57 seconds and 2–9
Qwen tokens before commitment. Voxtream had not produced PCM before commitment in these trials, so
the measured benefit came from hidden Qwen prefill/generation rather than buffered audio. No
speculative text or PCM escaped before promotion, and the observed stale-candidate escape rate was
zero. These are small best-case measurements, not a trained turn-detector evaluation.

## Process and lifecycle boundaries

Runtime readiness has four stages: speech detection, streaming ASR, language model, and speech
synthesis. Each model worker must emit its typed ready event within 180 seconds or the parent
terminates it and marks the stage failed. Qwen must continue producing an event within 10 seconds
during generation and acknowledge cancellation within 2 seconds. Nemotron finalization is bounded
at 15 seconds. Kyutai generation progress is bounded at 5 seconds and cancellation at 2 seconds.

Nemotron, Qwen, and Kyutai managers grant an exclusive worker lease to one operation. A normal
terminal event releases the reusable worker. A broken pipe, malformed event, worker exception,
progress timeout, or cancellation timeout terminates the failed child, clears ownership, and
releases the lease. Replacement is lazy: the next operation constructs and validates a fresh child,
so cold model loading cannot delay delivery of the original error. Kyutai validates that a lazily
restarted worker reports the original output sample rate.

Within the composite speech-understanding session, ASR is mandatory and authoritative. Any ASR
send, stream, or finalization failure remains session-fatal and uses the existing Nemotron
replacement path. The standalone interaction detector is optional. It runs behind a bounded queue
and cannot delay ASR ingestion. Queue overflow is an acoustic continuity failure: it cancels and
disables the standalone detector for the conversation rather than continuing recurrent state across
a gap. A degraded event reports the cumulative discarded-observation count while ASR plus Silero
continue. Detector exceptions likewise disable that source. Finalizing a turn cancels and reports
any lagging old-turn detector work. Inputs from old stream or turn epochs are rejected. Within a
turn, policy also rejects evidence beyond its bounded sample lag or superseded by a later
speech-bearing chunk, so an old yield cannot create or commit a candidate after speech resumes.

The compatibility reducer bounds playback-condition and incomplete evidence-group retention,
prunes both by stream/turn epoch and observed-through sample watermark, removes completed groups
immediately, and clears all retained causal input at logical-turn reset.

The session lifecycle is explicit: `created`, `connected`, `ready`, `failed`, `stopping`, and
`closed`. A generation moves through `created`, `streaming`, cancellation or stream completion, and
a terminal cancelled, failed, or playback-complete state. Receive and recognition tasks are owned
for the connection lifetime; each generation owns one text task and one synthesis-output task.
Qwen text and Kyutai audio run as a pipeline: complete words are sent to TTS while Qwen continues
generating later text.

Worker exceptions and orchestration failures are logged with session/generation context and sent to
the development browser as structured `error` events containing `component`, `operation`, optional
`generation_id`, `retryable`, and `message`. The browser prints the complete error object and shows
the fields in its connection status. Failures are not converted into successful empty output.

Committed assistant text appears immediately as muted text; speculative text remains private until
promotion. Kyutai's delayed-stream state exposes
the model step at which each input word starts. The server converts that step at Mimi's 12.5 Hz frame
rate into a generation-relative PCM sample offset and sends the boundary to the playback worklet.
Kyutai's delayed zero-code frames are decoded to prime Mimi but omitted from the transmitted PCM,
so playback sample zero begins after the model's built-in text/audio delay instead of adding 1.28
seconds of client-side silence.
When playback reaches the start of the next word, the page highlights and acknowledges the previous
original word, including attached punctuation. This proves the previous word completed. A forced
mid-word pause records `forced_sample` and does not durably acknowledge that partial word. The
server validates generation ID, text offset, boundary sample, played source position, command
identity, and monotonicity. Interrupted entries retain only the proven word prefix with a trailing
`...`; generated but unplayed text remains visible and muted but does not enter model history.
Completely drained playback commits the full response without an ellipsis. The step/sample
conversion follows Kyutai's documented frame rate and still requires codec-priming calibration on
the target GPU deployment.

## Protocol

Browser-to-server JSON events:

- `session.start` with `input_sample_rate` (currently exactly 16000)
- `playback.progress` with `generation_id`, `text_offset`, `boundary_start_sample`, and
  `played_sample_count`, browser monotonic time, rendered output position, and output rate
- `playback.acknowledgement` with command/generation/epoch identity, requested action, resulting
  state, browser monotonic time, rendered and source positions, output rate, boundary versus forced
  pause result, gain-ramp state, buffered/discarded/replayed/skipped samples, and resume rejection
- compatibility `playback.stopped` with final positions for older clients
- `playback.complete` with generation ID and final browser positions
- `session.stop`

All microphone audio uses unwrapped binary PCM16 messages. A bounded server queue decouples socket
ingestion from recognition and applies backpressure if the single ASR worker cannot keep up.

Server-to-browser JSON events include `session.ready`, VAD boundaries, partial/final transcripts,
turn commitment, the exact `llm.history` snapshot, assistant text deltas, word-start audio
metadata, generation audio boundaries, typed `playback.command` controls, and errors. Each playback
command carries a unique command ID, generation ID, action, server issue time, causal evidence
ID/source, stream and turn epochs, confidence, and action-specific sample, gain, or age limits.
`assistant.audio.text_boundary`
maps an original generated-text offset to a generation-relative PCM start sample. Assistant binary
frames begin with three little-endian unsigned 32-bit integers: generation ID, sequence number, and
the chunk's generation-relative start sample. Remaining bytes are mono
PCM16 at the `output_sample_rate` announced by `session.ready`. The browser rejects stale/cancelled
generations and out-of-sequence frames. Its playback worklet resamples from the announced server
rate to the browser AudioContext rate while retaining generation-relative input sample progress.

For microphone-path diagnostics, the page retains the exact 16 kHz PCM buffers that it successfully
sends over the WebSocket. When a session stops or disconnects, it exposes the buffers as a playable
and downloadable mono PCM16 WAV file. Starting another session releases the previous recording.

The voice research WebSocket is intentionally unauthenticated so the browser can connect directly.
Other compute HTTP APIs retain bearer-token authentication. The compute service rejects voice
connections while its models are not ready and rejects concurrent live voice connections with code
`1013`.

## Predictive latency instrumentation

Each candidate records monotonic timestamps together with input/output media positions for the
first endpoint observation, speculation, Qwen start and first complete word, first TTS word and PCM,
turn commitment, ASR finalization, promotion/invalidation, first released PCM, and first browser
playback acknowledgement. Session summaries report candidate hit and invalidation rates with
reasons, stale-candidate escapes, commit-to-first-playback p50/p90/p95, ground-truth end latency
when annotations are supplied, hidden pre-commit work, wasted Qwen output tokens and TTS samples,
baseline latency without a candidate, and latency after invalidation. Qwen token accounting uses
the worker tokenizer's cumulative tokenization of decoded response text rather than word or
character estimates.

Overlap summaries additionally report onset-to-duck, onset-to-pause, explicit-stop,
onset-to-resume, cooperative and competitive overlap duration, paused-buffer age, synthesized,
buffered, discarded, replayed, and skipped samples, command-to-acknowledgement latency, and
rendered-position estimate error. All latency origins remain the causal Silero observation or
server command issue time rather than a later acknowledgement.
