# Streaming Voice Agent Prototype

The prototype page is served by the local application at `http://127.0.0.1:8000/voice-agent`.
The endpoint field on that page must point directly at the compute WebSocket, for example
`ws://<compute-host>:8000/v1/voice`. The local application only serves the page; it does not proxy
or orchestrate voice sessions.

## Session ownership

The compute runtime admits exactly one live voice WebSocket. A second connection is closed with
retryable WebSocket code `1013` until the active session releases its lease. Admission is released
in the route's `finally` block on normal stop, disconnect, or failure.

The admitted WebSocket creates one ephemeral `VoiceSession`. That object owns the per-session
Silero iterator,
pre-roll buffer, Nemotron decoder session, turn policy, ordered conversation, Qwen generation,
Kyutai TTS generation, generation identifiers, and cancellation. Closing the WebSocket destroys all of
that state. The compute parent remains CUDA-free for live voice orchestration. Persistent Nemotron,
Qwen, and Kyutai subprocesses own their respective CUDA models and outlive a connection. The
loaded Silero model also outlives a connection, while a fresh iterator carries each session's VAD
state. There is no persistence, session re-entry, or reconnection protocol.

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
  stream. Single-session admission makes cross-WebSocket serialization unnecessary. The browser
  uses a native 16 kHz audio context so its resampler converts microphone
  audio before PCM16 quantization. Set `VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS` to `0`, `1`, `6`, or `13` to compare
  the model's 80, 160, 560, and 1120 ms streaming configurations without changing code.
- Qwen3-1.7B runs in a persistent child process. A typed generation-ID protocol streams
  non-thinking text deltas for the complete ordered conversation.
- Each committed user turn emits `llm.history` with the immutable conversation snapshot supplied to
  that generation. The browser logs both a table and the complete formatted JSON snapshot to its
  developer console for prompt inspection.
- Each complete whitespace-delimited word enters Kyutai TTS while Qwen generates later text. The
  final trailing word is flushed when Qwen finishes.
- A new authoritative server speech-start immediately requests Qwen/TTS cancellation and tells the
  browser to discard that generation's queued audio. VAD and ASR continue processing the new
  utterance while teardown runs. Before starting the successor Qwen generation, the session awaits
  a teardown barrier, so old and new Qwen/TTS work never overlap accidentally.

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

Generated assistant text appears immediately as muted text. Kyutai's delayed-stream state exposes
the model step at which each input word starts. The server converts that step at Mimi's 12.5 Hz frame
rate into a generation-relative PCM sample offset and sends the boundary to the playback worklet.
Kyutai's delayed zero-code frames are decoded to prime Mimi but omitted from the transmitted PCM,
so playback sample zero begins after the model's built-in text/audio delay instead of adding 1.28
seconds of client-side silence.
When playback consumes the first sample at or after a word start, the page highlights and
acknowledges the entire original word, including attached punctuation. The server validates the
generation ID, text offset, boundary sample, played sample count, and monotonicity. Interrupted
entries retain that optimistic whole-word prefix with a trailing `...`; generated but unplayed text
remains visible and muted but does not enter model history. Completely drained playback commits the
full response without an ellipsis. The step/sample conversion follows Kyutai's documented frame
rate and still requires codec-priming calibration on the target GPU deployment.

## Protocol

Browser-to-server JSON events:

- `session.start` with `input_sample_rate` (currently exactly 16000)
- `playback.progress` with `generation_id`, `text_offset`, `boundary_start_sample`, and
  `played_sample_count`
- `playback.stopped` with the final generation ID, text offset, and played sample count after cancel
- `playback.complete` with `generation_id`
- `session.stop`

All microphone audio uses unwrapped binary PCM16 messages. A bounded server queue decouples socket
ingestion from recognition and applies backpressure if the single ASR worker cannot keep up.

Server-to-browser JSON events include `session.ready`, VAD boundaries, partial/final transcripts,
turn commitment, the exact `llm.history` snapshot, assistant text deltas, word-start audio
metadata, generation audio boundaries, cancellation, and errors. `assistant.audio.text_boundary`
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
