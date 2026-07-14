# Streaming Voice Agent Prototype

The prototype page is served by the local application at `http://127.0.0.1:8000/voice-agent`.
The endpoint field on that page must point directly at the compute WebSocket, for example
`ws://<compute-host>:8000/v1/voice`. The local application only serves the page; it does not proxy
or orchestrate voice sessions.

## Session ownership

Each compute WebSocket creates one ephemeral `VoiceSession`. That object owns the Silero VAD,
pre-roll buffer, Nemotron decoder session, turn policy, ordered conversation, Qwen generation,
Pocket TTS work, generation identifiers, and cancellation. Closing the WebSocket destroys all of
that state. Only loaded model runtimes outlive a connection; there is no persistence, session
re-entry, or reconnection protocol.

The browser captures microphone PCM, renders server events, plays server PCM, and reports playback
progress. The server remains authoritative for VAD, turn boundaries, cancellation, and which
validated playback offsets enter model history.

## Pipeline and turn policy

- The browser continuously sends mono 16 kHz PCM16 as binary WebSocket messages.
- Server-side Silero uses threshold `0.4`, 250 ms minimum silence, and 100 ms speech padding.
- The session retains 300 ms of pre-roll and waits another 500 ms after Silero becomes inactive.
  Thus the configured acoustic silence before commitment is approximately 750 ms, plus frame
  scheduling, rather than 500 ms total.
- NVIDIA Nemotron Speech Streaming English 0.6B uses its 560 ms lookahead configuration until the
  session finalizes the utterance. One worker serializes active ASR streams across WebSocket
  sessions. The browser preserves resampling phase and boundary samples when converting microphone
  audio to 16 kHz PCM. Set `VOICE_LIGHT_ASR_LOOKAHEAD_TOKENS` to `0`, `1`, `6`, or `13` to compare
  the model's 80, 160, 560, and 1120 ms streaming configurations without changing code.
- Qwen3-1.7B receives the complete ordered conversation and streams non-thinking text deltas.
- Each committed user turn emits `llm.history` with the immutable conversation snapshot supplied to
  that generation. The browser logs both a table and the complete formatted JSON snapshot to its
  developer console for prompt inspection.
- Completed sentence chunks enter Pocket TTS while Qwen continues generating later text.
- A new authoritative server speech-start cancels the active Qwen/TTS task and tells the browser
  to discard that generation's queued audio.

Generated assistant text appears immediately as muted text. Each independently synthesized
sentence is buffered until its exact PCM sample count is known. The server sends its text and sample
metadata before its audio so interpolation is available from the first played sample. The playback
worklet reports played samples, and the page linearly interpolates character-level visual progress
within the sentence. Whenever playback crosses the end of a word, the page acknowledges that
generated-text offset. The server validates generation ID, sentence ID, word boundary, range, and
monotonicity, then updates one assistant entry in model history. Interrupted entries retain the
acknowledged whole-word prefix with a trailing `...`; generated but unplayed text remains visible
and muted but does not enter model history. Completely drained playback commits the full response
without an ellipsis.

## Protocol

Browser-to-server JSON events:

- `session.start` with `input_sample_rate` (currently exactly 16000)
- `playback.progress` with `generation_id`, `sentence_id`, and `text_offset`
- `playback.complete` with `generation_id`
- `session.stop`

All microphone audio uses unwrapped binary PCM16 messages. A bounded server queue decouples socket
ingestion from recognition and applies backpressure if the single ASR worker cannot keep up.

Server-to-browser JSON events include `session.ready`, VAD boundaries, partial/final transcripts,
turn commitment, the exact `llm.history` snapshot, assistant text deltas, per-sentence audio
metadata, generation audio boundaries, cancellation, and errors. `assistant.audio.sentence`
identifies a sentence's generation, sentence ID, generated-text range, and exact PCM sample count
before that sentence's audio frames arrive. Assistant binary frames begin with three little-endian
unsigned 32-bit integers: generation ID, sequence number, and sentence ID. Remaining bytes are mono
PCM16 at the `output_sample_rate` announced by `session.ready`. The browser rejects stale/cancelled
generations and out-of-sequence frames. Its playback worklet resamples from the announced server
rate to the browser AudioContext rate while retaining per-sentence sample progress.

For microphone-path diagnostics, the page retains the exact 16 kHz PCM buffers that it successfully
sends over the WebSocket. When a session stops or disconnects, it exposes the buffers as a playable
and downloadable mono PCM16 WAV file. Starting another session releases the previous recording.

The voice research WebSocket is intentionally unauthenticated so the browser can connect directly.
Other compute HTTP APIs retain bearer-token authentication. The compute service rejects voice
connections while its models are not ready.
