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

The browser is a thin streaming client. It captures microphone PCM, renders server events, plays
server PCM, and acknowledges when playback has drained. It does not make VAD, turn, cancellation,
or history decisions.

## Pipeline and turn policy

- The browser continuously sends mono 16 kHz PCM16 as binary WebSocket messages.
- Server-side Silero uses threshold `0.4`, 250 ms minimum silence, and 100 ms speech padding.
- The session retains 300 ms of pre-roll and waits another 500 ms after Silero becomes inactive.
  Thus the configured acoustic silence before commitment is approximately 750 ms, plus frame
  scheduling, rather than 500 ms total.
- NVIDIA Nemotron Speech Streaming English 0.6B receives stateful 80 ms chunks until the session
  finalizes the utterance. One worker serializes active ASR streams across WebSocket sessions.
- Qwen3-1.7B receives the complete ordered conversation and streams non-thinking text deltas.
- Completed sentence chunks enter Pocket TTS while Qwen continues generating later text.
- A new authoritative server speech-start cancels the active Qwen/TTS task and tells the browser
  to discard that generation's queued audio.

An assistant response enters conversation history only after its audio has been completely sent
and the browser reports that playback drained. If the user interrupts generation or playback, the
partial assistant response is deliberately not committed. This avoids giving later turns context
that the user did not hear, at the cost of omitting a partially heard interrupted response.

## Protocol

Browser-to-server JSON events:

- `session.start` with `input_sample_rate` (currently exactly 16000)
- `playback.complete` with `generation_id`
- `session.stop`

All microphone audio uses unwrapped binary PCM16 messages. A bounded server queue decouples socket
ingestion from recognition and applies backpressure if the single ASR worker cannot keep up.

Server-to-browser JSON events include `session.ready`, VAD boundaries, partial/final transcripts,
turn commitment, assistant text deltas, audio boundaries, cancellation, and errors. Assistant
binary frames begin with two little-endian unsigned 32-bit integers: generation ID and sequence
number. Remaining bytes are mono PCM16 at the `output_sample_rate` announced by `session.ready`.
The browser rejects stale/cancelled generations and out-of-sequence frames. Its playback worklet
resamples from the announced server rate to the browser AudioContext rate.

The voice research WebSocket is intentionally unauthenticated so the browser can connect directly.
Other compute HTTP APIs retain bearer-token authentication. The compute service rejects voice
connections while its models are not ready.
