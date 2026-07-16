# Speculative Speech During Tool Use

Reduce a voice agent's perceived latency by committing a short, speakable opening before tool calls finish, then generate the rest of the response with the tool results while text-to-speech is still playing the opening.

## Motivation

A cascaded voice agent usually serializes too much work:

```text
User finishes speaking
    → model decides whether to use a tool
    → tool runs
    → model writes the full answer
    → text-to-speech begins
```

Even when the final answer is fast enough, the silent interval before speech begins can make the agent feel unresponsive. Speech creates an opportunity that text chat does not have: generating and playing audio takes time. Work needed for the later part of an answer can run during that playback window.

## Core Idea

Split response production into a committed opening and a tool-informed continuation:

```text
Conversation state
    ├─→ predict a short, safe opening → TTS → immediate playback
    └─→ emit internal thinking/tool event → run tools
                                              ↓
Opening + tool results → generate continuation → TTS stream
```

The opening might be one short sentence, or even a clause, that is useful regardless of the exact tool result. After committing it to speech, the model emits an internal control event—conceptually a thinking token or tool-call token—that starts Python, an API request, retrieval, or another background operation. The continuation is then conditioned on both the already-spoken prefix and the tool result.

The control event must never enter the spoken text stream. It should be represented as a typed orchestration event rather than relying on a literal token being filtered out late in the audio pipeline.

## Central Constraint: Prefix Commitment

Once playback begins, the opening cannot be edited. It therefore must not make claims that a pending tool might contradict. Good speculative openings establish intent, frame the next step, or state information already supported by conversation history.

Examples of relatively safe openings:

- “Let me compare those options.”
- “There are two parts to that.”
- “I’ll check the current figures and walk through what they mean.”

Weak openings are empty filler repeated on every turn. Risky openings contain a predicted result, promise success, or assume a tool will return usable data.

The useful target is not merely any early audio. It is an opening that advances the conversation while preserving enough semantic freedom for the continuation.

## Possible Architectures

### Single-model staged decoding

One model emits a bounded speech segment followed by a structured tool call. The orchestrator sends the speech segment to TTS immediately, runs the tool, and resumes generation with the committed text and tool result in context.

This is the simplest system to prototype, but training or prompting must strongly separate spoken output from control events.

### Opening planner plus response model

A small, low-latency model predicts a safe opening and a tool plan. A stronger response model receives the opening, executes or awaits the plan, and writes the continuation.

This makes latency and commitment policy easier to tune independently, but introduces disagreement risk between the planner and response model.

### Draft-and-verify

A fast model drafts an opening. Before playback, a verifier checks whether every claim is supported without the pending tool result and whether the continuation has sufficient freedom. Rejected drafts fall back to the normal tool-first path.

Verification adds latency, so it is only useful if it remains much faster than the tool call it overlaps.

## Orchestration Requirements

The runtime needs explicit states rather than treating the response as one text stream:

1. Decide whether this turn is eligible for speculative speech.
2. Produce a maximum-length opening and a structured background-work request.
3. Commit the opening and begin TTS.
4. Run tools concurrently with TTS generation and playback.
5. Generate a continuation that starts naturally after the exact committed opening.
6. Queue continuation audio without a perceptible gap or duplicated words.
7. Fall back gracefully if tools fail or exceed the available playback window.

Useful eligibility signals include expected tool latency, confidence that a safe opening exists, the semantic risk of the domain, and the estimated playback duration of the opening.

## Failure Modes

- **Contradiction:** the opening implies an answer that the tool result disproves.
- **Filler behavior:** the agent repeatedly says generic phrases that users learn to ignore.
- **Seam artifacts:** the continuation restarts, changes tone, or does not grammatically follow the opening.
- **Tool overrun:** playback ends before the result is ready, recreating a silent gap mid-response.
- **Tool underrun:** the result arrives immediately and the forced opening only makes the answer longer.
- **Cancellation:** the user interrupts while tools continue consuming resources in the background.
- **Error disclosure:** a failed tool leaves the continuation unable to honor what the opening promised.
- **Prosody mismatch:** separately synthesized segments sound like separate turns.

High-stakes answers should require a stricter policy or disable speculative claims entirely.

## First Experiment

Build a replayable benchmark from real tool-using turns and compare three paths:

1. tool-first baseline;
2. fixed acknowledgment followed by tool use;
3. model-predicted, verified opening followed by tool use.

Start with one deterministic, artificial tool whose latency can be varied. Cap openings by both tokens and predicted audio duration. Preserve the exact event timeline so text generation, tool execution, TTS generation, playback, and gaps can be inspected separately.

Measure:

- time from end of user speech to first audio;
- time to first useful spoken content;
- silent gap before the continuation;
- total answer completion time;
- opening rejection and fallback rate;
- contradiction or unsupported-claim rate;
- grammatical and prosodic seam quality;
- user preference against the normal tool-first response.

The first question is whether a model can reliably produce openings that are both useful and result-independent. If it cannot, the safer version of this idea is to overlap only nonverbal feedback or a small set of context-specific acknowledgments with tool execution.

## Relationship to the Voice Agent

The current streaming ASR–LLM–TTS prototype already exposes the boundaries needed to measure this idea: end of user speech, model text deltas, TTS boundaries, playback progress, and cancellation. A prototype should keep the speculation policy in orchestration code and avoid coupling it to any one LLM or TTS backend.
