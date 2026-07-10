# Natural Streaming Voice Agent Plan

## Goal

Build a natural-feeling voice assistant that responds quickly, avoids false turn starts, stops immediately when interrupted, and can eventually handle conversational backchannels without treating them as normal turns.

The system should optimize for:

- Low time to first assistant audio after the user has finished a turn.
- Accurate end-of-turn detection.
- Fast interruption handling while the assistant is speaking.
- Low false-start rate during user pauses.
- Support for user backchannels such as acknowledgements, laughter, and short continuers.
- Optional assistant backchannels that can be generated without taking a full conversational turn.

The target experience is a full-duplex spoken agent that feels closer to a live conversation than a push-to-talk assistant.

## Latency Target

The aspirational target is under 500 ms from detected user turn completion to first assistant audio.

This budget should be treated as an end-to-end metric, not only model inference time. It includes:

- End-of-turn decision latency.
- LLM first output token latency.
- TTS first audio chunk latency.
- Audio playback scheduling.
- Any buffering required for stability.

A rough initial budget:

| Component | Target |
| --- | ---: |
| End-of-turn confirmation | 50-150 ms |
| LLM first generated token after commit | 50-150 ms |
| TTS first audio chunk | 100-200 ms |
| Playback and orchestration overhead | 50-100 ms |
| Total | 250-600 ms |

The system should measure these stages separately from the beginning, because optimizing aggregate latency without attribution will make progress difficult.

## Proposed Architecture

Use a streaming cascaded architecture:

1. Streaming audio input.
2. Streaming ASR encoder and decoder.
3. Frozen ASR model with a trainable turn-taking adapter head.
4. Streaming transcript output.
5. Streaming LLM prefill while the user is still speaking.
6. LLM generation once the turn is committed.
7. Streaming TTS.
8. Full-duplex playback with interruption handling.

The core design principle is that every component should begin useful work as early as possible. ASR partials should feed the LLM context incrementally so that, when the turn detector commits the user turn, the LLM is already prefetched and ready to generate.

## Streaming Prefill Strategy

As the ASR produces stable partial text, those tokens should be committed into the LLM context and used to advance the KV cache before the user turn is complete. The goal is to avoid recomputing the system prompt, conversation context, and already-spoken stable user prefix at the exact moment the assistant needs to begin generation.

The orchestrator needs to distinguish:

- Tentative ASR tokens that may still change.
- Stable ASR tokens that can be committed to the LLM prefix.
- Final turn text that triggers assistant generation.

The cleanest interface is likely not raw string appends. The ASR stream should expose typed transcript events:

- `partial`: unstable hypothesis.
- `stable`: prefix believed unlikely to change.
- `final`: committed segment or turn.
- `revision`: correction to earlier unstable text.

The expected runtime behavior is:

1. The system prompt and prior conversation are prefetched before the user speaks.
2. Stable ASR tokens are appended and prefetched as they become available.
3. Unstable ASR hypotheses may be tracked separately but are not required to enter the durable KV cache.
4. When the turn-taking policy commits the turn, generation starts from the already-prefilled cache plus the final validated suffix.

This should make LLM startup close to a next-token prediction rather than a full prompt prefill. The exact gain needs measurement. If the prompt and context are 500-1,000 tokens, cache reuse may save a meaningful part of first-token latency.

Research is needed to determine which local LLM runtime supports the right form of streaming prefill and cache reuse. If the runtime cannot revise prefills cheaply, only stable ASR prefixes should be prefetched. Tentative unstable prefill should be treated as an optional optimization, because cache rollback and ASR revision handling may add more complexity than it saves.

## Turn-Taking Model

The turn-taking component should be tightly coupled to the streaming ASR model, but the ASR weights should remain frozen.

Attach a small trainable adapter to the frozen ASR. The adapter should consume features from multiple ASR layers:

- Early or mid encoder layers for acoustic/prosodic cues.
- Later encoder or decoder-adjacent layers for semantic cues.
- Optional text-side features from stable ASR output.

The exact attachment points depend on the selected ASR architecture. Many likely candidates will use an audio encoder such as a Conformer, but the path from encoder states to transcript tokens may vary. The adapter design should therefore stay architecture-flexible at first: define it around timestamped feature streams and stable text events rather than assuming a specific decoder implementation.

The model should predict continuously over time, not only at silence boundaries.

Candidate labels:

- `continue`: user is likely to keep speaking.
- `pause`: user paused but the turn is not complete.
- `end_of_turn`: user has completed a turn and the assistant may respond.
- `backchannel`: user produced a short acknowledgement or non-turn-taking signal.
- `interruption`: user is trying to take the floor while the assistant is speaking.

The output should be calibrated probabilities rather than hard classes. The orchestrator can then apply policy thresholds and hysteresis for different latency and false-start tradeoffs.

## Adapter Design

Start with the smallest useful model:

- Frozen ASR feature taps from selected layers.
- Projection layers to a compact shared dimension.
- A small temporal model over recent frames.
- A classification head for turn-taking labels.

Temporal model candidates:

- Small LSTM or GRU.
- Lightweight causal convolution.
- Small causal attention block with a bounded lookback window.

The adapter needs enough temporal context to distinguish a brief intra-turn pause from an actual turn ending. That context can come from recurrent state, a bounded attention window, cached convolutional state, or a combination of acoustic feature history and stable transcript history.

The first version should prioritize reliable measurements and fast iteration over architectural novelty.

Important constraints:

- The adapter must run continuously without materially increasing ASR latency.
- It must be causal or have a very small bounded lookahead.
- It should support frame-level or chunk-level predictions.
- It should expose confidence and not just a single decision.

## Orchestration Policy

The model should produce probabilities. The runtime policy decides when to act.

Initial policy behavior:

- Start LLM prefill as soon as stable ASR text appears.
- Commit the user turn only when `end_of_turn` confidence crosses a threshold.
- Suppress assistant start when `pause` or `continue` remains likely.
- Stop assistant audio immediately when user interruption confidence crosses a threshold.
- Treat user backchannels as non-interrupting unless they become sustained speech.

The first policy should use smoothing and hysteresis rather than raw frame decisions. Candidate mechanisms:

- Exponential moving average over class probabilities.
- Minimum confidence duration, such as `end_of_turn` remaining high for three consecutive frames.
- Separate thresholds for starting speech and cancelling speech.
- Cooldowns after assistant start or stop events.
- A maximum wait policy that can respond even if confidence is high but not perfect.

The policy should log both raw probabilities and smoothed decisions. This keeps the adapter trainable as a probabilistic model while allowing product behavior to be tuned without retraining.

The policy should support tuning without retraining the adapter. This allows separate iteration on model quality and product behavior.

## Backchannels

Backchannels should be treated as a separate conversational event type, not as shortened normal turns.

User backchannels:

- Should usually not interrupt the assistant.
- Should not trigger a normal assistant response.
- May update dialogue state lightly, such as marking that the user is following.

Assistant backchannels:

- Should be generated only when the user is clearly holding the floor.
- Should be short and non-disruptive.
- Should not consume the same generation path as full assistant answers unless that path can produce very low-latency audio.

Backchannels can be added after the basic turn-taking loop is measurable and stable.

## Candidate Model Research Areas

Research should focus on models that support true streaming behavior and expose intermediate features.

ASR questions:

- Which open streaming ASR models provide usable frame-level or chunk-level encoder states?
- What chunk size and stride do they use?
- Can intermediate layer features be accessed efficiently?
- How stable are their partial transcripts?
- What is their real-time factor on target hardware?

LLM questions:

- Which small local instruction models provide acceptable conversational quality around 1-2B parameters?
- Which runtime supports streaming generation and reusable KV cache prefill?
- Can tentative ASR text be revised without throwing away too much cache?
- What is the measured time to first token after a prefetched prompt?

TTS questions:

- Which streaming TTS systems produce intelligible first audio quickly?
- What is the minimum useful audio chunk size?
- How much text does the TTS require before it can start speaking?
- Can the TTS be cancelled cleanly on interruption?
- Does the voice remain natural when generation starts from very short text prefixes?

Turn-taking research questions:

- What public datasets contain conversational turn boundaries, pauses, overlaps, interruptions, or backchannels?
- Are acoustic signals available, or only transcripts?
- Are labels fine-grained enough for pause versus end-of-turn?
- Can labels be derived from speaker overlap and timing if explicit labels are unavailable?
- How well do existing end-of-turn models perform as baselines?

## Dataset Research Plan

The first dataset pass should prioritize natural dialogue data with separated speaker tracks. Track separation matters because the prototype ASR path should represent the deployment condition: the user channel should contain the user, not the assistant's generated audio mixed back into the input.

Ideal dataset properties:

- One speaker per audio track or reliable speaker-isolated channels.
- Natural conversational dialogue rather than read speech.
- Accurate timestamps, preferably human-annotated.
- Transcripts aligned to utterances or words.
- Backchannels, overlaps, interruptions, pauses, or dialogue acts when available.

Automatically generated timestamps may still be usable, but only after evaluation. If timestamps are automatic, the research pass should record the alignment method and whether we can improve or validate alignment ourselves.

The first dataset pass should collect options in four categories:

1. Multi-speaker conversational speech with timestamps.
2. Datasets with explicit backchannel or dialogue act labels.
3. Datasets with interruption, overlap, or floor-taking annotations.
4. Human-assistant or task-oriented spoken dialogue datasets as a lower-priority later addition.

For each dataset, record:

- License and redistribution limits.
- Audio availability.
- Speaker diarization quality.
- Word-level or utterance-level timestamps.
- Turn boundary labels.
- Backchannel labels.
- Overlap and interruption labels.
- Language and domain.
- Audio quality and recording conditions.
- Suitability for training, validation, and testing.

If labels are incomplete, derive weak labels from timing:

- Short user utterance during assistant speech can be a candidate backchannel.
- User speech that causes assistant stop can be a candidate interruption.
- Long silence after syntactically complete speech can be a candidate end of turn.
- Short silence followed by continuation can be a candidate pause.

Weak labels should be treated as bootstrapping data, not final evaluation data.

Task-oriented dialogue is not required for the first phase. The initial research target is turn-taking and orchestration, not assistant task competence. Task execution can be layered in later through the LLM once the streaming speech loop is stable.

One later tool-use pattern could be:

1. The LLM quickly emits a spoken bridge such as "Let me check that."
2. TTS speaks the bridge while the LLM or tool runtime performs work.
3. The assistant resumes with the result before the spoken bridge finishes or shortly after.

This is a capability-layer concern and should not block the turn-taking research.

## Measurement Plan

Build instrumentation before optimizing.

Required timestamps:

- User audio frame arrival.
- ASR partial token emitted.
- ASR stable token emitted.
- Turn-taking probability update.
- End-of-turn committed.
- LLM prefill started and completed.
- LLM first generated token.
- TTS request started.
- TTS first audio chunk ready.
- Audio playback started.
- User interruption detected.
- Assistant audio stopped.

Core metrics:

- Time from user turn completion to first assistant audio.
- Time from end-of-turn model commit to first assistant audio.
- LLM time to first generated token after prefill.
- TTS time to first audio chunk.
- Interruption stop latency.
- False assistant starts during user pauses.
- Missed end-of-turn events.
- Backchannel false interruption rate.

The project should include repeatable offline evaluation and a live latency dashboard as early as possible.

Latency evaluation should separate model-pipeline latency from network latency. During rented-GPU prototyping, the cascaded ASR, LLM, and TTS stack should run on one machine so internal component timing is measured without network transport. Client-to-server audio and playback networking can be measured separately, but it should not be included in the core model orchestration metric.

The intended prototype represents a future local or single-machine deployment where all models are colocated and can run sequentially or in parallel as needed. The rented GPU setup is a practical approximation of that target, not the final user-facing latency environment.

## Milestones

### Milestone 1: Baseline Streaming Loop

- Choose initial streaming ASR, small LLM, and streaming TTS candidates.
- Build a minimal full-duplex pipeline.
- Add timestamp instrumentation across all stages.
- Measure baseline latency and failure modes.
- Run the full cascaded stack on one rented GPU machine and report model-pipeline latency separately from network latency.

### Milestone 2: Rule-Based Turn-Taking Baseline

- Implement a simple silence and transcript-punctuation baseline.
- Measure false starts, missed turns, and interruption latency.
- Use this as the minimum bar for the learned adapter.

### Milestone 3: Dataset and Label Pipeline

- Select datasets for turn-taking and backchannel training.
- Build conversion into a local typed training format.
- Create train, validation, and test splits.
- Add weak-label generation where useful.

### Milestone 4: Frozen-ASR Adapter Prototype

- Expose selected ASR intermediate features.
- Train a small causal adapter.
- Compare against the rule-based baseline.
- Tune policy thresholds separately from model training.

### Milestone 5: Streaming Prefill Optimization

- Integrate stable ASR partials with LLM cache prefill.
- Measure time to first LLM token after turn commit.
- Handle ASR revisions conservatively.
- Decide whether tentative unstable prefill is worth the complexity.
- Compare cold prompt generation against prefetched system prompt, prefetched conversation context, and prefetched stable user prefix.

### Milestone 6: Interruption and Backchannel Handling

- Add full-duplex assistant playback cancellation.
- Train or tune interruption detection while assistant audio is active.
- Add user backchannel classification.
- Add optional assistant backchannel generation only after user backchannels are reliable.

### Milestone 7: End-to-End Tuning

- Tune latency thresholds against false-start rate.
- Evaluate on held-out conversations.
- Run live tests with scripted and natural interactions.
- Document hardware-specific performance.

## Initial Risks

- Public streaming ASR models may not expose intermediate layers cleanly.
- LLM runtimes may not support efficient speculative or revisable prefill.
- Streaming TTS may require more text context than the latency target allows.
- Turn-taking datasets may not label the exact categories needed.
- Very low latency may increase false starts unless the policy is carefully calibrated.
- Backchannels are socially and acoustically subtle, so they should not block the first working system.

## Near-Term Next Steps

1. Research candidate streaming ASR models and record their chunk rates, feature access, and licenses.
2. Research streaming TTS candidates and measure first-audio latency.
3. Research small LLM runtimes with KV cache reuse and streaming generation.
4. Identify datasets for turn boundaries, backchannels, overlap, and interruptions.
5. Define the first typed event interfaces for audio, ASR, turn-taking, LLM, and TTS streams.
6. Build the simplest measurable streaming baseline before training the adapter.
