# Synthetic Conversational Turn-Taking Dataset

## Status and decision

This document is the authoritative design for the synthetic turn-taking dataset revamp. It
supersedes the isolated-utterance design in `synthetic-completion-dataset.md` where the two
documents disagree. The existing completion generator remains useful as a TTS, trimming, and
audio-quality foundation, but its prompt and sample contracts are not the target corpus contract.

The product is a causal turn-taking adapter over a continuously streaming English ASR encoder. Its
deployed decision surface is intentionally small:

- the primary output is `p_user_floor_now`, the probability that the user currently owns or is
  actively claiming the conversational floor;
- secondary speculative outputs estimate whether a genuine user end of turn is approaching within
  configured horizons;
- the only additional runtime input is the current probability that the assistant is speaking.

The assistant waveform is neither a model input nor a synthetic dataset artifact. Synthetic
assistant text exists only to make conversations coherent and to estimate realistic assistant-turn
durations. All audible speech in this corpus is user speech.

The first pilot contains 5-20 representative English conversations. It is a design and listening
gate, not an attempt to generate a statistically complete corpus.

The August 2026 clone-consistency pilot rejected Qwen Base ICL cloning for dataset generation. It
kept speaker identity stable but made every reviewed conversation substantially more robotic than
direct Qwen synthesis. Naturalness is the binding requirement; no scaled clone run is approved.

## Product objective

The adapter serves two related runtime decisions from the same causal stream:

1. When the assistant is inactive, commit the user's end of turn quickly enough to begin the
   response without an awkward wait.
2. When the assistant is active, yield promptly to a genuine user floor claim but continue through
   a backchannel or other non-floor feedback.

Speculative response generation may begin before commitment, but it is an optimization rather than
the primary objective. An early speculative start does not compensate for a late committed
response.

At frame time `t`, the model may use only:

```text
streaming ASR encoder state through t
+ assistant-speaking probability through t
+ causal adapter state through t
```

It must not use future audio, a future assistant state, an offline transcript, or labels derived
from ASR output.

### Primary output

`p_user_floor_now(t)` has the following intended behavior:

| Situation | Target behavior |
| --- | --- |
| User speaks while owning the floor | High |
| User pauses and will continue the same turn | High through the pause |
| User genuinely completes a turn | Transitions from high to low |
| User backchannels or reacts while the assistant retains the floor | Low despite user audio |
| User responds after the assistant yields | Transitions from low to high |
| User interrupts while the assistant is active | Transitions from low to high promptly |

The synthetic compiler provides semantic targets, normally zero or one. Soft ramps or masked frames
are allowed only around a genuinely uncertain transition boundary. The trained output becomes a
calibrated probability across varied examples.

The current training code names the inverse concept `yield_probability`. The revamp should make the
orientation explicit at one compatibility boundary rather than using both meanings throughout the
new pipeline.

### Secondary speculative outputs

Separate heads estimate a genuine EOT within future horizons, initially 500 and 1,000 ms. These
heads may start cancellable LLM generation. They cannot commit a response or authorize playback by
themselves.

The existing future-user-activity bins may be retained as an ablation or auxiliary task, but they do
not replace explicit speculative EOT targets. Auxiliary event heads may predict completion, HOLD,
non-floor feedback, and floor take for training and diagnosis. They are not required in the product
interface.

## Semantic conditions

Build the corpus around five conditions rather than a large collection of loosely related event
classes:

1. **User completion:** user floor changes from high to low at genuine EOT.
2. **User HOLD:** the user pauses and continues; user floor remains high.
3. **Non-floor user speech:** one lexical backchannel occurs while user floor remains low.
4. **Response after assistant yield:** user floor changes from low to high after the assistant stops.
5. **User interruption:** user floor changes from low to high while assistant-speaking probability
   is still high, and the assistant then yields.

There is no failed-interruption class. A deliberate user interruption is a floor claim to which the
assistant should submit. Ambiguous micro-utterances belong to the non-floor or interruption
condition according to their planned conversational intent; using the same surface forms in both
conditions prevents a lexical shortcut.

The dataset also needs ordinary negative context: assistant-active spans with no user event,
user-active spans with no imminent completion, and quiet transition-free spans. These are explicit
sampling strata, not additional semantic conditions, and they must be materialized even when an
event-focused crop would be easier to obtain.

## Conversation plan

The LLM generates coherent source conversations rather than isolated 20-second examples. Plans are
English-only and contain enough events for several 20-second views. A plan may contain multiple user
turns, virtual assistant replies, backchannels, HOLDs, and interruptions. The source is longer than
the training view so several dense, low-padding crops can be drawn from one consistent speaker and
topic without forcing an event into every view.

Each plan has four typed components:

### Conversation content

- topic domain and concrete topic;
- speech acts and relationship between successive turns;
- virtual assistant reply text for coherence and duration estimation;
- ordered interaction condition for every user unit;
- explicit distinction between floor-owning responses, micro-backchannels, and interruptions.

Domains, speech acts, and interaction conditions are selected through deterministic stratified
sampling. The LLM realizes a selected plan; it does not freely choose the dataset distribution.
Topic quotas, similarity rejection, and vocabulary checks prevent narrow clusters such as the
farmer-heavy prompts observed in the first review batch.

### Conversation speaker

A conversation requests one coherent user identity: English accent or dialect, approximate age,
pitch, vocal weight, and baseline conversational manner. Clean close-mic speech is invariant.
Speaker consistency is desirable but must not be purchased with robotic prosody.

The next Qwen listening pilot compares direct VoiceDesign with a high-quality preset/custom voice
route that supports per-unit delivery instructions. Reuse the same identity description or preset
within a conversation, batch compatible units, and measure identity drift. Reject a route if either
naturalness or intelligibility is poor. The rejected Base ICL clone route remains available only as
an experimental benchmark; it is not a candidate for scaled generation.

### User units

Each audible user unit has text, semantic condition, and a bounded delivery change relative to the
base voice. Pace, energy, and affect can vary by unit to reflect conversational context, for example
curious to frustrated or calm to urgent, without changing speaker identity. The corpus deliberately
covers slow, moderate, fast, engaged, restrained, playful, mysterious, and other natural delivery
styles.

User units deliberately cover several duration bands:

- **micro-backchannels:** one lexical acknowledgment only, normally `yeah`, `yep`, `right`,
  `okay`, `sure`, `mhm`, `uh-huh`, or `mm-hmm`; no clause, evaluation, or proposition is allowed;
- **brief turns:** roughly 2-8 spoken words;
- **normal turns:** roughly 12-28 spoken words;
- **extended turns:** roughly 45-90 spoken words, targeting 20-30 seconds of actual user speech.

A reaction that communicates a proposition such as "that sounds about right" is not a
backchannel. It must be a floor-owning response or another explicitly planned turn. Collaborative
completions are diagnosed separately and are not included in the first clone-consistency pilot.
Duration quotas apply across a source conversation so it does not collapse into uniformly short
question-answer pairs.

### Virtual assistant units

Virtual assistant units contain text, sampled speaking rate, response latency, pause profile, and
the interaction they enable. They are never passed to TTS. Their word count and delivery parameters
produce an estimated duration, which is converted into an assistant-speaking probability curve
after the user renders have known durations.

## Generation and post-TTS timing

Semantic relationships are planned before TTS, but exact timestamps are not. The materialization
sequence is:

1. Generate and validate a typed conversation plan.
2. Select a quality-gated direct-synthesis voice and delivery instruction for each user unit while
   retaining the conversation's requested identity.
3. Send every user unit as one uninterrupted TTS request, batching compatible units where useful.
4. Measure each rendered unit's actual active speech and silence regions.
5. Remove terminal synthesis silence and reject noisy, empty, truncated, or artifact-heavy output.
6. Estimate virtual assistant durations and response latencies.
7. Place the measured user units and virtual assistant spans into a post-TTS scenario timeline.
8. Construct the assistant-speaking input curve and semantic user-floor targets.
9. Rasterize the scenario and materialize 20-second training views at 80 ms per frame.

No symmetric pre-TTS timestamp plan is treated as ground truth. No ASR or word alignment is needed
to recover event timing. Audio energy locates the actual start, end, and internal silence of a
render. A planned HOLD is only a content and delivery request: the compiler emits HOLD supervision
only when the rendered unit actually contains an internal silence of at least 500 ms followed by
resumed speech. It emits normal completion supervision when no such pause occurs. The pipeline must
never split a HOLD into clauses or insert zero-valued silence manually.

An internal silent interval lasting at least 500 ms is a HOLD when active speech from the same
planned user turn resumes afterward. Genuine EOT is the final active-speech offset of a normal user
turn after terminal generated silence is removed. A backchannel's acoustic offset is not an EOT
because its planned semantic user-floor state remains low.

Ordinary source waveform duration is the final resolved user or virtual-assistant event plus one
second of trailing context. Estimated prompt duration is not padding. A fixed longer duration is
legal only for an explicitly typed control source, such as an assistant-only or quiet negative
window.

## Assistant-speaking probability

Assistant activity is a frame-aligned scalar input, not an audible track and not a prediction
target. It should approximate the runtime signal produced by assistant playback control while
remaining imperfect enough to prevent a shortcut.

Curves may contain:

- active plateaus sampled below and up to one;
- near-zero inactive plateaus;
- shallow pause or hesitation dips;
- finite rise and fall ramps;
- small perturbations, delayed changes, and brief dropouts.

For a backchannel, assistant probability normally stays high or dips slightly and recovers. For an
interruption, it is high at user onset and falls after a sampled assistant-yield delay. For a normal
response, it falls before user onset and is followed by a sampled response latency.

Assistant duration and contour parameters may be resampled when a conversation is rematerialized,
provided event ordering and semantics remain valid. Later user units must be repositioned with the
changed virtual duration, and every frame label must be regenerated from the resulting timeline.
Future assistant-state changes must never be exposed to the model before they occur.

## Twenty-second training views

Every materialized view is exactly 20 seconds: 250 frames at 80 ms. The current four-second causal
burn-in may remain, leaving 16 supervised seconds, but pilot reports must count only events in the
supervised region.

A conversation can yield different views across epochs without duplicating source audio. Use a
deterministic seed derived from conversation identity, epoch, and view index to vary:

- crop start within the post-TTS scenario;
- virtual assistant speaking rate and duration within plan bounds;
- response latency and assistant-yield latency;
- assistant probability contour details;
- approved acoustic augmentation.

The same seed must reproduce the waveform recipe, probability input, labels, and crop exactly.
Conversation and speaker identities remain in one split across all materializations.

Event-focused sampled views should contain one or more supervised events, and a view may contain several
EOTs, HOLDs, backchannels, or floor acquisitions. Do not reduce a view to a single selected boundary
or discard additional labels. A separate quota sampler must also produce control views, including:

- assistant probability high with no user speech;
- continuous user speech with no imminent EOT;
- ordinary silence or transition-free context;
- a single event surrounded by longer context.

The default source must be long enough that ordinary views use zero source padding. Padding is a
last-resort boundary behavior, not a sampling strategy. Pilot validation reports the fraction of
left- or right-padded crops and rejects a batch when it exceeds five percent. At least ten percent
of views must be assistant-only controls and at least ten percent must be user-only controls; at
least another ten percent must be event-light. A view can satisfy more than one control stratum, but
the report must show each count independently.

Crop selection must not remove audio needed by a speculative horizon label. Frames whose required
future interval extends beyond the source scenario or crop contract are masked.

## Label construction

The primary label is dense across all valid supervised frames:

```text
p_user_floor_now = 1  user owns or actively claims the floor
p_user_floor_now = 0  user does not own or claim the floor
```

The five conditions compile as follows:

| Condition | User audio | Assistant probability | Primary label |
| --- | --- | --- | --- |
| Completion | Active, then ends | Normally low before EOT | High, then low at EOT |
| HOLD | Active, silence, active | Normally low | High throughout |
| Non-floor feedback | Short active unit | High or shallow dip | Low throughout |
| Response after yield | Begins after latency | Falls before onset | Low, then high at user floor claim |
| Interruption | Begins during assistant activity | High at onset, then falls | Low, then high at user floor claim |

Auxiliary labels are derived from the same semantic plan and measured render boundaries:

- genuine turn completion;
- continuation/HOLD;
- non-floor feedback;
- floor take;
- EOT within 500 and 1,000 ms.

The primary and auxiliary targets are not inferred from ASR, transcripts, VAD classes, or the
assistant probability curve. In particular, the assistant probability is context: changing it does
not silently relabel a planned backchannel as an interruption or vice versa.

## Sampling and split policy

Balance unique semantic events before generating repeated crops. A useful initial scaling mixture
after the listening pilot is:

- ordinary exchanges and completions;
- HOLD-heavy exchanges;
- assistant-active backchannels and reactions;
- interruptions;
- difficult combinations containing several event types;
- an explicit event-light allocation.

Exact percentages should be selected after counting valid pilot events rather than forcing every
small pilot to match a nominal distribution. Balance topics, voices, delivery styles, unit lengths,
assistant durations, and event positions independently enough to avoid shortcuts.

Split by conversation plan, prompt family, topic lineage, and voice identity. All renders, crops,
timing variations, and augmentations descended from one conversation belong to the same split.

## Representative pilot

Generate 10-20 source conversations, preferably 20 if the valid-render rate permits it. Each source
should normally yield 6-8 distinct 20-second views. The pilot should
cover every semantic condition, several event-light contexts, broad topics, varied voice identities,
and varied pace and affect. It need not exhaust the combination space or present polished production
statistics.

The review surface must let a reviewer:

- move between conversations directly;
- listen to the composed user-only scenario and selected 20-second view;
- slide the crop while preserving a valid materialization;
- inspect the user waveform and user speech activity;
- inspect assistant-speaking probability;
- inspect dense `p_user_floor_now` and speculative EOT targets;
- inspect HOLD, completion, non-floor feedback, and floor-take events;
- see source bounds, zero padding, frame indices, prompt text, voice instructions, seeds, and
  generation provenance.

The pilot passes only when the reviewer accepts speech naturalness, speaker consistency, topic and
style diversity, event timing, assistant-contour plausibility, and label semantics. Failed renders
remain visible in validation reports but do not enter training views.

## Evaluation

Evaluate the primary policy as a risk-constrained committed-latency problem.

### Risk

Count at least:

- premature commitment during a HOLD;
- commitment before genuine EOT;
- stopping assistant playback for a backchannel or non-floor reaction;
- failure to yield to a genuine user interruption;
- excessive interruption-yield latency.

Negative EOT latency is a premature commitment and therefore risk, not a latency improvement. Tune
the commit and interruption thresholds, hysteresis, and persistence on validation only. Compare
systems at matched risk budgets and report the latency-risk Pareto curve.

### Committed EOT latency

The main latency metric is:

```text
committed response timestamp - true acoustic EOT timestamp
```

Report p50, p90, p95, missed commits, and the fraction of commitments that violate the risk
definition. This metric determines whether the user waits after stopping.

### Speculative lead

The secondary metric is:

```text
true acoustic EOT timestamp - speculative generation start
```

Report it alongside candidate invalidation rate, wasted generated tokens or compute, and cases where
speculation begins early but commitment remains late. It must not dominate model selection.

### Evaluation stages

Synthetic evaluation is appropriate for pipeline debugging and the first controlled learning
experiment because the semantic labels are exact. Hold out complete synthetic conversations and
report results by condition, duration, voice, topic, delivery, event density, and assistant contour.

Synthetic accuracy is not evidence of natural-conversation readiness. After the pilot establishes
that the task is learnable, use the existing real conversation validation/test setup and external
causal turn-taking benchmarks to compare against the prior Voice-Light adapter and open-source
turn-taking systems. Production selection remains based on real held-out latency-risk behavior,
backchannel robustness, interruption response, calibration, and streaming causality.

## Non-goals and invariants

- English only.
- No assistant audio encoder or assistant waveform in this corpus.
- No ASR-, transcript-, or word-alignment-derived ground truth.
- No failed-interruption class.
- No minute-long uninterrupted synthetic stories; 60-120 second source conversations are allowed
  only when they contain varied turn lengths and useful interaction or control spans.
- No single-boundary reduction when a view contains multiple useful events.
- No split leakage across related plans, renders, crops, or voices.
- No synthetic validation result replaces the later real evaluation gate.
