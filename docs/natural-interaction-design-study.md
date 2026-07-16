# Voice Light Natural Interaction Design Study

Research and architecture decision record, 2026-07-16.

## Executive Decision

Voice Light should pursue a **hybrid full-duplex orchestration architecture**:

- Keep an optimized ASR -> Qwen -> streaming-TTS cascade as the semantic and speech-quality path.
- Add a continuous interaction-control plane that remains active while either participant speaks.
- Make playback duck, pause, resume, boundary-drain, and cancel independent from LLM generation.
- Bind speculative language, speech, and tool candidates to transcript revisions and invalidate them
  explicitly.
- Treat backchannels as ephemeral interaction acts, not shortened durable conversation turns.
- Run a parallel native-duplex research track, but do not make a native-duplex model the product
  controller until it beats the hybrid system on timing, rejection, state-update reasoning, and
  traceability gates.

For user speech during assistant playback, the initial policy should be:

> **Duck immediately, pause at the nearest safe word boundary, classify for at most 500 ms, resume
> after a backchannel, and yield plus invalidate stale speech after a genuine floor-taking act.**

This is preferable to immediate hard cancellation because cancellation is irreversible and turns
ordinary `mm-hm`, laughter, or brief agreement into a broken response. It is preferable to
continuing at normal volume because that creates competitive masking and can lose the first
semantic word of an interruption. It is preferable to an unbounded buffered pause because generated
content becomes stale quickly after correction, disagreement, topic change, or a revised request.

The proposed timings below are **initial controller constants to test**, not claims that research
has established one universal threshold:

| Deadline from user onset | Required behavior |
| --- | --- |
| 0-50 ms p95 | Begin a 20-30 ms playback gain ramp to approximately -18 dB. |
| 0-120 ms p95 | Pause at the next word boundary, or force-pause if no boundary is available. |
| 160 ms target | Produce the first acoustic interaction prediction. |
| 300-350 ms target | Combine acoustic evidence with stable lexical evidence and decide most ordinary cases. |
| 500 ms hard deadline | Resume, yield/cancel, or enter an explicit unresolved-overlap fallback. Never remain indefinitely ducked. |

The hard architectural distinction is:

- **Audible speech is reversible until played.**
- **Generated semantics are speculative until committed.**
- **Tool side effects are not reversible and therefore require a separate commitment rule.**

## Research Basis

Voice Activity Projection predicts the joint future voice activity of both participants rather than
reducing turn-taking to present silence. The standard VAP representation projects two seconds into
four horizons: 0-200, 200-600, 600-1200, and 1200-2000 ms. A real-time implementation retained
accuracy with about one second of transformer context and ran in real time on CPU
([VAP](https://www.isca-archive.org/interspeech_2022/ekstedt22_interspeech.html),
[continuous VAP](https://arxiv.org/abs/2401.04868)).

Fine-tuned VAP has also been used for continuous frame-wise backchannel timing and type prediction.
That work labels opportunities 500 ms before the observed backchannel, feeds the listener's own
backchannel audio back into the model to avoid unnatural consecutive responses, and finds that
assessment-type backchannels need more context than simple continuers
([Inoue et al., 2025](https://aclanthology.org/2025.naacl-long.367/)).

Moshi demonstrates that parallel user and assistant audio streams can support real-time,
unsegmented interaction, with time-aligned text improving linguistic quality
([Moshi](https://arxiv.org/abs/2410.00037)). It does not establish that the resulting policy is
naturally timed. Full-Duplex-Bench v1.5 found strong scenario-dependent tradeoffs: systems that yield
rapidly to real interruptions often falsely react to side speech, while systems that filter
non-floor-taking speech often react too slowly to genuine interruption
([Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159)).

HumDial-FDBench makes the distinction concrete with separate scenarios for follow-up questions,
negation or dissatisfaction, repetition requests, topic switches, stop requests, user
backchannels, user pauses, third-party speech, and speech directed to others. Its leaderboard shows
that both open cascades and native-duplex systems still have large gaps in interruption and
rejection behavior; Moshi is not naturally strong merely because it is architecturally full duplex
([HumDial-FDBench paper](https://arxiv.org/abs/2604.21406),
[benchmark repository](https://github.com/ASLP-lab/HumDial-FDBench)).

Recent alignment results reinforce the same point. Preference alignment from live spoken
interactions needed explicit timing-related examples for cutting users off and failing to answer in
time
([Wu et al., 2025](https://arxiv.org/abs/2506.21463)). Multi-faceted interaction alignment treats
pause handling, turn-taking, backchanneling, and user interruption as separate reward axes and adds
a semantic reward because timing-only optimization can degrade response meaning
([Ohashi et al., 2026](https://arxiv.org/abs/2606.11167)).

Two newer diagnostics are especially important for Voice Light's architecture decision:

- Native-duplex models can exhibit **state inertia**: while speaking, they may remain biased toward
  generation and miss the first semantic word of an abrupt interruption. Interruption reduced
  comprehension across Moshi, PersonaPlex, and another duplex model
  ([Zero-Buffer Benchmark](https://arxiv.org/abs/2606.11386)).
- Even when a system audibly yields, it can continue reasoning from stale state. EchoChain identifies
  contextual inertia, interruption amnesia, and objective displacement; no evaluated real-time
  system exceeded a 50% pass rate, and failures fell 40.2% when the same information was delivered
  without interruption
  ([EchoChain](https://arxiv.org/abs/2604.16456)).

Therefore, full duplex is a transport and modeling capability, not evidence of a correct interaction
policy.

## 1. Current Overlap Behavior Map

The current prototype is an interruptible half-duplex cascade with continuous microphone capture.
Its behavior is implemented primarily in:

- `app/compute/voice/session.py`
- `app/compute/voice/speech_detection.py`
- `app/compute/voice/qwen_worker.py`
- `app/local/web/pages/voice-agent/app.js`
- `app/local/web/pages/voice-agent/playback-worklet.js`

The current path is:

```text
continuous microphone PCM
  -> server Silero VAD
  -> Nemotron streaming partial ASR
  -> 250 ms Silero end decision + 500 ms additional silence
  -> ASR finalization
  -> append durable user turn
  -> start Qwen
  -> send complete words to Kyutai
  -> queue and play PCM in browser
```

When the server detects new user speech during an active assistant generation:

```text
VAD speech start
  -> mark generation cancelled
  -> mark already acknowledged assistant prefix as interrupted
  -> send assistant.cancel
  -> browser immediately clears the generation's queued PCM
  -> cancel Qwen task and Kyutai session
  -> wait for teardown before the next generation
```

### What the prototype already does well

- User audio continues to be captured during assistant playback.
- Cancellation is generation-ID-scoped, so stale audio is rejected.
- Played assistant words, rather than all generated words, enter durable history.
- Interrupted assistant history retains an ellipsis; unplayed text does not enter history.
- Qwen text and TTS audio are pipelined word by word.
- Server and client report sample-relative playback progress.
- Generation teardown is explicit and old/new Qwen and TTS work do not overlap accidentally.

### Current limitations

| Area | Current behavior |
| --- | --- |
| User backchannel | Any detected user speech hard-cancels assistant generation and playback. |
| Laughter | Treated identically to a correction or topic switch. |
| Ducking | None. |
| Playback pause/resume | None; only queue drain or queue clear. |
| Resumable audio | None. |
| Generation pause | None; generation completes or is cancelled. |
| Phrase-boundary stop | None; cancellation acts on the audio queue immediately. |
| Assistant backchannel | None. |
| Assistant interjection | None. |
| Cooperative overlap | Not represented. |
| Competitive overlap | Reduced to user VAD start. |
| Transcript revisions | Partial and final text exist, but no stable-prefix/volatile-suffix identity. |
| Turn prediction | Silence policy only in the live controller; the trained adapter is not integrated. |
| Background reasoning/tools | No independent structured action lane. |
| Causal trace | Generation IDs and samples exist, but prediction confidence, revision provenance, and causal parents do not. |

The current live endpoint delay is approximately 750 ms of configured acoustic silence before
finalization, plus scheduling and model work. The measured client path attributes about 966 ms to
endpointing and about 1.8 seconds from true user end to first assistant PCM. This makes endpoint and
floor prediction the first latency target; replacing Qwen3 1.7B with 0.6B saves only about 10 ms to
the first complete word, while VoXtream2 could save roughly 350 ms relative to isolated Kyutai.

### Work that can overlap

The latency stages are not strictly additive:

| Work | When it can run |
| --- | --- |
| Streaming ASR and turn prediction | Continuously while the user speaks and pauses. |
| System/history LLM prefill | Before the current user turn begins. |
| Stable-user-prefix prefill | While later volatile speech is still arriving. |
| Speculative response candidates | While the user speaks, bound to a transcript revision. |
| Qwen decoding and TTS input | Already overlap word by word in the prototype. |
| TTS generation and client playback | Overlap once the first playable audio arrives. |
| Interruption classification and generation teardown | Can run while VAD/ASR continues on the new user speech. |
| Read-only tools and semantic analysis | Can run independently of audible playback when their inputs are stable. |

Endpoint policy delay cannot be hidden by simply adding model concurrency. It must be reduced through
prediction or moved earlier through stable-prefix speculation. Likewise, an unprepared assistant
backchannel cannot meet a 200 ms budget merely because Qwen and TTS are individually fast.

## 2. Recommended Initial Conversational Personality

Adopt the proposed personality:

> **Cooperative listener:** backchannel when confident that the user retains the floor; interject at
> a projected opening only when a useful response is ready; competitively interrupt only for urgent
> repair or a critical misunderstanding.

This should be expressed as policy parameters, not only prose:

| Dimension | Initial setting |
| --- | --- |
| Backchannel intensity | Moderate-low. Prefer precision over recall. |
| Turn-claim aggressiveness | Low. |
| Competitive interruption | Disabled except urgent repair or safety-critical misunderstanding. |
| Response length | Usually one or two interruptible clauses. |
| Repair style | Brief acknowledgment of the correction, then corrected substance. |
| Resumption style | Continue without recapping unless the interruption changed meaning. |
| User control | Expose later as a conversational-style preference, not a hidden prompt change. |

RESPOND's small pilot found a preference for moderate backchanneling and low overtaking
aggressiveness, with sentence-final acknowledgments preferred over mid-sentence ones. The study is
underpowered, but its direction matches a safe product default
([RESPOND](https://arxiv.org/abs/2603.21682)).

### Evidence required to emit an assistant backchannel

All must hold:

1. `p_user_speech` for the next 600 ms is high enough that the user is expected to retain the floor.
2. `p_user_yield` is low and not rising sharply.
3. A predicted backchannel opportunity is present near a prosodic or semantic boundary.
4. The selected backchannel does not claim new factual understanding beyond available evidence.
5. No backchannel has occurred inside the cooldown.
6. No correction, distress, safety issue, explicit question, or addressee ambiguity is active.

Initial thresholds for evaluation:

- `p_backchannel_opportunity >= 0.82`
- `p_user_speech_0_600 >= 0.65`
- `p_user_yield <= 0.25`
- at least 1.2 seconds of user floor holding
- at least 2.5 seconds since the last verbal assistant backchannel

### Evidence required for a projected-opening interjection

All must hold:

1. Future activity predicts an opening in 0-600 ms rather than a mere acoustic dip.
2. Stable transcript text is semantically complete enough to support the proposed response.
3. The candidate response is already generated to a speakable first clause.
4. The response-conditioned utility is high: the candidate fits this opening better than waiting.
5. The response advances the active conversational goal and is not a recap.
6. No high revision risk exists in the candidate's transcript anchor.

Initial thresholds:

- `p_user_yield_0_600 >= 0.70`
- stable-prefix age at least 120 ms
- stable-prefix revision probability below 0.15
- prepared-response usefulness at least 0.75
- projected first-clause duration no more than 1.5 seconds

### Evidence required for competitive interruption

Competitive interruption is permitted only when:

- urgency is at least 0.90;
- the misunderstanding or risk is already clear from stable evidence;
- waiting until the next normal transition would materially increase harm or task cost; and
- the prepared repair is short, specific, and immediately actionable.

Examples include preventing a dangerous action, stopping an incorrect irreversible tool action, or
repairing a critical identity/location misunderstanding. Ordinary disagreement, enthusiasm, or
having a useful idea does not meet this bar.

## 3. User Speech During Assistant Playback

### The first 500 ms

```text
0 ms      user speech onset
0-50      duck assistant audio
0-120     pause at next word boundary; retain queued audio
80-160    acoustic classifier update
160-350   fuse stable lexical evidence, duration, VAP horizons, and addressee evidence
<=350     resume common backchannels; cancel clear interruptions
350-500   unresolved fallback with generation paused at a boundary
500       mandatory resume/yield decision
```

### Comparison of candidate strategies

| Strategy | Benefit | Failure mode | Decision |
| --- | --- | --- | --- |
| Immediate hard cancellation | Fastest unambiguous yield; simple. | High false-cancel cost for `mm-hm`, laughter, and short agreement; loses TTS/LLM state. | Use only for explicit stop/repair or high-confidence interruption. |
| Immediate duck then classify | Fast reversible response; user can be heard; preserves options. | Needs a bounded buffer and reliable deadline. | Primary policy. |
| Buffered pause, generation continues | Fast resume after backchannel. | Wastes GPU and can build stale speech. | Allow only within a bounded speculation budget. |
| Playback and generation pause | Minimizes stale work. | Resume can be slower; some runtimes cannot pause cleanly. | Fallback after the short classification window. |
| Continue to word/phrase boundary | Avoids clipped words. | Continued overlap can mask the user's first words. | Only at ducked gain and with a strict 120 ms word-boundary cap. |
| Continue speaking while classifying | Smooth for backchannels. | User may miss assistant content and assistant may miss user onset. | Do not continue at normal volume. |
| Cancel only after competitive evidence | Protects cooperative overlap. | Can yield too slowly to explicit stop or correction. | Combine with immediate duck and explicit fast paths. |

### Playback and generation budgets

Initial limits:

- Pause playback at the next known word boundary no more than 120 ms after user onset.
- Retain no more than 800 ms of resumable queued audio; target typical retained age below 500 ms.
- While overlap remains unresolved, allow Qwen to generate at most one clause or 16 additional
  tokens, whichever occurs first.
- Allow TTS to synthesize at most 500 ms of audio beyond the paused playback position.
- At 350 ms unresolved overlap, pause generation at its next text boundary.
- At 500 ms, either resume or cancel. Do not keep a response indefinitely parked.

These limits protect the backchannel case without allowing a long correction or topic switch to
create seconds of obsolete work.

### Phrase-boundary stop rules

- **Explicit stop, correction, safety repair, or high-confidence competitive interruption:** stop
  after the current audio render quantum. Do not wait for a phrase boundary.
- **Uncertain overlap:** duck immediately and pause at the next word boundary, with a 120 ms cap.
- **Assistant-initiated semantic cancellation while the user is silent:** drain to the next clause
  boundary if it is within 250 ms; otherwise stop at the next word boundary.
- **Cooperative overlap classified as a backchannel:** resume from the exact paused sample. Do not
  replay already heard words.
- **No trustworthy alignment boundary:** force-pause at the sample deadline and mark the final
  partial word as not durably heard.

### Semantic staleness

Generated material becomes stale when any of these occur:

1. A transcript revision diverges before the generation's anchor.
2. The user corrects, negates, or changes a requirement referenced by the candidate.
3. The user asks for repetition or says they did not hear/understand the response.
4. The user switches topic or introduces a new primary goal.
5. The user sustains a floor-taking attempt beyond the resumable window.
6. A tool result, safety rule, or external state invalidates the candidate.

Age alone does not make a candidate stale, but an 800 ms paused-buffer age is the absolute resume
cap for the initial experiment. A short laugh or `mm-hm` does not invalidate meaning. A short
`yeah` remains resumable only while it is lexically closed; `yeah, but...` is a floor-taking act.

### Case-specific decision matrix

| User event | Early evidence | Playback action | Generation action | Durable semantics |
| --- | --- | --- | --- | --- |
| `mm-hm`, `uh-huh` | Short, low lexical novelty, continuer prosody, user activity projected to end quickly | Duck, word-boundary pause, resume | Continue within speculation budget | Ephemeral user backchannel event only |
| Laughter | Non-lexical; short; aligned with assistant content | Duck and pause; resume if short and non-distressed | Continue briefly | Ephemeral affect event |
| Bare agreement | `yes`, `right`, `exactly` with no continuation | Duck/pause/resume | Continue briefly | Optional non-durable agreement evidence |
| Agreement plus clause | `yes, and...`, `right, but...` | Yield | Cancel stale speech | New user turn |
| Correction | `no`, `actually`, `that's not...`, named-value replacement | Stop immediately | Cancel and start repair candidate | New user repair turn |
| Disagreement | Negative stance or objection with content | Yield | Cancel; preserve only played prefix | New user turn |
| Topic switch | New entity/goal with low semantic link | Yield | Invalidate old candidate and related speculative tools | New user turn |
| Repetition request | `what?`, `say that again`, inaudibility/clarification cue | Stop immediately | Cancel continuation; prepare repair/repeat | Repair event plus new turn |
| Sustained interruption | Speech over 650 ms, more than three lexical tokens, or rising interruption probability | Yield by 500 ms even without final ASR | Cancel | New user turn |
| Side speech / speech to other | Addressee mismatch, off-axis/far-field cues | Remain ducked briefly, then resume | Preserve | Logged non-addressed speech, not a user turn |

## 4. Assistant Backchannels

### Recommended v1 implementation choice

Use **curated, presynthesized audio selected by a learned opportunity/type model**.

| Option | Timing | Semantic flexibility | Risk | Decision |
| --- | ---: | ---: | --- | --- |
| Curated presynthesized audio | Best; ready immediately | Low | Repetition and poor contextual fit | Use as audio bank. |
| Learned selector over cached audio | Best with better fit | Medium | Needs labeled type/prosody data | Recommended v1. |
| Main Qwen plus streaming TTS | Too slow if started at opportunity | High | Missed timing, overly long output | Full turns, not reactive backchannels. |
| Small dedicated interaction model | Good after training | Medium-high | New model/data surface | v2 after cached selector baseline. |
| Speculative generation while user speaks | Can prepare assessments or repairs | High | Stale/miscalibrated meaning | Research path, gated by stable transcript revision. |

A 200 ms onset budget is realistic only when both the opportunity and candidate audio are prepared.
The 2025 VAP backchannel work explicitly predicts 500 ms before observed onset, supporting
opportunity prediction rather than reaction after a final transcript.

### Backchannel rendering policy

- Duration: 120-450 ms; absolute maximum 600 ms.
- Relative loudness: begin 4-8 dB below the normal assistant voice.
- Prosody: short, falling or level continuer contour; no emphatic floor-claim onset.
- Cooldown: 2.5 seconds minimum and at least one new semantic/prosodic unit.
- Consecutive limiter: no second verbal backchannel until the controller has ingested the first as
  assistant activity.
- Initial bank: several speaker-specific recordings of `mm-hm`, `right`, `yeah`, and a neutral
  laugh/breath response. Avoid `I understand` until semantic confidence is explicitly modeled.
- Selection: condition on opportunity type, recent backchannel history, user affect, and whether
  the candidate is a continuer or assessment.

### Durable history

An assistant backchannel must be stored as an `InteractionAct`, not appended to the ordinary
assistant message history. It may influence immediate social context, but it must not:

- satisfy a pending user request;
- count as an assistant answer;
- become a tool-call commitment;
- create a new turn boundary by itself; or
- be summarized later as a factual assistant statement.

### Recovery when the user unexpectedly stops

If the user stops immediately after an assistant backchannel:

1. Keep `USER_HOLDS` for a 250-450 ms recovery window.
2. Do not emit a second filler.
3. If yield probability and semantic completeness become high, transition to `OPEN_TRANSITION`.
4. Use a prepared full-turn candidate if it is bound to the current stable transcript revision.
5. If the backchannel was interpreted as a request to continue, a brief repair such as `Go on` may
   be used only when user behavior indicates confusion; do not use it by default.

## 5. Assistant Interjection Policy

Assistant speech while the user holds the floor has three distinct forms:

| Form | Floor intent | Typical content | Default permission |
| --- | --- | --- | --- |
| Backchannel | User retains floor | `mm-hm`, `right` | Allowed with high opportunity confidence |
| Cooperative interjection | Claims a projected opening | Short useful contribution or question | Allowed conservatively when prepared |
| Competitive interruption | Seizes floor | Urgent repair or warning | Normally prohibited |

The controller should use response-conditioned turn taking: deciding when to speak depends partly
on what useful response is ready, not only on an end-of-turn probability. Response-conditioned
TurnGPT showed that conditioning on a candidate response improves selection of ambiguous transition
points and can act as an incremental response ranker
([Jiang et al., 2023](https://aclanthology.org/2023.findings-acl.776/)).

### Interjection evidence vector

The decision should include:

- future user activity at 0-200, 200-500, 500-1000, and 1000-1500 ms;
- acoustic pause-versus-yield confidence;
- semantic completeness of stable text;
- transcript revision risk;
- whether a transition-relevance place is projected;
- candidate readiness, usefulness, and first-clause duration;
- active conversation goal and whether the candidate advances it;
- urgency and reversibility of waiting;
- recent failed interjections and user preference;
- assistant playback and tool/action state.

No single feature grants the floor. In particular, a pause plus a ready answer is insufficient when
future user activity remains high.

## 6. Orthogonal State Models

The controller should publish state changes independently and derive policy from their product.
There is no single `conversation_state` enum.

### User activity

| From | Trigger | To | Deadline / invariant |
| --- | --- | --- | --- |
| `SILENT` | Speech onset detector fires | `SPEAKING` | Timestamp at the first attributed input sample. |
| `SPEAKING` | Acoustic inactivity begins | `PAUSE_CANDIDATE` | Do not imply yield. |
| `PAUSE_CANDIDATE` | Speech resumes | `SPEAKING` | Preserve same revision lineage if stable prefix extends. |
| `PAUSE_CANDIDATE` | Yield policy commits | `SILENT` | Requires prediction evidence, not silence alone. |
| `SPEAKING` | Stream discontinuity/session end | `SILENT` | Emit explicit discontinuity cause. |

Invariant: `PAUSE_CANDIDATE` is an observation state, never a floor assignment.

### Playback

| From | Trigger | To | Rule |
| --- | --- | --- | --- |
| `IDLE` | Playable audio accepted | `QUEUED` | Candidate and revision must be valid. |
| `QUEUED` | First sample rendered | `SPEAKING` | Emit output sample position. |
| `SPEAKING` | User onset or attenuation command | `DUCKING` | Begin gain ramp within 50 ms p95. |
| `DUCKING` | Safe pause boundary reached | `PAUSED_BUFFERED` | Within 120 ms p95. |
| `DUCKING` | Backchannel classified before pause | `SPEAKING` | Unduck smoothly; no replay. |
| `PAUSED_BUFFERED` | Resume authorized | `RESUMING` | Buffer age and revision must remain valid. |
| `RESUMING` | Gain reaches normal and samples render | `SPEAKING` | First resumed sample is traceable. |
| `SPEAKING` | Non-overlap semantic stop at boundary | `DRAINING_TO_BOUNDARY` | Boundary no more than 250 ms away. |
| `DRAINING_TO_BOUNDARY` | Boundary reached | `COMPLETED` or `CANCELLED` | Depends on semantic completion. |
| Any active state | Immediate cancel | `CANCELLED` | No later sample for that playback ID may render. |
| `SPEAKING` | End marker and queue drains | `COMPLETED` | Full played response may commit. |
| `COMPLETED` or `CANCELLED` | Cleanup acknowledged | `IDLE` | Terminal state is retained in trace. |

Invariants:

- `PAUSED_BUFFERED` never advances the played sample counter.
- A `CANCELLED` playback ID is permanently ineligible for future audio.
- Resume requires the same semantic candidate and a non-superseded transcript anchor.

### Floor relation

| From | Evidence | To |
| --- | --- | --- |
| `USER_HOLDS` | Projected yield/opening | `OPEN_TRANSITION` |
| `OPEN_TRANSITION` | Assistant begins an authorized full turn/interjection | `AGENT_HOLDS` |
| `OPEN_TRANSITION` | User resumes | `USER_HOLDS` |
| `AGENT_HOLDS` | User speech onset | `OVERLAP_UNRESOLVED` |
| `OVERLAP_UNRESOLVED` | User backchannel/non-addressed speech | `AGENT_HOLDS` |
| `OVERLAP_UNRESOLVED` | User floor-taking act | `USER_HOLDS` |
| `OVERLAP_UNRESOLVED` | Both yield | `OPEN_TRANSITION` |
| `USER_HOLDS` | Assistant backchannel | `USER_HOLDS` |
| `USER_HOLDS` | Authorized urgent competitive interruption | `OVERLAP_UNRESOLVED`, then `AGENT_HOLDS` only after evidence |

Invariant: emitting a backchannel never changes floor ownership.

### Generation

| From | Trigger | To | Rule |
| --- | --- | --- | --- |
| `PREFILLING` | Required context/cache prepared | `READY` | Bound to a transcript revision. |
| `READY` | Tokens requested | `STREAMING` | Candidate ID becomes immutable. |
| `STREAMING` | Policy pause at safe text boundary | `PAUSED_AT_BOUNDARY` | Preserve resumable cache if runtime supports it. |
| `PAUSED_AT_BOUNDARY` | Resume and anchor still valid | `STREAMING` | No duplicate tokens. |
| `PREFILLING`, `READY`, `STREAMING`, `PAUSED_AT_BOUNDARY` | Transcript or goal supersedes anchor | `INVALIDATED` | Candidate may finish background computation but cannot become audible. |
| Active state | Resource/user cancellation | `CANCELLED` | Terminal. |
| `STREAMING` | Candidate accepted for durable response | `COMMITTED` | Commitment is semantic, not proof audio was played. |

Invariants:

- `INVALIDATED` and `CANCELLED` candidates cannot return to a playable state.
- A committed semantic candidate still requires playback acknowledgments before its text becomes
  durable conversation history.
- Generation can continue while playback is paused only inside the explicit speculation budget.

## 7. Typed Protocol Proposal

The protocol should use frozen Pydantic models and discriminated unions, matching Voice Light's
existing type conventions.

```python
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from app.shared.base_model import FrozenBaseModel


class CausalSource(StrEnum):
    ENERGY_VAD = "energy_vad"
    SILERO_VAD = "silero_vad"
    NEMOTRON_ASR = "nemotron_asr"
    TURN_ADAPTER = "turn_adapter"
    LEXICAL_CLASSIFIER = "lexical_classifier"
    RESPONSE_RANKER = "response_ranker"
    FLOOR_POLICY = "floor_policy"
    PLAYBACK_ENGINE = "playback_engine"
    USER_COMMAND = "user_command"


class TraceStamp(FrozenBaseModel):
    event_id: str
    parent_event_ids: tuple[str, ...]
    monotonic_time_ns: int = Field(ge=0)
    input_sample_position: int = Field(ge=0)
    output_sample_position: int | None = Field(default=None, ge=0)
    transcript_revision_id: int | None = Field(default=None, ge=0)
    source: CausalSource
    model_name: str | None
    model_revision: str | None


class ActivityHorizon(FrozenBaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    probability: float = Field(ge=0.0, le=1.0)


class PlaybackState(StrEnum):
    IDLE = "idle"
    QUEUED = "queued"
    SPEAKING = "speaking"
    DUCKING = "ducking"
    PAUSED_BUFFERED = "paused_buffered"
    RESUMING = "resuming"
    DRAINING_TO_BOUNDARY = "draining_to_boundary"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class InteractionPrediction(FrozenBaseModel):
    type: Literal["interaction.prediction"] = "interaction.prediction"
    stamp: TraceStamp
    p_user_speech: float = Field(ge=0.0, le=1.0)
    p_user_yield: float = Field(ge=0.0, le=1.0)
    p_user_backchannel: float = Field(ge=0.0, le=1.0)
    p_user_interruption: float = Field(ge=0.0, le=1.0)
    future_user_activity_horizons: tuple[ActivityHorizon, ...]
    assistant_playback_state: PlaybackState
    confidence: float = Field(ge=0.0, le=1.0)
```

```python
class FloorActionKind(StrEnum):
    HOLD = "hold"
    DUCK = "duck"
    PAUSE = "pause"
    RESUME = "resume"
    CONTINUE = "continue"
    YIELD = "yield"
    TAKE_FLOOR = "take_floor"
    EMIT_BACKCHANNEL = "emit_backchannel"
    CANCEL = "cancel"


class FloorActionReason(StrEnum):
    USER_ONSET_REFLEX = "user_onset_reflex"
    USER_BACKCHANNEL = "user_backchannel"
    USER_INTERRUPTION = "user_interruption"
    USER_CORRECTION = "user_correction"
    USER_STOP_REQUEST = "user_stop_request"
    USER_TOPIC_SWITCH = "user_topic_switch"
    NON_ADDRESSED_SPEECH = "non_addressed_speech"
    PROJECTED_OPENING = "projected_opening"
    URGENT_REPAIR = "urgent_repair"
    TRANSCRIPT_INVALIDATION = "transcript_invalidation"


class FloorAction(FrozenBaseModel):
    type: Literal["floor.action"] = "floor.action"
    stamp: TraceStamp
    action: FloorActionKind
    reason: FloorActionReason
    target_playback_id: str | None
    target_generation_id: str | None
    execute_by_monotonic_time_ns: int = Field(ge=0)
    execute_by_output_sample: int | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
```

```python
class SpeechActKind(StrEnum):
    FULL_TURN = "full_turn"
    INTERJECTION = "interjection"
    BACKCHANNEL = "backchannel"
    REPAIR = "repair"
    BRIDGE = "bridge"


class Durability(StrEnum):
    EPHEMERAL = "ephemeral"
    DURABLE_AFTER_PLAYBACK = "durable_after_playback"


class SpeechAct(FrozenBaseModel):
    type: Literal["speech.act"] = "speech.act"
    stamp: TraceStamp
    speech_act_id: str
    kind: SpeechActKind
    durability: Durability
    candidate_id: str | None
    text: str | None
    expected_duration_ms: int | None = Field(default=None, gt=0)
    utility: float = Field(ge=0.0, le=1.0)
    urgency: float = Field(ge=0.0, le=1.0)
```

```python
class TranscriptRevision(FrozenBaseModel):
    type: Literal["transcript.revision"] = "transcript.revision"
    stamp: TraceStamp
    revision_id: int = Field(ge=0)
    supersedes_revision_id: int | None = Field(default=None, ge=0)
    stable_prefix: str
    volatile_suffix: str
    audio_sample_position: int = Field(ge=0)
    stable_prefix_end_sample: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
```

The four required events are not enough for complete orchestration. Add:

- `GenerationCandidateCreated`, `GenerationCandidateInvalidated`, and
  `GenerationCandidateCommitted`;
- `PlaybackCheckpoint` with played, queued, and boundary sample positions;
- `ToolActionPlanned`, `ToolActionDispatched`, `ToolActionCompleted`, and `ToolActionObsolete`;
- `InteractionAnnotation` for evaluation labels and adjudication.

Every policy event must carry a `TraceStamp`. Derived actions must list the prediction and transcript
events that caused them in `parent_event_ids`.

## 8. Background Reasoning and Tool Work

Audible speech must not be the scheduler for semantic reasoning or tools.

Use three independent lanes:

```text
interaction lane:  VAD / VAP / overlap / floor actions
speech lane:       response candidates / TTS / playback
action lane:       planning / tools / external state
```

### Transcript-bound speculation

Each candidate is anchored to:

```text
conversation version
stable transcript prefix hash
transcript revision ID
input audio sample position
active goal version
```

On a new revision:

- If the stable prefix only extends, reuse prefill state and re-rank candidates.
- If text diverges after the anchor, invalidate candidate speech but retain reusable tool results
  only when their inputs remain semantically valid.
- If text diverges before the anchor, invalidate the candidate and all dependent uncommitted tool
  plans.
- Never execute irreversible side effects from a volatile suffix.

Incremental generation can reduce gaps, but a COLING 2025 system that generated candidates from
partial ASR reduced measured turn gaps without improving user ratings; incomplete user input caused
misaligned responses
([Chiba and Higashinaka, 2025](https://aclanthology.org/2025.coling-main.249/)). Voice Light should
therefore optimize **speculative-candidate hit rate at preserved semantic quality**, not speculative
latency alone.

### Tool action state

Recommended action lifecycle:

```text
PLANNED -> ELIGIBLE -> DISPATCHED -> RUNNING -> RESULT_READY -> COMMITTED
                   \-> CANCELLED
RUNNING/RESULT_READY -> OBSOLETE
```

- Read-only lookups may dispatch from a stable prefix when their arguments are complete.
- Reversible local preparation may dispatch speculatively.
- External writes, purchases, messages, navigation changes, or device actions require explicit
  semantic commitment.
- An obsolete result can still be cached, but cannot alter the active response without revalidation.

DuplexSLA demonstrates the value of a synchronized action channel that can emit planning and
structured tool calls without halting audio. Its results are a promising native-duplex research
direction, not yet a reason to couple Voice Light's product tools to the speech decoder
([DuplexSLA](https://arxiv.org/abs/2605.20755)).

## 9. Prompting

### Current prompt

The current prompt asks for a natural, direct answer to the latest message, no repeated answers,
substantive opening content, no generic filler, and plain text. It does not explicitly optimize:

- spoken clause length;
- interruption recovery;
- repair language;
- response utility at a projected opening;
- speech-act-specific content; or
- distinction between ephemeral and durable output.

The repository benchmark shows:

- Qwen3 1.7B current prompt: median 50 words.
- Cooperative-listener prompt: median 26 words.
- First complete word remains approximately 80-100 ms.

The prompt shortened speech without materially improving first-word latency, confirming that prompt
work is a lexical/naturalness lever rather than a timing controller.

### Prompt variants to compare

| Prompt | Intended effect | Risk |
| --- | --- | --- |
| Current | Direct general response | Turns remain too long for spoken interruption. |
| Concise conversational | One or two short sentences; no recap | Can become terse or omit nuance. |
| Cooperative listener | One or two clauses; specific reaction plus one useful move | Best current default; needs live calibration. |
| Explicit speech-act/control | Supplies `SpeechAct`, word budget, repair context, and candidate purpose | Must not leak control timing into free-form model judgment. |

Recommended generation interface:

```text
speech_act = full_turn | interjection | repair | bridge
spoken_word_budget = integer
must_preserve = tuple of established facts/constraints
must_update = tuple of revised facts/constraints
opening_context = ordinary_turn | projected_opening | post_interruption
```

The LLM should produce lexical content for an already selected act. It may score whether a prepared
response is useful enough to schedule, but the floor controller remains authoritative.

### Prompting can improve

- spoken brevity;
- short interruptible clauses;
- reduced recap;
- natural contractions;
- repair wording;
- whether a candidate is worth taking a projected opening;
- bridge language during long tool work; and
- semantic preservation after interruption.

### Prompting cannot solve

- acoustic pause versus yield;
- classification in the first 200 ms of overlap;
- user addressee detection from spatial/acoustic cues;
- duck, pause, resume, or cancellation;
- a sub-200 ms backchannel that starts generating after the opportunity;
- simultaneous listen/speak perception; or
- sample-accurate traceability.

## 10. Orchestration Versus Training

| Capability | Prompting | Orchestration | Training |
| --- | ---: | ---: | ---: |
| Short spoken clauses | Primary | Enforce word/time cap | Helpful |
| Duck/pause/resume | No | Primary | No |
| Backchannel durability | No | Primary | No |
| Stable-prefix prefill | No | Primary | No |
| Transcript-bound speculation | No | Primary | Candidate scoring can be trained |
| Phrase-boundary stop | No | Primary | TTS boundary quality can be trained |
| Pause versus turn end | No | Thresholds/hysteresis | Primary predictor |
| Backchannel versus interruption | No | Fast rules and fusion | Primary predictor |
| Cooperative versus competitive overlap | Limited lexical clue | Policy | Primary predictor |
| Assistant backchannel timing | No | Cooldown/eligibility | Primary opportunity model |
| Assistant take-floor decision | Candidate utility prompt | Policy constraints | Response-conditioned model |
| Overlapping prosody | Limited content guidance | Gain/routing | Primary |
| Addressee / third-party speech | No | Routing and speaker identity | Primary acoustic/multimodal model |
| Stale-result handling | No | Primary | Semantic equivalence model may help |
| Tool/action commitment | No | Primary | Planner quality may be trained |

### Sufficiency of the current frozen-Nemotron adapter

The adapter currently outputs:

- a dense user yield probability;
- five event classes: turn completion, continuation pause, backchannel, interruption, other;
- future user activity over four horizons; and
- recurrent state, conditioned on whether the assistant is speaking.

This is sufficient for an **initial learned controller experiment** when fused with:

- current speech activity from VAD;
- stable partial ASR text;
- a small lexical speech-act classifier;
- explicit stop/correction rules; and
- playback sample/boundary state.

It is not sufficient for the final controller. Missing targets include:

- assistant backchannel opportunity and type;
- assessment versus continuer backchannel;
- cooperative versus competitive overlap;
- floor-claim attempt versus successful floor claim;
- addressee and third-party speech;
- laughter and affect subtype;
- explicit correction, disagreement, repetition request, and topic switch;
- semantic completeness and transcript revision risk;
- response-conditioned take-floor utility;
- urgency and conversation-goal relevance;
- calibrated uncertainty under assistant echo/noise; and
- appropriate overlapping prosody.

Do not expand one head into a catch-all state. Add focused heads or a separate semantic interaction
model, then fuse their outputs in policy.

## 11. Ordered Implementation and Research Plan

### Phase 0: Instrumentation and replay

1. Add monotonic and input/output sample stamps to all voice events.
2. Add transcript revision IDs with stable prefix and volatile suffix.
3. Add a deterministic duplex replay harness with recorded user and assistant tracks.
4. Record playback gain, queue age, word/phrase boundaries, and generation validity.
5. Add policy event logs with causal parents and confidences.

Gate: at least 99.9% of actions have complete causal stamps; replay reproduces action choices and
sample positions within 10 ms.

### Phase 1: Reversible playback control

1. Implement client worklet gain ramps.
2. Add pause/resume without clearing queued audio.
3. Add word-boundary and forced-sample pause.
4. Retain the existing hard-cancel path.
5. Measure duck, pause, stop, and resume latency before adding learned policy.

Gate: user-onset-to-duck p95 <= 50 ms; user-onset-to-pause p95 <= 120 ms; no stale sample after
cancel.

### Phase 2: Rule-based duck-then-classify baseline

1. Use duration, VAD confidence, partial lexicon, and explicit stop/correction phrases.
2. Add the 350 ms normal decision target and 500 ms hard deadline.
3. Implement bounded LLM/TTS speculation.
4. Separate ephemeral backchannel events from durable user turns.

Gate: backchannel false-cancel rate below 5%, stale resume below 1%, and genuine interruption stop
p95 below 220 ms.

### Phase 3: Integrate the Nemotron adapter

1. Stream encoder taps and recurrent adapter state.
2. Calibrate yield, event, and activity horizons under assistant playback.
3. Fuse acoustic predictions with lexical partials.
4. Tune thresholds on conversation-disjoint validation data.

Gate: adapter policy strictly improves the latency-versus-false-cancel Pareto frontier over the
rule baseline.

### Phase 4: Low-latency TTS and stable-prefix prefill

1. Integrate VoXtream2 behind the existing typed TTS interface.
2. Measure cancellation, alignment, short utterances, and persistent-session stability.
3. Add exact-prefix cache reuse.
4. Add stable-user-prefix prefill only after revision tracking exists.

Gate: commit-to-first-played-audio p95 below 350 ms on unambiguous turns, without worse speech
quality or cancellation behavior than Kyutai.

### Phase 5: Assistant backchannels

1. Build a curated audio bank.
2. Train an opportunity/type selector.
3. Add cooldown, self-activity feedback, and unexpected-user-stop recovery.
4. Run user studies before increasing frequency.

Gate: precision >= 0.85, takeover error below 1%, and no increase in user repair burden.

### Phase 6: Prepared interjections and background actions

1. Add transcript-bound response candidates.
2. Add response-conditioned usefulness scoring.
3. Add read-only speculative tools and explicit action commitment.
4. Enable projected-opening interjections with low aggressiveness.
5. Keep competitive interruption behind an urgency gate.

Gate: assistant interruption precision >= 0.90 and interruption regret below 5%.

### Phase 7: Native-duplex parallel track

1. Evaluate Moshi/PersonaPlex-class models on the same replay harness.
2. Include HumDial-FDBench, Full-Duplex-Bench, Zero-Buffer, and EchoChain-style tests.
3. Test interaction alignment or a native interaction/action model.
4. Consider a native model first as a timing/prosody adviser or fast draft path, not the authority
   for irreversible tools.

Gate: must beat the hybrid product path on rejection, interruption comprehension, state revision,
semantic quality, and traceability, not only onset latency.

## 12. Dataset and Annotation Requirements

### Data sources

Use complementary sources:

- **CANDOR:** large natural dyadic conversations and frequent backchannels; audit algorithmic labels.
- **HumDial-FDBench:** controlled human-recorded target scenarios; valuable evaluation and targeted
  training, but partly script-generated and actor-performed rather than fully spontaneous.
- **Fisher/Switchboard:** large separated-channel conversational priors and VAP pretraining.
- **Seamless Interaction:** modern naturalistic/improvised interaction data used in recent
  full-duplex alignment.
- **EgoCom, EasyCom, DiPCo:** noisy and multi-party cross-domain evaluation.
- **Voice Light live sessions:** consented, privacy-aware product distribution and device acoustics.

### Annotation layers

Use a shared media clock and annotate:

1. Speaker and addressee identity.
2. Speech, silence, breath, laughter, and non-speech vocalization spans.
3. Stable and volatile ASR text revisions with word timestamps.
4. Turn hold, pause candidate, projected transition-relevance place, and actual shift.
5. Backchannel span and type: continuer, assessment, empathy, laughter, repair.
6. Floor claim attempt, claim success, and time to other-speaker yield.
7. Cooperative versus competitive overlap.
8. User interruption intent: agreement, correction, disagreement, follow-up, repetition, topic
   switch, stop, urgent repair.
9. Assistant opportunity labels: stay silent, backchannel, projected-opening interjection,
   competitive interrupt.
10. Prepared-response usefulness and semantic completeness at each candidate opening.
11. Post-interruption state-update result: correct, contextual inertia, interruption amnesia,
    objective displacement.
12. Playback and policy actions with sample positions.

### Annotation unit and reliability

- Frame targets at 20-50 ms for activity and floor projection.
- Event spans for backchannels, interruptions, laughter, and overlap.
- Word/phrase boundary alignment for playback decisions.
- Soft labels plus independent annotation reliability.
- At least three annotators for ambiguous cooperative/competitive and backchannel/turn-claim cases.
- Conversation- and speaker-disjoint splits.
- Slice by accent/dialect where permitted, SNR, echo, device, speaking rate, and ASR error.

### Scale

For a production-oriented v1 target:

| Category | Minimum unique gold events |
| --- | ---: |
| Hard pause versus yield | 25,000 |
| User backchannels | 20,000 |
| Laughter/non-speech feedback | 10,000 |
| Corrections/disagreements | 10,000 |
| Follow-ups/topic switches/repetition/stop | 5,000 each |
| Cooperative overlap | 15,000 |
| Competitive interruption | 15,000 |
| Assistant backchannel opportunities | 20,000 |
| Projected-opening interjection opportunities | 10,000 |
| Post-interruption state-update cases | 5,000 |

Create a smaller locked gold evaluation set before large-scale weak labeling.

## 13. Experiment Plan and Acceptance Gates

All latency metrics must report p50, p90, and p95 from monotonic timestamps and sample positions.
Bootstrap confidence intervals by conversation/session.

### A. Endpoint latency versus false-cutoff Pareto

Sweep:

- Silero thresholds and silence duration;
- Nemotron lookahead modes;
- adapter thresholds and hysteresis;
- acoustic-only versus acoustic+lexical fusion.

Report:

- true-end to commit;
- true-end to first played audio;
- false cutoff by pause duration;
- missed shift;
- user repair burden.

Gate: choose no operating point that improves median latency while worsening the locked-test false
cutoff rate beyond 3%. For unambiguous ends, target true-end-to-commit p95 <= 450 ms.

### B. Hard cancel versus duck-classify versus pause-resume

Cross:

- policy;
- event type;
- overlap onset at 200, 500, and 1000 ms into assistant speech;
- assistant word, clause, and phrase positions;
- clean, echo, and noise conditions.

Required scenarios:

- `mm-hm`;
- laughter;
- bare agreement;
- agreement plus clause;
- correction;
- disagreement;
- follow-up;
- repetition request;
- topic switch;
- stop;
- sustained interruption;
- speech to another person.

Gate:

- duck p95 <= 50 ms;
- pause p95 <= 120 ms;
- explicit-stop audible stop p95 <= 120 ms;
- genuine interruption audible stop p95 <= 220 ms;
- backchannel false-cancel below 5%, stretch goal 3%;
- resume success at least 90%;
- backchannel-end-to-resume p95 <= 180 ms, with onset-to-resume reported conditional on
  backchannel duration;
- stale-resume rate below 1%.

### C. Assistant backchannels

Compare:

1. no backchannel;
2. random cached backchannel;
3. rule-selected cached backchannel;
4. learned cached selector;
5. main-Qwen generated;
6. stable-partial speculative generated.

Report:

- opportunity precision and recall;
- onset error from annotated opportunity;
- duration and level;
- takeover rate;
- cooldown violations;
- unexpected-user-stop recovery;
- subjective naturalness, attentiveness, and distraction.

Gate:

- precision >= 0.85 before optimizing recall;
- opportunity recall >= 0.40;
- takeover error below 1%;
- no significant increase in user repair burden;
- generated/speculative systems must beat cached selection on contextual appropriateness without
  missing the timing gate.

### D. Assistant interjection

Compare:

- interject only after detected opening;
- stable-partial prepared candidate;
- response-conditioned prepared candidate;
- native-duplex baseline.

Report:

- assistant interruption precision;
- interruption regret;
- response usefulness;
- projected-opening lead/lag;
- user continuation after assistant onset;
- cooperative and competitive overlap duration;
- conversation-goal advancement.

Gate:

- precision >= 0.90;
- regret below 5%;
- no worse task completion;
- competitive overlap p95 below 250 ms except explicitly annotated cooperative overlap;
- no user-preference subgroup shows a material trust decline.

### E. System prompt comparison

Use identical histories, audio/transcripts, seeds, temperature, and candidate openings. Blind
raters to prompt condition.

Compare:

- current;
- concise conversational;
- cooperative listener;
- explicit speech-act/control.

Report separately:

- first token/word latency;
- word count and clause duration;
- recap/filler rate;
- repair quality;
- candidate usefulness;
- lexical naturalness;
- timing metrics from the unchanged controller.

Gate: adopt a prompt only if semantic quality is non-inferior and median spoken duration falls by at
least 25% relative to current. Prompt changes receive no credit for controller timing that they do
not affect.

### F. Persistent-session GPU tail latency

Run at least 30-minute sessions with overlapping:

- Nemotron streaming ASR;
- turn adapter;
- Qwen prefill/generation;
- Kyutai or VoXtream TTS;
- cancellation/resume churn;
- background read-only tools.

Report:

- p50/p90/p95 ASR chunk, LLM first word, TTS first PCM, cancellation, and resume;
- queue depths;
- GPU utilization and memory;
- first-audio regression from isolated runs;
- worker restarts and errors.

Gate:

- no unbounded queue growth;
- no memory growth after warmup;
- p95 first-audio regression under overlap <= 15%;
- zero stale audio after cancel;
- session error rate below 0.1%.

### G. Semantic quality after interruption and resume

Use EchoChain-style cases plus simple conversational repair cases.

Report:

- correct state update;
- contextual inertia;
- interruption amnesia;
- objective displacement;
- preserved original goals;
- stale tool/result use.

Gate:

- stale semantic continuation below 2%;
- no irreversible speculative side effect;
- interrupted condition no more than 10 percentage points below the paired half-duplex control
  before broad release.

### Required metric dashboard

For every policy build, report p50/p90/p95 where applicable:

- true-end to first played audio;
- commit to first played audio;
- user onset to duck, pause, stop, and resume;
- user backchannel false-cancel rate;
- resume success and stale-resume rate;
- assistant backchannel precision and opportunity recall;
- assistant interruption precision and interruption regret;
- cooperative and competitive overlap duration;
- user repair burden;
- speculative-candidate hit and invalidation rates;
- semantic quality after interruption and resume.

## 14. Architecture Recommendation

### Optimized cascade alone

An optimized cascade can substantially improve ordinary turn latency through better endpointing,
stable-prefix prefill, and VoXtream2. It remains insufficient if it still treats any VAD onset as a
hard turn boundary. Pure half-duplex optimization is therefore not the target architecture.

### Hybrid full-duplex orchestration

This is the recommended product path:

- continuous ASR and interaction prediction;
- explicit floor and playback control;
- Qwen3 1.7B as the semantic model;
- low-latency incremental TTS;
- reversible buffered speech;
- transcript-bound speculation;
- independent structured action/tool state;
- learned timing components where rules stop generalizing.

It preserves Voice Light's measured semantic quality and typed component boundaries while directly
addressing the dominant endpoint and overlap costs.

### Parallel native-duplex track

Maintain this as a research track because native models can ultimately provide:

- richer paralinguistic perception;
- learned overlapping prosody;
- integrated pause/backchannel/interruption behavior;
- lower interaction-control latency; and
- synchronized action streams.

Do not promote it based on simultaneous I/O or advertised codec latency. Promotion requires:

1. strong rejection of backchannels, side speech, and background speech;
2. first-word interruption comprehension;
3. correct post-interruption state revision;
4. comparable semantic and safety quality;
5. stable tool/action behavior;
6. controllable personality; and
7. complete traceability to the shared media clock.

The likely long-term outcome is not a binary cascade-versus-native choice. The product may retain a
typed action and commitment layer while a native model supplies interaction predictions, prosody,
or a fast draft response. For the next implementation cycle, however, hybrid full-duplex
orchestration is the highest-confidence path.

## Final Decision Summary

1. Replace hard-cancel-on-any-speech with immediate duck, boundary pause, and a 500 ms bounded
   classifier.
2. Resume only when the transcript anchor and buffered audio remain valid.
3. Cancel immediately for explicit stop, correction, critical repair, and high-confidence
   floor-taking.
4. Keep Qwen/TTS alive only inside a small, measured speculation budget.
5. Start assistant backchannels with a learned selector over curated cached audio.
6. Keep backchannels out of durable turn history.
7. Permit assistant interjection only at projected openings with a useful prepared response.
8. Separate interaction, speech, generation, and action state.
9. Integrate the existing Nemotron adapter as an initial predictor, while adding missing semantic,
   overlap, and assistant-action targets.
10. Build the product as a hybrid full-duplex orchestrator and keep native duplex as a parallel
    benchmark-driven research track.
