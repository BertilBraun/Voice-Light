# Natural Interaction Research Brief

Conduct a ground-up, research-backed design study for Voice Light's natural conversational
interaction. This is a decision task, not an implementation task. Treat current behavior as a
prototype rather than the intended final architecture.

## Primary Goal

Determine how Voice Light should support:

- User backchannels while the assistant is speaking.
- Assistant backchannels while the user is speaking.
- User interruption of the assistant.
- Assistant-initiated interjection.
- Cooperative and competitive overlap.
- Assistant duck, pause, resume, continue, phrase-boundary stop, and cancel.
- Background semantic reasoning and tool work independent of audible speech.

The central candidate policy is immediate audio ducking followed by a short classification window:
resume after a user backchannel, but yield and cancel stale speech after a genuine floor-taking
attempt. Research whether this is preferable to immediate hard cancellation or buffered pause.

## Measured Baseline

- True user speech end to first assistant PCM: approximately 1.8 seconds.
- Endpointing contributes approximately 966 ms in the measured client path.
- ASR finalization is normally 70-100 ms.
- Qwen3 1.7B first complete word is approximately 80-120 ms depending on integration.
- Live Kyutai first word to first PCM is approximately 640 ms.
- Isolated Kyutai first word to first PCM is approximately 469 ms.
- VoXtream2 fixed-prompt first PCM is approximately 113 ms wall-clock and supports genuinely
  incremental text.
- Qwen3 0.6B saves only about 10 ms to the first complete word and loses substantial conversational
  quality.

Do not assume these stages are strictly additive. Identify which work can overlap.

## Required Decisions

### User Speech During Assistant Playback

Define the first 500 ms after user speech begins. Compare:

- Immediate hard cancellation.
- Immediate duck followed by acoustic and lexical classification.
- Buffered pause with generation continuing.
- Playback pause and generation pause.
- Continue until the next word or phrase boundary.
- Continue speaking while classifying.
- Cancel only after evidence of a competitive floor-taking attempt.

Specify duck and classification deadlines, maximum resumable buffer age, phrase-boundary stop
rules, when Qwen/TTS remain useful after playback pauses, and when generated material becomes
semantically stale.

Treat `mm-hm`, laughter, agreement, correction, disagreement, topic switch, repetition request,
and sustained interruption as distinct cases.

### Assistant Backchannels

Compare:

- Curated presynthesized audio.
- A learned selector over cached audio.
- Main Qwen plus streaming TTS.
- A small dedicated interaction model.
- Speculative generation while the user speaks.

Define timing, duration, volume, prosody, cooldown, eligibility, durable-history semantics, and
recovery when the user unexpectedly stops after a backchannel.

A 200 ms budget is realistic only when the opportunity is predicted and the response or audio is
already prepared.

### Assistant Interjection

Evaluate this initial conversational personality:

> Cooperative listener: backchannel when confident that the user retains the floor; interject at a
> projected opening only when a useful response is ready; competitively interrupt only for urgent
> repair or a critical misunderstanding.

Specify what future-activity, semantic-completeness, prepared-response, pause-versus-yield, urgency,
and conversation-goal evidence is required before taking the floor.

## Orthogonal State Models

Do not build one monolithic state machine. Define transitions, deadlines, and invariants for:

```text
User activity:
SILENT
SPEAKING
PAUSE_CANDIDATE

Playback:
IDLE
QUEUED
SPEAKING
DUCKING
PAUSED_BUFFERED
RESUMING
DRAINING_TO_BOUNDARY
CANCELLED
COMPLETED

Floor relation:
USER_HOLDS
AGENT_HOLDS
OPEN_TRANSITION
OVERLAP_UNRESOLVED

Generation:
PREFILLING
READY
STREAMING
PAUSED_AT_BOUNDARY
INVALIDATED
CANCELLED
COMMITTED
```

## Protocol Requirements

Propose typed events for:

```text
InteractionPrediction
  p_user_speech
  p_user_yield
  p_user_backchannel
  p_user_interruption
  future_user_activity_horizons
  assistant_playback_state

FloorAction
  hold
  duck
  pause
  resume
  continue
  yield
  take_floor
  emit_backchannel
  cancel

SpeechAct
  full_turn
  interjection
  backchannel
  repair
  bridge

TranscriptRevision
  stable_prefix
  volatile_suffix
  revision_id
  audio_sample_position
```

Every decision must be traceable to monotonic time, media sample position, transcript revision,
model confidence, and causal source.

## System Prompting

Evaluate prompting explicitly.

Prompting may improve spoken brevity, interruptible clauses, reduced recap, social calibration,
repair language, and whether a prepared response is useful enough to schedule.

Prompting cannot solve acoustic pause interpretation, early overlap classification, playback
duck/pause/resume, sub-200 ms timing when generation begins after the opportunity, or continuous
simultaneous listening and speaking.

Compare the current prompt against concise conversational, cooperative-listener, and explicit
speech-act/control prompts. Keep lexical naturalness separate from interaction timing.

## Orchestration Versus Training

Likely orchestration:

- Playback duck/pause/resume.
- Separating backchannels from durable turns.
- Rule-based overlap handling.
- Presynthesized backchannels.
- Stable-ASR-prefix prefill.
- Transcript-revision-bound speculative responses.
- Async tools and structured action state.
- Phrase-boundary cancellation.
- Candidate invalidation and stale-result handling.

Likely training:

- Pause versus end-of-turn.
- Backchannel versus floor-taking interruption.
- Transition-opportunity prediction.
- Cooperative versus competitive overlap.
- Assistant backchannel timing.
- Whether the assistant should take the floor.
- Response-conditioned interruption policy.
- Appropriate overlapping prosody.

Evaluate whether the existing frozen-Nemotron turn-taking adapter has sufficient outputs for an
initial controller and identify missing targets.

## Required Experiments and Metrics

Run:

- Endpoint latency versus false-cutoff Pareto sweeps.
- Hard-cancel versus duck-then-classify versus pause-then-resume.
- User backchannel, laughter, correction, disagreement, follow-up, and topic-switch scenarios.
- Presynthesized versus generated versus speculative assistant backchannels.
- Assistant interjection after detected openings versus stable-partial speculation.
- Blinded system-prompt comparison using identical histories and controlled seeds.
- Persistent-session GPU tail latency while ASR, LLM, and TTS overlap.

Report p50, p90, and p95 for:

- True-end to first played audio.
- Commit to first played audio.
- User onset to duck, pause, stop, and resume.
- User backchannel false-cancel rate.
- Resume success and stale-resume rate.
- Assistant backchannel precision and opportunity recall.
- Assistant interruption precision and interruption regret.
- Cooperative and competitive overlap duration.
- User repair burden.
- Speculative-candidate hit and invalidation rates.
- Semantic quality after interruption and resume.

## Required Output

Produce:

1. Current overlap behavior map.
2. Recommended initial conversational personality.
3. State machines and transition tables.
4. Typed protocol proposal.
5. Duck/pause/resume/continue/cancel decision matrix.
6. Firm boundary between prompting, orchestration, and training.
7. Ordered implementation and research plan.
8. Dataset and annotation requirements.
9. Experiment plan with acceptance gates.
10. Recommendation among optimized cascade, hybrid full-duplex orchestration, and a parallel
    native-duplex model track.

Use current primary research, including Voice Activity Projection, continuous VAP, Moshi,
HumDial-FDBench, and recent full-duplex interaction-alignment work. Do not assume that a
full-duplex model is naturally well timed merely because it can consume and produce simultaneous
audio.
