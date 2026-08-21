# Turn-Detection Evaluation Plan

## Objective

Evaluate the two Voice Light turn-taking checkpoints as causal HOLD/YIELD detectors and compare only
their primary `p_user_yield` output with pause-gated external baselines. The five event targets and
four future-activity targets remain training auxiliaries and internal diagnostics; they are not part
of any external baseline claim.

The benchmark adapts LiveKit's open eot-bench evaluation model: score real silence decisions using
only causal context, sweep the deployment policy, and report the false-cutoff versus endpointing
latency frontier. It does not claim direct comparability with the public eot-bench leaderboard
because the Voice Light corpus is a windowed, soft-label conversation corpus rather than one complete
user turn per row with a final hard EOT span.

## Audited Inputs

- Corpus: `BertilBraun/voice-light-audio` at immutable revision
  `56e68eb8fb1d42159483612f508b9ce27672f724`.
- Validation: 1,503 overlapping 20-second windows from 11 conversations.
- Test: 1,302 overlapping 20-second windows from 11 different conversations.
- Frame contract: 250 frames at 80 ms per window, with `-1` as the target mask.
- Validation categories: `overlap_interruption`, `hold_pause`, `non_floor_feedback`, `background`,
  and `turn_shift`.
- Step 3,500 checkpoint SHA-256:
  `60161b84d58149a56351dbbf5875fbf436c5ea90be15965b323140a6eb13e7bc`.
- Step 7,000 checkpoint SHA-256:
  `8888e7107fded4314ab013213350d94e766b7ff46d61cffc47ebb6a28c31cc65`.
- Existing dense validation BCE totals are 0.5789657332 and 0.5817793716 respectively. These do
  not select the operational checkpoint.
- Existing local detector code already integrates Silero VAD, Pipecat Smart Turn v3.2 CPU ONNX,
  and LiveKit `v1-mini`, but its UI result type discards sparse candidate probabilities and model
  provenance. The benchmark will reuse inference helpers while owning a separate typed artifact
  contract.

## Causal Candidate Contract

Build the inventory independently for validation and test from the pinned materialized samples:

1. Group windows by dataset, conversation/sample ID, target side, and user audio asset.
2. Convert each valid 80 ms annotation into an absolute recording timestamp.
3. De-duplicate overlapping observations. Identical observations must agree within a documented
   tolerance; conflicting labels are an inventory error rather than silently averaged.
4. Derive contiguous user-silence spans from valid `p_user_has_floor` observations. A span becomes a
   candidate after the configured minimum silence, with score points only at timestamps where that
   detector can natively return a score.
5. Assign the candidate's soft EOT target from `p_user_yield`. For deployment classification,
   explicitly lock a validation-selected target cutoff; preserve the soft value for BCE and Brier
   diagnostics.
6. Keep the candidate's dataset, source category set, conversation ID, absolute timestamps, silence
   duration, source window IDs, and deterministic content/provenance hashes.

No pause-gated model output is copied across every 80 ms frame. Smart Turn, LiveKit, and Voice Light
predictions are sparse rows at their actual candidate timestamps. A silence-only policy may act at a
timeout without a learned score.

## Prediction Artifacts

Use frozen Pydantic models with forbidden extra fields for serialized manifests and rows. Each run
records:

- schema version, corpus repository and revision, split, candidate inventory SHA-256, and row count;
- adapter kind and display name;
- model repository/revision or local checkpoint path, file SHA-256, package version, and full
  inference configuration;
- one row per native candidate score containing the candidate ID, timestamp, silence duration,
  `p_user_yield`/completion probability, target, inference duration, and provenance.

Store compact JSON Lines prediction caches so policy and threshold sweeps never rerun Nemotron,
Smart Turn, or LiveKit inference.

## Baselines And Adapters

1. **Silero VAD plus causal timeout/hysteresis.** Cache native VAD speech probabilities or segments,
   then sweep silence timeouts. This is the timing-only reference and does not expose an EOT
   probability for calibration comparison.
2. **Pipecat Smart Turn v3.2 CPU ONNX.** Run after at least 200 ms of silence on up to the latest
   eight seconds of the current user turn. Pin and hash `smart-turn-v3.2-cpu.onnx`.
3. **LiveKit v1-mini.** Use the installed local CPU model when its dependency contract is available;
   record package/model versions and its native 300 ms candidate gate.
4. **Voice Light steps 3,500 and 7,000.** Extract the frozen Nemotron features once per batch, run
   both adapters on the same taps, and cache only the primary yield probability at candidate frames.
5. **VAP.** Optional contextual analysis only after the primary baselines. Its two-channel inputs and
   native output semantics remain in a separate report and never share the primary ranking table.
6. **Deepgram Flux.** Excluded.

## Policy And Metrics

Tune on validation only. Jointly sweep:

- score threshold;
- minimum action delay after speech;
- maximum silence timeout;
- detector-specific causal hysteresis where applicable.

For every policy report false-cutoff rate on HOLD candidates, EOT recall, mean and p50/p90/p95/p99
endpointing latency on YIELD candidates, and counts/support. Produce Pareto points plus best cutoff
rate at 300/600 ms mean-latency budgets and best mean latency at 5%/10% cutoff budgets when those
points exist. Report BCE, Brier score, and calibration bins only for native probabilities with the
same HOLD/YIELD interpretation. Report dataset/category breakdowns and bootstrap confidence
intervals by conversation when support permits; never bootstrap overlapping windows as independent
examples.

Lock the candidate contract, target cutoff, selected policy, checkpoint choice, and all thresholds
from validation. The test CLI requires that locked manifest and refuses ad-hoc policy options. Run
test once at the final gate.

## Smart Turn Contamination Audit

Smart Turn v3.2's published training data credits MundoAI, while Voice Light `dataset_3` is derived
from Mundo TurnBench. This creates a known provenance-level contamination risk before any byte hash
comparison. The benchmark will:

- compare normalized source names and external IDs;
- support exact full-audio and fixed-duration PCM fingerprint comparison when the external Smart
  Turn inventory is available;
- report exact matches, compared support, unverified support, and provenance-risk sources;
- mark Mundo-derived comparative slices contaminated unless exact evidence proves disjointness;
- present aggregate results both with and without contaminated sources.

Downloading the roughly 41 GB Smart Turn training corpus is not required to run the benchmark. A
provenance-risk result is sufficient to prevent an unsupported clean comparison; exact hash audit is
an optional strengthening step.

## Implementation And Commit Sequence

1. Add typed inventory, prediction, provenance, overlap, and policy models; deterministic JSONL I/O;
   inventory construction; and unit tests.
2. Add pure local metrics, calibration, policy sweep, Pareto selection, locked-policy manifest, and
   tests against hand-calculated examples.
3. Add sparse baseline scorers for Silero, Smart Turn, and LiveKit plus their dependency/model
   provenance and injected test doubles.
4. Add shared Nemotron feature extraction for both checkpoints, compact prediction caching, CPU
   feasibility timing, and a reproducible GPU command when CPU throughput is impractical.
5. Run validation predictions and sweeps, lock the winner and policy, then run the test split once.
6. Publish a result report with interpretation boundaries, contamination exclusions, exact commands,
   and artifact hashes.

Before each feature-sized commit, run `ruff format`, `ruff check --fix`, and the relevant pytest
suite. Work only on `master` and preserve unrelated changes.

## Feasibility Gate

Baseline inference and all metric analysis are CPU-suitable. Benchmark a small deterministic subset
of Nemotron windows locally and extrapolate total runtime. Continue locally only if the projected
validation run is reasonable. Otherwise, use the existing GPU deployment path to generate the two
compact Voice Light prediction caches in one shared-backbone pass, copy only those artifacts back,
and keep tuning, analysis, policy locking, and the final report local. The test split remains sealed
until validation artifacts and the locked manifest exist.
