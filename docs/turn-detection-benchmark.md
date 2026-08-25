# Causal Turn-Detection Benchmark

## Scope and interpretation

The original v1 benchmark compares Voice Light's primary `p_user_yield` output with external
end-of-turn detectors. Its completed validation and test results are retained below as historical
evidence, but its label means future user silence rather than semantic turn completion. The current
v2 validation diagnostic instead evaluates the existing auxiliary `turn_completion` head at
speech-offset boundaries. It is explicitly a diagnostic of whether the saved checkpoints already
contain usable completion signal, not a retrospective external claim about the primary head.

Deepgram Flux is excluded. VAP is not included because its two-channel inputs and native targets do
not share this benchmark's single-channel decision contract.

The protocol is adapted from LiveKit eot-bench at source revision
`7f2acca997211908c6ee962ace8bcc8d6a66fbac`. Results are not entries on its public leaderboard:
Voice Light uses overlapping windows and soft conversation labels, not the leaderboard's row and
label contract.

## V2 boundary-only completion diagnostic

### Candidate and label contract

V2 uses the same pinned corpus revision as v1. It reconstructs absolute 80 ms frame observations
from the overlapping 20-second windows and rejects conflicting duplicates. A candidate begins only
at a frame carrying valid `turn_completion` supervision when all of the following hold:

- the assistant is inactive (`assistant_has_floor < 0.5`);
- the user is no longer speaking (`p_user_has_floor < 0.5`);
- the immediately preceding observed frame belongs to a contiguous user-speech span; and
- the subsequent candidate opportunity is fully observed, either until the user resumes or through
  the fixed two-second causal horizon.

Candidates that lack the required observed suffix are censored instead of padded with labels.
Assistant-active anchors are excluded. A candidate is `continuation` when same-user speech resumes
within two seconds and `terminal` otherwise. The boundary's soft completion target is carried across
the candidate for bookkeeping; no future observation is exposed to detector inference.

The operational labels deliberately use high-confidence bands:

- HOLD requires `turn_completion <= 0.2` and `continuation_pause >= 0.8`;
- EOT requires `turn_completion >= 0.8` without a conflicting
  `continuation_pause >= 0.8`; and
- all other candidates are ignored for cutoff, recall, AUROC, and AP.

The validation inventory contains 1,005 candidates from 1,503 windows but only 11 independent
conversations. Its SHA-256 is
`8f4bd09b216a05a02472c2023d4088c63966ef98546da199808dbc3afc48d8e6`. There are 726 terminal and
279 continuation opportunities. The clean evaluation support is 789: 134 HOLD and 655 EOT, with
216 ambiguous candidates ignored and no completion/continuation conflicts. The EOT prevalence in
the clean set is 83.02%, so EOT average precision must be interpreted against a 0.8302 prevalence
baseline.

| Dataset | HOLD | EOT | Ignored |
| --- | ---: | ---: | ---: |
| `dataset_1-local` | 95 | 343 | 153 |
| `dataset_2` | 1 | 12 | 0 |
| `dataset_3` | 14 | 29 | 10 |
| `meetings-s3` | 24 | 271 | 53 |

These are candidate counts, not independent conversation counts. In particular, dataset-level
percentages cannot be treated as stable estimates with only 11 validation conversations.

### Native gates and policy interpretation

Every detector receives only causal audio, but the detectors do not score at a shared timestamp:

- Voice Light emits one completion-head score at the first native frame, 80 ms after the annotated
  boundary. Its one-frame lookahead and user/assistant inputs remain causal through that scoring
  frame. The boundary-only score is latched for later policy delays; it is not recomputed throughout
  the pause.
- Smart Turn emits one score at 240 ms, the first 80 ms inventory frame after its 200 ms native
  silence gate, and that score is latched.
- LiveKit v1-mini emits one score at 320 ms, the first inventory frame after its 300 ms gate, and
  that score is latched.
- Silero emits current inverse-speech scores on its native 32 ms chunks. Its scores are not latched,
  and they are a silence proxy rather than completion probabilities.

Accordingly, first-score AUROC and AP measure ranking at each model's own native gate, not ranking
after identical amounts of silence. Calibration is reported for Voice Light, Smart Turn, and
LiveKit because they expose completion-like probabilities, but their different gates and the soft
heuristic target limit direct comparison. Silero calibration is undefined here.

A policy sweep covers thresholds 0.05 through 0.95 plus 0.99 and 1.0, action delays from 80 ms
through two seconds, and timeouts from 300 ms through two seconds. Under the 5% cutoff budget,
selection maximizes detector EOT recall, then minimizes timeout-inclusive mean latency and cutoff.
If no point satisfies the budget, selection falls back to the Pareto point with the lowest cutoff
and latency. A HOLD false cutoff is an action before the observed continuation opportunity ends.
EOT recall counts only detector-triggered actions before the timeout or opportunity end; timeout-only
actions do not count. Latency is capped by the shorter of timeout and observed opportunity.

### V2 validation result and decision

| Detector | Threshold / delay / timeout | False cutoff | EOT recall | Mean latency | AUROC | EOT AP | BCE | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Voice Light completion step 3,500 | 0.50 / 1,600 ms / 2,000 ms | 3.73% (5/134) | 65.80% (431/655) | 1,732 ms | 0.6866 | 0.8993 | 0.6719 | 0.1843 | 0.2175 |
| Voice Light completion step 7,000 | 0.50 / 1,600 ms / 2,000 ms | 4.48% (6/134) | 65.80% (431/655) | 1,732 ms | 0.6832 | 0.8952 | 0.6803 | 0.1875 | 0.2194 |
| Smart Turn v3.2 | 0.05 / 1,600 ms / 2,000 ms | 4.48% (6/134) | 70.53% (462/655) | 1,713 ms | 0.5595 | 0.8532 | 1.5817 | 0.3653 | 0.4111 |
| LiveKit v1-mini | 0.35 / 1,600 ms / 2,000 ms | 4.48% (6/134) | 63.66% (417/655) | 1,741 ms | 0.5238 | 0.8316 | 0.8732 | 0.2636 | 0.2864 |
| Silero VAD 6.2.1 | 0.99 / 1,600 ms / 2,000 ms | 3.73% (5/134) | 86.56% (567/655) | 1,676 ms | 0.5504 | 0.8531 | n/a | n/a | n/a |

The probability metrics use all 1,005 soft completion targets, whereas AUROC and AP use only the
789 clean hard labels. A constant soft-prior predictor has BCE 0.5847 and Brier 0.1453. Both Voice
Light checkpoints are worse than that calibration baseline despite ranking above chance, so the
completion head is under-confident and not a satisfactory probability estimator.

Every selected policy has p95 EOT latency of two seconds. The Voice Light, Smart Turn, and LiveKit
points have p50 latency of 1.6 seconds; Silero's p50 is 1.616 seconds. No detector has a swept point
that simultaneously achieves false cutoff at or below 5%, EOT recall at or above 70%, and p95
latency at or below 800 ms. Silero's high recall comes from a delayed sequential silence policy, and
its first-score AUROC/AP are not semantically comparable to completion-probability models. Smart
Turn remains contamination-risk context because its training data and Voice Light `dataset_3` both
include Mundo-derived material; it cannot support a clean aggregate superiority claim.

Step 3,500 has some completion ranking signal, but the validation diagnostic fails the predefined
go/no-go standard: AUROC is below 0.75, and reaching 65.80% recall requires a 1.6-second action
delay with two-second p95 latency. Step 7,000 is no better. The v2 test split must therefore remain
unopened; there is no v2 test inventory, threshold lock, or test result.

The next step is a deterministic human audit of the validation completion labels, especially the
216 ignored cases, continuation pauses, backchannels, overlap, and dataset-specific errors. If that
audit supports the label contract, semantic completion/EOT should become the primary training
objective and the model should be retrained. Continuing the old `p_user_yield` objective or merely
training its current checkpoint longer is not justified by these results.

### Completion-label audit package

`completion-label-audit-v2` builds a validation-only, deterministic 320-case review package from
the v2 inventory and the step-3,500, Smart Turn, and LiveKit prediction artifacts:

- 80 ambiguous cases ranked by label midpoint uncertainty, completion/continuation tension, Voice
  Light disagreement, and cross-model disagreement;
- 120 confident HOLD cases and 120 confident EOT cases, each split evenly between high-disagreement
  challenges and deterministic representative controls; and
- 100 cases flagged for independent double review.

Selection is balanced round-robin across dataset/conversation streams before filling additional
slots. Presentation order is separately SHA-256 shuffled so reviewers do not encounter the three
groups in blocks. The generated review page hides the automatic group, selection reason, and model
scores until the reviewer opens the metadata panel. Reviewer-local autosave is keyed by both the
manifest hash and reviewer name, preventing one reviewer from seeing another's decisions.

Each seven-second, 16 kHz stereo WAV contains four seconds before and three seconds after the
boundary. The candidate user is on the left and the other speaker is on the right. The visual
reviewer renders these as separate synchronized waveforms with a pink `t = 0` decision line,
pre-boundary and outcome-context regions, relative time ticks, a playback cursor, click-to-seek,
and playback ranges that stop at or surround the boundary. It explicitly asks whether starting the
assistant at that line would cut off the candidate user. Keyboard decisions are `1` safe to take,
`2` hold, and `3` ambiguous/unratable. Structured error tags, notes, reviewer-local progress, and
typed JSON export are also available. `review-template.csv` is a non-interactive fallback.

Open an existing package through a local HTTP origin so audio, autosave, and canvas behavior use a
normal browser security context:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory .cache\local\training-runs\2026-08-25-completion-v1\completion-label-audit
```

Then open `http://127.0.0.1:8765/index.html`. After changing only the reviewer UI, rebuild an
existing package without regenerating its clips or selection manifest:

```powershell
.\.venv\Scripts\python.exe -m app.training.turn_taking.benchmark_cli refresh-completion-label-audit-ui-v2 .cache\local\training-runs\2026-08-25-completion-v1\completion-label-audit
```

The generated package is
`.cache/local/training-runs/2026-08-21-4080-pilot/benchmark/completion-label-audit-v1/`.
Its item SHA-256 is
`f30bd9d18f382116dd8042411025633a80a016a90a3880f13deb2882f32389f4`. It contains all 13
available `dataset_2` candidates and covers all ten validation conversations that produced v2
candidates. The selected ambiguous panel contains 72 middle-band targets and eight low-completion
cases lacking strong HOLD confirmation; 47 end in a same-user continuation within the two-second
opportunity and 33 do not.

After one or more reviewers export their JSON files, the analysis command validates reviewer and
manifest identity, rejects duplicate or unknown items, forms per-item consensus, reports automatic
label agreement and unsafe EOT errors, and calculates exact double-review agreement and Cohen's
kappa:

```powershell
$auditRoot = '.cache\local\training-runs\2026-08-21-4080-pilot\benchmark\completion-label-audit-v1'
.\.venv\Scripts\python.exe -m app.training.turn_taking.benchmark_cli analyze-completion-label-audit-v2 (Join-Path $auditRoot 'audit-manifest.json') (Join-Path $auditRoot 'analysis.json') reviewer-a.json reviewer-b.json
```

The label gate remains at least 90% agreement on confident labels, at most 5% unsafe EOT errors,
at least 85% exact double-review agreement, and preferably Cohen's kappa of at least 0.70. The
ambiguous panel is diagnostic and must not be used to estimate population-wide error prevalence.

The v2 reports and compact predictions are under
`.cache/local/training-runs/2026-08-21-4080-pilot/benchmark/`. Prediction row hashes are
`469a2491f65027454e0cf085c8964813711975a82ab2bf4670fbcfe1c00272e2` for Voice Light step 3,500,
`8e341a4ca0617ef09b38741d737271d5035484bc2502e4adaf1254b4ac653c8e` for step 7,000,
`de03b4bfecd9b4b77e7249a66f4b7d71c635c54b8c3450b56b95afda70dbde27` for Smart Turn,
`aa5150c09f33c08834767892bbab8ffcf9d038e01a2704511a0d5b9d08ee9414` for LiveKit, and
`43a91468299f53da1f71c3db298600d4209b922be31376d1ed33d32e91395994` for Silero.

The validation sequence is reproducible with the v2-only commands below. Repeat the baseline
prediction and analysis commands for `smart-turn` and `livekit`, and analyze each generated Voice
Light checkpoint artifact separately.

```powershell
$benchmarkRoot = '.cache\local\training-runs\2026-08-21-4080-pilot\benchmark'
$inventory = Join-Path $benchmarkRoot 'v2-validation-inventory.json'
.\.venv\Scripts\python.exe -m app.training.turn_taking.benchmark_cli completion-inventory-v2 $inventory
.\.venv\Scripts\python.exe -m app.training.turn_taking.benchmark_cli predict-completion-baseline-v2 $inventory (Join-Path $benchmarkRoot 'v2-validation-silero-predictions.json') --detector silero
.\.venv\Scripts\python.exe -m app.training.turn_taking.benchmark_cli predict-voice-light-completion-v2 $inventory $benchmarkRoot '.cache\local\training-runs\2026-08-21-4080-pilot\adapter-pilot.pt' '.cache\local\training-runs\2026-08-21-4080-pilot\adapter-step-007000.pt' --batch-size 4 --data-loader-workers 0
.\.venv\Scripts\python.exe -m app.training.turn_taking.benchmark_cli analyze-completion-v2 $inventory (Join-Path $benchmarkRoot 'v2-validation-silero-predictions.json') (Join-Path $benchmarkRoot 'v2-validation-silero-analysis.json')
```

## Historical v1 future-silence protocol

The source corpus is `BertilBraun/voice-light-audio` at revision
`56e68eb8fb1d42159483612f508b9ce27672f724`. The inventory reconstructs absolute timestamps from
overlapping 20-second windows, rejects conflicting duplicate observations, and creates maximal
causal user-silence candidates. The fixed scoring label is the first native 80 ms target point at or
after 200 ms of silence, classified as EOT when `p_user_yield >= 0.5`. Soft values are retained for
calibration. The candidate ID, source windows, recording, side, dataset, categories, timestamps, and
inventory hash make candidate selection deterministic.

Models are evaluated only at timestamps where they produce a native causal output:

- Voice Light scores its native 80 ms output frames. Both adapters share one frozen Nemotron pass.
- Silero VAD 6.2.1 scores causal 512-sample chunks at 16 kHz (32 ms). Its state is reset per
  candidate after replaying the candidate's preceding speech.
- Pipecat Smart Turn v3.2 scores once at the first inventory timestamp at or after its 200 ms gate,
  using at most the latest eight seconds. The pinned CPU ONNX revision is
  `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`, and the model SHA-256 is
  `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`.
- LiveKit local turn detector `v1-mini` scores once at the first inventory timestamp at or after its
  300 ms gate. The measured package was `livekit-local-inference` 0.2.7. This is research-only use;
  its licensing does not imply permission to redistribute the model.

Sparse pause-gated scores are latched after their native candidate timestamp; they are never copied
onto each 80 ms frame. A policy jointly specifies score threshold, minimum action delay, and maximum
silence timeout. Validation sweeps thresholds 0.05 through 0.95, delays 80 through 640 ms, and
timeouts 300 through 1,600 ms. Selection minimizes latency under a 5% false-cutoff budget, falls back
to 10%, then to the Pareto frontier. Reports include false cutoffs on HOLD, detector EOT recall,
timeout-inclusive mean and p50/p90/p95/p99 EOT latency, Pareto and budget operating points,
dataset/category breakdowns, and native-timestamp BCE/Brier/ECE where output semantics are
comparable. Silero is excluded from calibration because VAD speech probability is not EOT
probability.

The `lock-validation` command freezes every detector's provenance, prediction hash, selected policy,
validation metrics, the chosen Voice Light checkpoint SHA, and the contamination verdict. Test
inventory and prediction commands require that lock. `final-test` has no sweep options and rejects a
prediction artifact whose full detector provenance is absent from the lock.

## Historical v1 validation artifacts and preliminary baselines

The generated validation inventory contains 1,547 silence candidates from 1,503 windows and 11
conversations. Its candidate SHA-256 is
`ec97e2c93a629dab8bc3a6eeb5c7fd265936f241ef1d9094c5afa87b9a6a687b`. At the fixed label point it
contains 50 HOLD and 1,497 EOT candidates, so the false-cutoff denominator is small and differences
of one or two HOLD errors matter. These baseline values are preliminary context, not the final model
comparison:

| Detector | Selected threshold / delay / timeout | False cutoff | EOT recall | Mean latency | BCE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Silero VAD 6.2.1 | 0.05 / 640 ms / 800 ms | 8% (4/50) | 95.66% | 656 ms | n/a | n/a |
| Smart Turn v3.2 | 0.95 / 80 ms / 800 ms | 4% (2/50) | 20.57% | 685 ms | 1.8351 | 0.3929 |
| LiveKit v1-mini | 0.15 / 640 ms / 800 ms | 4% (2/50) | 87.11% | 661 ms | 0.8894 | 0.2502 |

Prediction row hashes are:

- Silero: `40786c4e169883ca13a35a19ce2911abef262ccec8b5916a7f54624c502b9d1f`
  (353,959 native rows).
- Smart Turn: `c3ae5ed716322adeee9745b60dda932fd0d2982a874b6090c6be4ec621482a5e`
  (1,547 rows).
- LiveKit: `8c428fa06dcee88844be9f2a823f5d870f0d77f4773f12854b755c11f1982ed7`
  (1,547 rows).

The artifacts live under
`.cache/local/training-runs/2026-08-21-4080-pilot/benchmark/` and are intentionally ignored by Git.

### Voice Light validation selection

Both checkpoints used one completed shared-backbone local CPU pass. Step 3,500 was locked as the
operational checkpoint because it produced one false cutoff rather than two; step 7,000's 2.5 ms
mean-latency improvement and one percentage point of detector EOT recall did not justify doubling
the observed interruption count on the small HOLD support.

| Checkpoint | Locked threshold / delay / timeout | False cutoff | EOT recall | Mean latency | BCE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Step 3,500 | 0.90 / 560 ms / 800 ms | 2% (1/50) | 13.56% | 768 ms | 0.3705 | 0.0577 |
| Step 7,000 | 0.90 / 560 ms / 800 ms | 4% (2/50) | 14.56% | 766 ms | 0.3741 | 0.0593 |

The validation prediction hashes are
`0cc3ebac2c4b3075e739a5d5efcd22509337833b0e7e2c22077b14e00f0d7ea1` for step 3,500 and
`f72f5ed023a9fc30f79e5682337bf2f749b10d47826d66dd55a543ece2ecd37c` for step 7,000.

## Smart Turn overlap boundary for both protocols

Smart Turn's pinned v3.2 training-data repository is
`pipecat-ai/smart-turn-data-v3.2-train` at
`e564e2ac567f774d1880aa1db6ce97afb8c519b7`. It names MundoAI as a source, while Voice Light
`dataset_3` is Mundo TurnBench-derived. The local audit found this provenance overlap. It did not
download the roughly 41 GB external corpus, so no exact audio/PCM comparison was performed and all
11 local source records remain hash-unverified. Consequently
`clean_comparative_claim_permitted=false`: Smart Turn may be shown as contaminated context, but it
must not support a clean aggregate superiority claim. The audit artifact is
`smart-turn-overlap-audit.json`.

## Historical v1 Voice Light feasibility and execution

The checkpoint hashes are:

- step 3,500: `60161b84d58149a56351dbbf5875fbf436c5ea90be15965b323140a6eb13e7bc`;
- step 7,000: `8888e7107fded4314ab013213350d94e766b7ff46d61cffc47ebb6a28c31cc65`.

The Nemotron model revision is pinned to `ebe59e5a817142986528bbbee5dba8db7b38ed50`.
A local CPU probe on one real 20-second validation window took 5.08 seconds for shared backbone
features and 0.12 seconds total for both adapters, projecting about 2.12 hours for the 1,503-window
validation pass. The complete validation and test runs were subsequently executed locally with
per-batch elapsed-time and ETA reporting. The three baselines and all metric analysis were also run
locally.

On a prepared GPU workspace containing this Git revision, the two checkpoints, and the Hugging Face
cache, run:

```powershell
.venv/bin/python -m app.training.turn_taking.benchmark_cli inventory /workspace/turn-benchmark/validation-inventory.json --hub-cache-directory /workspace/hf-cache
.venv/bin/python -m app.training.turn_taking.benchmark_cli predict-voice-light /workspace/turn-benchmark/validation-inventory.json /workspace/turn-benchmark /workspace/adapter-pilot.pt /workspace/adapter-step-007000.pt --hub-cache-directory /workspace/hf-cache --batch-size 8 --data-loader-workers 4
```

Copy only the inventory and the two compact prediction JSON artifacts back. Run `analyze` locally
for both, select the operational checkpoint from turn-decision metrics rather than the existing
dense validation BCE totals, and create the lock with both Voice Light reports plus the baseline
reports. Only then create the test inventory:

```powershell
.\.venv\Scripts\python.exe -m app.training.turn_taking.benchmark_cli inventory .cache\local\training-runs\2026-08-21-4080-pilot\benchmark\test-inventory.json --split test --validation-lock .cache\local\training-runs\2026-08-21-4080-pilot\benchmark\validation-lock.json
```

## Historical v1 locked test result

The test inventory was created only after `validation-lock.json` selected step 3,500. It contains
1,673 candidates from 1,302 windows and 11 conversations, with candidate SHA-256
`402408bd4f6ae20e5f66c0eed6c41004eea102ba0b25298022aec410555a788e9`. The fixed label point has 37
HOLD and 1,636 EOT candidates. The first inference attempt stopped before producing an artifact when
a final source window exceeded its FLAC by 10,069 decoded samples. The evaluator now permits
zero-padding only for a contiguous missing suffix, while training and interior gaps remain strict.
The last scored candidate in that recording ends more than five seconds before the real audio end,
so the padded samples cannot affect a causal scored output. No partial results were analyzed or used
to change the lock.

The completed locked-policy results are:

| Detector | Locked threshold / delay / timeout | False cutoff | EOT recall | Mean latency | BCE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Voice Light step 3,500 | 0.90 / 560 ms / 800 ms | 2.70% (1/37) | 12.53% | 770 ms | 0.3899 | 0.0590 |
| Silero VAD 6.2.1 | 0.05 / 640 ms / 800 ms | 2.70% (1/37) | 95.60% | 656 ms | n/a | n/a |
| Smart Turn v3.2 | 0.95 / 80 ms / 800 ms | 13.51% (5/37) | 20.72% | 684 ms | 1.7473 | 0.3820 |
| LiveKit v1-mini | 0.15 / 640 ms / 800 ms | 2.70% (1/37) | 91.50% | 654 ms | 0.7809 | 0.2098 |

Voice Light preserves a low false-cutoff rate but crosses its learned threshold for only 205 of
1,636 EOT candidates; most endpoint decisions therefore use the locked 800 ms timeout. On this
protocol it does not outperform either the Silero timing policy or LiveKit v1-mini. The HOLD support
is only 37, so one error changes the reported cutoff rate by 2.70 percentage points. Smart Turn's
result remains contamination-risk context and cannot support a clean comparative claim.

Test prediction hashes are:

- Voice Light step 3,500: `45533fe5842c2712bea4b2f75970df43a74974e9c64614f4bc80dea677ba846a`.
- Silero: `984d2155cfc420e3d208ca81a0684b5c585f18301623412a528ed7cf2a327d3f`.
- Smart Turn: `709c5c9cf530f835a2b3bd24f9de9f8e8d727c4e2f785b6d3b793a3469ffe920`.
- LiveKit: `5d7b38e213607e3744805dc263009f0b8e833213ca87b5adc898cf952c1cd973`.

## Historical v1 CLI sequence

Run `python -m app.training.turn_taking.benchmark_cli --help` for all arguments. The intended order
is `inventory`, `predict-baseline` and `predict-voice-light`, `analyze`, `overlap-audit`,
`lock-validation`, locked test inventory/prediction generation, then `final-test`. Prediction caches
are typed JSON documents: the manifest owns provenance and hashes, while rows contain only candidate
ID, native timestamp/silence duration, probability, and inference duration. Targets remain in the
separately hashed inventory.
