# Causal Turn-Detection Benchmark

## Scope and interpretation

This benchmark compares only Voice Light's primary `p_user_yield` output with external end-of-turn
detectors. The five event labels and four future-activity labels are training auxiliaries and are
excluded from every external comparison. Deepgram Flux is excluded. VAP is not included because its
two-channel inputs and native targets do not share this benchmark's single-channel HOLD/YIELD
contract.

The protocol is adapted from LiveKit eot-bench at source revision
`7f2acca997211908c6ee962ace8bcc8d6a66fbac`. Results are not entries on its public leaderboard:
Voice Light uses overlapping windows and soft conversation labels, not the leaderboard's row and
label contract.

## Reproducible protocol

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

## Validation artifacts and preliminary baselines

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

## Smart Turn overlap boundary

Smart Turn's pinned v3.2 training-data repository is
`pipecat-ai/smart-turn-data-v3.2-train` at
`e564e2ac567f774d1880aa1db6ce97afb8c519b7`. It names MundoAI as a source, while Voice Light
`dataset_3` is Mundo TurnBench-derived. The local audit found this provenance overlap. It did not
download the roughly 41 GB external corpus, so no exact audio/PCM comparison was performed and all
11 local source records remain hash-unverified. Consequently
`clean_comparative_claim_permitted=false`: Smart Turn may be shown as contaminated context, but it
must not support a clean aggregate superiority claim. The audit artifact is
`smart-turn-overlap-audit.json`.

## Voice Light feasibility and GPU handoff

The checkpoint hashes are:

- step 3,500: `60161b84d58149a56351dbbf5875fbf436c5ea90be15965b323140a6eb13e7bc`;
- step 7,000: `8888e7107fded4314ab013213350d94e766b7ff46d61cffc47ebb6a28c31cc65`.

The Nemotron model revision is pinned to `ebe59e5a817142986528bbbee5dba8db7b38ed50`.
A local CPU probe on one real 20-second validation window took 5.08 seconds for shared backbone
features and 0.12 seconds total for both adapters, projecting about 2.12 hours for the 1,503-window
validation pass. The three baselines and all metric analysis are practical locally; the shared
Voice Light pass should run on a GPU.

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

The test split has not been opened or evaluated in this implementation run. This is intentional:
the two Voice Light validation artifacts are required to choose and freeze the checkpoint and policy
first.

## CLI sequence

Run `python -m app.training.turn_taking.benchmark_cli --help` for all arguments. The intended order
is `inventory`, `predict-baseline` and `predict-voice-light`, `analyze`, `overlap-audit`,
`lock-validation`, locked test inventory/prediction generation, then `final-test`. Prediction caches
are typed JSON documents: the manifest owns provenance and hashes, while rows contain only candidate
ID, native timestamp/silence duration, probability, and inference duration. Targets remain in the
separately hashed inventory.
