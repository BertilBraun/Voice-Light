# Tomorrow: Turn-Taking Training Experiment

## Starting point

Begin only after the local corpus has passed the validation gates in
`docs/training-corpus-preparation.md` and the exact corpus revision has been published privately on
Hugging Face. Do not ingest more meeting data for this experiment.

## 1. Complete the trainer

- Make the Hugging Face loader select exactly one of `train`, `validation`, or `test`.
- Pin both the corpus revision and the Nemotron backbone revision.
- Efficiently seek to referenced 20-second FLAC windows and reuse the Hugging Face file cache.
- Apply augmentation only to the training split.
- Add deterministic event/category-aware sampling and report the samples seen per source dataset.
- Validate at a configurable interval, retain the best checkpoint, and implement early stopping.
- Save and resume the adapter, optimizer, scheduler, step, configuration, and random state.
- Export an inference artifact containing adapter weights, architecture configuration, operating
  threshold, hysteresis, backbone revision, corpus revision, Git commit, and hashes.

## 2. Run the four training gates

1. Pipeline test: load several Hub rows, run forward and backward, then save and reload a checkpoint.
2. Overfit test: deliberately overfit one or two conversations. Stop and fix the pipeline if this
   fails.
3. Smoke run: train for 100-500 optimizer steps and verify losses, validation metrics, memory, and
   throughput.
4. Pilot run: train for at most about 10,000 optimizer steps, with a 300-500-step warmup, validation
   every 250-500 steps, and early stopping.

Use a Linux compute node with one CUDA GPU, BF16 where supported, and persistent storage for the
Hugging Face and model caches. Start with a 24 GB GPU; increase memory only if the measured batch
size requires it.

## 3. Evaluate the locked test split

- Tune thresholds and hysteresis on validation only.
- Compare with silence/VAD, transcript-gap, Smart Turn, LiveKit, TurnSense, and every other
  operational repository baseline.
- Report false starts per hour, missed shifts per hour, false cutoffs during continuation pauses,
  response latency p50/p90/p95, backchannel confusion, calibration, and per-dataset slices.
- Bootstrap uncertainty by conversation rather than frame.
- Continue only if the adapter improves the latency-versus-false-cutoff trade-off over the best
  simple baseline and the result is directionally consistent across source datasets.

## 4. Validate streaming behavior

- Replay held-out conversations through the actual Nemotron cache API at 80, 160, and 560 ms.
- Carry both the encoder caches and adapter GRU state across chunks.
- Measure offline-versus-streaming probability drift, real-time factor, p95 compute per chunk, and
  peak GPU memory.
- Do not integrate an adapter whose offline result cannot be reproduced causally in streaming mode.

## 5. Package and integrate on Modal

- Publish the selected adapter and its complete run manifest to a private, revisioned model
  repository, then download and verify the artifact locally.
- Add the adapter behind a feature flag in the voice service.
- Drive assistant gain ducking from the calibrated floor-taking probability and retain a separate
  hard interruption threshold.
- Run it in shadow mode against the current detector before allowing it to control playback.
- Deploy the validated stack on the intended Modal GPU and confirm cold start, steady-state latency,
  concurrency, and cost.

## Definition of done

Tomorrow's experiment is complete when a fresh Linux node can reproduce training from pinned
revisions, the locked-test comparison has been generated, and the result gives a clear decision:
integrate, revise the model/labels, acquire more data, or stop.
