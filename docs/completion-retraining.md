# Completion-primary retraining

The next turn-detector experiment trains the semantic completion head as the primary objective. It
uses only confident, causally eligible completion boundaries, samples HOLD and EOT with equal
probability, and retains the other event and future-activity heads as auxiliaries.

## Hardware

Use one CUDA GPU with at least 16 GB VRAM. An RTX 4090 is preferred for price/performance; an L4,
A10/A10G, RTX 3090, or RTX 4080 is sufficient. The previous run reserved about 3.9 GiB, so compute
throughput matters more than memory. Provision at least 8 CPU cores, 32 GB RAM, and 100 GB of local
disk for the repository, model cache, corpus, checkpoints, and prediction artifacts.

## Training population and schedule

At corpus revision `56e68eb8fb1d42159483612f508b9ce27672f724`, the train split produces 7,795
clean causal boundaries before replacement sampling: 1,221 HOLD and 6,574 EOT. Weighted sampling
allocates half of draws to each class. A selected boundary owns the primary completion target for
its 20-second window; other completion anchors in that load are masked from the primary loss.

The node recipe uses physical batch 8 and two-way gradient accumulation, for effective batch 16.
One epoch is approximately 487 optimizer steps. The 2,500-step ceiling is about 5.1 epochs, but
validation begins immediately, checkpoint selection starts at step 1,000, and training stops after
four consecutive non-improving validations. Validation runs every 125 steps and the best AUROC
checkpoint is preserved separately.

Training augmentation is redrawn every load and can combine gain, Gaussian noise, synthetic room
reflections, 8/12 kHz bandwidth degradation, hard clipping, and a short packet-loss span. A resumed
invocation advances the run seed by the saved optimizer step unless `--run-seed` is supplied.

## One-command node run

Install the project with its locked Linux environment, confirm `torch.cuda.is_available()` is true,
and run:

```powershell
python -m app.training.turn_taking.completion_experiment_cli `
  /workspace/voice-light-runs/completion-v1 `
  --hub-cache-directory /workspace/hf-cache
```

The command trains with per-10-step speed and ETA output, evaluates clean completion AUROC/AP/BCE
and Brier every 125 steps, and then runs the validation completion protocol for Silero, Smart Turn,
LiveKit, and the best Voice Light checkpoint. Nemotron inference uses the GPU; CPU-native baselines
run on the same node. Results are written beneath `reports/` and compact predictions beneath
`predictions/`.

The command intentionally stops at validation. Do not open the test split until the validation
operating point has been reviewed and locked.
