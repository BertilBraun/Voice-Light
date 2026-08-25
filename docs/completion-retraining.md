# Completion-primary retraining

The next turn-detector experiment trains the semantic completion head as the primary objective. It
uses only confident, causally eligible completion boundaries, samples HOLD and EOT with equal
probability, and retains the other event and future-activity heads as auxiliaries.

## Hardware

Use one CUDA GPU with 16 GB VRAM preferred. An RTX 4090 is preferred for price/performance; an L4,
A10/A10G, RTX 3090, or RTX 4080 is sufficient. The completed experiment also verified that an RTX
3060 with 12 GB works: it reserved 3.85 GiB at physical batch 8. Compute throughput matters more
than memory. Provision at least 8 CPU cores, 32 GB RAM, and 100 GB of local disk for the repository,
model cache, corpus, checkpoints, and prediction artifacts.

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
LiveKit, and both the best and final Voice Light checkpoints. Nemotron inference uses the GPU;
CPU-native baselines run on the same node. Results are written beneath `reports/` and compact
predictions beneath `predictions/`. Pass `--augmentation-profile legacy` for the controlled
gain/noise/40 ms dropout profile; the default is the expanded profile described above.

The command intentionally stops at validation. Do not open the test split until the validation
operating point has been reviewed and locked.

## 2026-08-25 validation result

The first completion-primary run used an RTX 3060 12 GB and stopped at optimizer step 1,125 after
four validations failed to improve upon step 625. Training exposed approximately 2.3 nominal
epochs, peaked at 3.10 GiB allocated / 3.85 GiB reserved, and averaged 0.941 optimizer steps per
second after cold downloads. The selected step-625 checkpoint has SHA-256
`50b8b1295a3bdcdc29528a960a1c98da1f277eedb18b4896b27d35d0c94d6dd9`.

All figures below use the same 1,005-candidate validation inventory and 789 clean-label support
(134 HOLD / 655 EOT). The test split remains unopened.

| Detector | AUROC | EOT AP | BCE | Brier | False cutoff | EOT recall | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Voice Light completion-primary step 625 | 0.6864 | 0.9052 | 0.6270 | 0.1628 | 4.48% | 69.01% | 2.0 s |
| Voice Light completion-primary step 1,125 | 0.6698 | 0.8973 | 0.7158 | 0.1975 | 4.48% | 69.31% | 2.0 s |
| Voice Light old step 3,500 | 0.6866 | 0.8993 | 0.6719 | 0.1843 | 3.73% | 65.80% | 2.0 s |
| Silero silence policy | 0.5504 | 0.8531 | n/a | n/a | 3.73% | 86.56% | 2.0 s |
| Smart Turn v3.2 | 0.5508 | 0.8486 | 1.5510 | 0.3634 | 4.48% | 57.86% | 2.0 s |
| LiveKit v1-mini | 0.5238 | 0.8316 | 0.8732 | 0.2636 | 4.48% | 63.66% | 2.0 s |

The retrain improved AP, calibration, and selected recall, but did not materially improve AUROC and
missed the deployment gate of at least 70% recall at no more than 5% false cutoffs. The gap is
concentrated in `dataset_1-local` (50.73% recall) and `overlap_interruption` (62.17% recall with
5.17% false cutoffs). `turn_shift` recall was 88.99%, while `non_floor_feedback` recall was 81.71%.
Step 1,125 finds two more EOT cases than step 625 at the selected operating points and the same six
false cutoffs. That 0.30 percentage-point recall difference is not persuasive on a validation set
used to choose both thresholds: step 625 remains the checkpoint choice because its AUROC, AP, BCE,
and Brier are all better. Silero remains the stronger endpointing policy under this validation
contract, so the test split must not be opened for this checkpoint.

### Augmentation ablation

A controlled second run retained the completion-primary objective, balanced boundary sampler,
batching, optimizer schedule, seed, validation set, and early-stopping rule, but reverted waveform
augmentation to the earlier gain/noise/40 ms dropout profile. It stopped at step 1,000 and selected
step 500. This isolates the augmentation profile as the intended experimental change, while still
remaining one stochastic training run per condition.

| Augmentation | Checkpoint | AUROC | EOT AP | BCE | Brier | False cutoff | EOT recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expanded | Best, step 625 | 0.6864 | 0.9052 | 0.6270 | 0.1628 | 4.48% (6/134) | 69.01% (452/655) |
| Expanded | Final, step 1,125 | 0.6698 | 0.8973 | 0.7158 | 0.1975 | 4.48% (6/134) | 69.31% (454/655) |
| Legacy | Best, step 500 | 0.6747 | 0.9020 | 0.6558 | 0.1772 | 4.48% (6/134) | 63.51% (416/655) |
| Legacy | Final, step 1,000 | 0.6544 | 0.8924 | 0.7288 | 0.2040 | 4.48% (6/134) | 66.72% (437/655) |

The expanded profile wins the controlled comparison on best-checkpoint discrimination,
calibration, and operational recall. The next run should therefore retain it. This result does not
show that every individual transform helps; isolating reverb, bandwidth degradation, clipping, and
variable packet loss would require further single-factor ablations and is lower priority than the
completion-label audit.

### Completion-label review

The current review package is selected using the expanded step-1,125 predictions. It contains 320
validation cases: 80 ambiguous labels, 120 confident HOLD labels, and 120 confident EOT labels.
Within that panel, 175 cases have cross-model disagreement and 139 have Voice Light disagreement;
120 representative controls are mixed into the blind presentation. Each case includes a stereo
seven-second clip, keyboard review controls, structured error tags, local autosave, hidden automatic
metadata, and typed JSON export. Its item SHA-256 is
`079c19c95a66818a1ad42f4d571d776b8ea74989d6f627570f42b123a4f877aa`.
