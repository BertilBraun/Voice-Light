# Natural Spoken Tool-Use Fine-Tuning Results

This report records the completed Voice Light proof of concept for teaching a small language model
to speak naturally before a tool call, emit valid structured calls, consume tool results, continue
the same conversation, and avoid spurious calls in ordinary dialogue.

No adapter has been merged, integrated, or deployed. This is an offline training and evaluation
result.

## Result in one paragraph

A BF16 rank-16 LoRA adapter was trained for the pinned `Qwen/Qwen3-1.7B` model on 3,919 synthetic
source segments, independently regrouped across 16 logical passes. On an 80-record balanced
holdout, the final adapter raised correct tool decisions from 80.0% to 95.2%, exact tool names from
78.6% to 94.3%, concise spoken bridges from 7.6% to 94.3%, and correct no-tool behavior from 70.0%
to 100%. Teacher-forced validation loss was best much earlier and then worsened, indicating
overfitting to exact synthetic targets, but free-generation task behavior continued to improve.
For this experiment, checkpoint selection by validation loss alone would have selected the worse
behavioral model.

## Experiment lineage

| Stage | Commit | Purpose |
| --- | --- | --- |
| Initial schema and generator | `2cfb6cb` | Provider-neutral records and staged rollouts |
| Randomized time outcomes | `679a145` | Prevent one learned timestamp |
| Naturalness and grounding iterations | `fc658ac`–`c9c1ac9` | Better dialogue, search correction, constructive advice |
| Final unfiltered generator | `0b807fc` | Disable counterproductive semantic rejection |
| Initial LoRA pipeline | `e503269` | Composition, rendering, masking, and training |
| Scaled proof of concept | `de4e67d` | Balanced 80 holdout, 16 passes, baseline and TensorBoard |
| Complete pass coverage | `bf1e0b4` | Guarantee every training record in every pass |
| Behavioral evaluation | `83bbe00` | Base-versus-adapter production-setting generation |

The [synthetic-data report](tool-use-synthetic-generation.md) documents teacher generation.
The [training runbook](tool-use-lora-training.md) documents reproduction and retention.

## Dataset

The canonical source was `production-v2-unfiltered/records.jsonl`:

- 3,999 English provider-neutral records;
- eight behavior families and five speech styles;
- 3,498 structured calls across `search`, `calculate`, and `get_time`;
- sequential tool rounds represented as assistant call → linked result → later assistant call;
- canonical source SHA-256
  `9d984f1be6afc4b5b8052a5d52b35ff3f267aacb14df19b520b74ad328870bd0`.

The teacher was the official FP8 `Qwen/Qwen3.6-27B-FP8` checkpoint at revision
`e89b16ebf1988b3d6befa7de50abc2d76f26eb09`. The full source run took 2h49m02s and produced
3,999 of 4,000 planned records.

For training, two source records were held out from every behavior-family/speech-style cell,
creating a balanced 80-record validation set. The remaining 3,919 records were each included once
per logical pass. Within a pass, intact short segments of one speech style were reordered and
concatenated into 8–16-user-turn histories, preferentially mixing behavior buckets.

| Prepared split | Source records | Composed histories | Rendered tokens | Assistant targets |
| --- | ---: | ---: | ---: | ---: |
| Training, 16 passes | 3,919 | 14,391 | 17,842,761 | 6,856,508 |
| Validation, one fixed pass | 80 | 18 | 22,635 | 8,510 |

Every training source record appears in every pass, for 62,704 segment appearances. No source
record was omitted. No rendered history exceeded 2,068 tokens, so the 4,096-token limit caused no
truncation.

## Training configuration

| Property | Value |
| --- | --- |
| Base | `Qwen/Qwen3-1.7B` |
| Revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Precision | BF16 base and adapter computation |
| Adapter | LoRA rank 16, alpha 32, dropout 0.05 |
| Target modules | All attention and MLP projections |
| Trainable parameters | 17,432,576 |
| Total parameters | 1,738,007,552 |
| Micro-batch | 2 histories |
| Gradient accumulation | 8 |
| Effective batch | 16 histories |
| Optimizer | Fused AdamW |
| Learning rate | `1e-4` |
| Schedule | 5% warmup, then cosine decay |
| Logical passes | 16 |
| Optimizer steps | 904 |
| Random seed | 20260729 |

Assistant-only loss includes audible bridges, tool-call protocol tokens, final answers, and later
assistant turns. It excludes system, runtime tool definitions, user input, tool results, and
padding.

## Hardware and runtime

| Property | Measured value |
| --- | ---: |
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA-visible VRAM | 50,864,390,144 bytes |
| Peak allocated VRAM | 33,227,232,768 bytes |
| Peak reserved VRAM | 49,826,234,368 bytes |
| PyTorch | 2.10.0+cu129 |
| CUDA | 12.9 |
| Transformers | 4.57.6 |
| PEFT | 0.18.1 |
| Wall time | 2,821.03 seconds |
| Optimizer steps | 904 |

The measured training time was 47m01s. At the approximately USD 0.61/hour marketplace listing,
that corresponds to roughly USD 0.48 of GPU rental, excluding storage and bandwidth. The complete
base-versus-step-200 and base-versus-final behavioral evaluation took approximately nine minutes,
or roughly another USD 0.09 at the same listing rate.

## Loss curve and apparent overfitting

The base validation loss was measured before the first update using the zero-initialized LoRA
adapter. It is therefore directly comparable with later assistant-token validation losses.

| Completed pass | Optimizer step | Validation loss |
| ---: | ---: | ---: |
| Base | 0 | 4.5398 |
| 1 | 57 | 0.9630 |
| 2 | 114 | 0.8140 |
| 3 | 170 | 0.7733 |
| 4 | 227 | **0.7631** |
| 5 | 283 | 0.7715 |
| 6 | 339 | 0.7872 |
| 7 | 395 | 0.8170 |
| 8 | 451 | 0.8602 |
| 9 | 507 | 0.9043 |
| 10 | 564 | 0.9713 |
| 11 | 620 | 1.0168 |
| 12 | 676 | 1.0754 |
| 13 | 733 | 1.1182 |
| 14 | 790 | 1.1515 |
| 15 | 847 | 1.1659 |
| 16 | 904 | 1.1660 |

Final short-window training loss was 0.1852. The widening gap after pass 4 is clear evidence of
overfitting under the teacher-forced cross-entropy objective. The fixed validation histories are
small—18 composed histories containing 80 balanced source segments—and exact-token loss penalizes
valid paraphrases, different bridge wording, and alternate but correct query strings.

The loss curve must therefore be reported, not dismissed. It shows that 16 passes are excessive if
the objective is exact held-out target reproduction. It does not, by itself, prove that downstream
tool behavior became worse. Free-generation evaluation was added specifically to test that.

## Behavioral evaluation

The evaluator uses the exact production generation shape:

- `enable_thinking=False`;
- `max_new_tokens=256`;
- sampling enabled at temperature 0.6 and top-p 0.9;
- seeds 17, 29, and 43;
- the production Hermes streaming parser and typed tool schemas.

The 80 held-out records yield 160 generation points:

- 70 expected tool calls;
- 30 hard no-tool responses;
- 60 post-tool continuations.

Across three seeds, each model variant therefore produces 480 predictions: 210 tool-call
predictions, 90 no-tool predictions, and 180 continuations.

| Metric | Base | Step 200 | Final |
| --- | ---: | ---: | ---: |
| Parser-valid output | 100% | 100% | 100% |
| Correct tool decision | 80.0% | 84.8% | **95.2%** |
| Exact tool name | 78.6% | 84.8% | **94.3%** |
| Valid tool argument schema | 80.0% | 84.8% | **95.2%** |
| Spoken bridge present | 29.0% | 100% | **100%** |
| Spoken bridge 1–12 words | 7.6% | 91.9% | **94.3%** |
| Correct no-tool behavior | 70.0% | 100% | **100%** |
| Nonempty valid post-tool continuation | 98.3% | 100% | **100%** |

The final adapter's exact counts were:

- 200/210 expected tool calls emitted;
- 198/210 exact tool names;
- 200/210 schema-valid calls;
- 198/210 bridges within the stricter 1–12-word evaluation bound;
- 90/90 no-tool responses without a call;
- 180/180 nonempty post-result continuations without an extra call.

Across all 480 final-adapter predictions, spoken output averaged 12 words and had a maximum of 70
words, within the project's 100-word voice-turn ceiling.

## Why the final adapter is the current candidate

Step 200 is close to the best validation-loss point and was retained as the early comparison.
Nevertheless, the final adapter is materially stronger on every measured free-generation behavior:

- it adds 22 correct tool decisions across 210 predictions compared with step 200;
- it adds 20 exact tool names;
- it improves concise bridges while keeping bridge presence at 100%;
- it preserves perfect no-tool and continuation results.

This apparent conflict is plausible because the two evaluations measure different things.
Teacher-forced loss asks whether the model predicts the exact synthetic target token sequence.
Behavioral evaluation asks whether a sampled response takes the right action with valid structure
and natural speech. Later passes appear to improve the coarse behavioral convention while becoming
more confident in a narrower synthetic wording distribution.

The evidence supports retaining the final adapter as the integration candidate and the step-200
adapter as the diagnostic early checkpoint. It does not support claiming that later training
improved general language quality, factuality, or real-web-search grounding.

## Remaining errors

The final adapter still misses 10 of 210 expected calls and chooses the wrong tool for two more.
The misses cluster in four correction or refined-search histories. Typical failures are:

- answering from an earlier search result when the user changed the requested entity;
- saying that the information is unavailable instead of issuing a fresh search;
- asking “Would you like me to look that up?” despite an already explicit user request;
- confusing `search` and `calculate` in one synthetic department-budget chain.

The model has therefore learned the desired behavior strongly, but not perfectly. A future
targeted run should add hard examples for automatic re-search after correction and for preserving
tool identity across sequential chains. It should use fewer passes or early behavioral evaluation,
not simply repeat the same 16-pass schedule.

The synthetic search oracle is another limitation. Search results were generated by the teacher,
not retrieved from the web. The experiment demonstrates use of a supplied result, not factual
search quality, citation correctness, prompt-injection resistance, or live-search latency.

## Artifact retention

The stable local backup is:

```text
data/tool-use/training-artifacts/qwen3-1.7b-production-v2-poc-v2/
```

The reduced local backup retains:

- final adapter and tokenizer files;
- step-200 adapter and tokenizer files;
- exact training config and corpus manifest;
- complete step-level metrics and TensorBoard events;
- training summary and hardware measurements;
- step-200 and final behavioral reports plus all predictions;
- original full-run and reduced-backup hash manifests.

It intentionally removes:

- checkpoints 400, 600, and 800;
- materialized training and validation conversation JSONL;
- deterministic composition-plan JSONL;
- the duplicate top-level tokenizer copy.

Those removed files are reproducible from the canonical source SHA, training code, pinned
tokenizer, seed, and config. The canonical source corpus and raw teacher request journal are stored
separately under `data/tool-use/production-v2-unfiltered/`.

Model weights, generated datasets, and raw request journals remain outside Git. This repository
commits the scripts, small reproducibility manifest, runbooks, and measured report.

| Retained item | Size | SHA-256 |
| --- | ---: | --- |
| Final adapter weights | 69,782,384 bytes | `f5727303ca8637176ce7dde08561096ec34023f76f29e019a74199ef78522b94` |
| Step-200 adapter weights | 69,782,384 bytes | `3cc3a03f73c301806f405dc19f62f264cc4c019a375be5a055d9d1e3705c74f0` |
| Complete metrics JSONL | 192,112 bytes | `76b72d8bcc8bb61aa433c023908a69044fa481d7db7f8fde4c3bf088debc285b` |
| TensorBoard event file | 93,096 bytes | `75520027151d707f21cc1966d7705bc7e33d69d47f4b2e51ff5b1d8929d482c6` |

The reduced bundle is 172,781,150 bytes across 31 files. Its manifest validates all 30 other files;
the manifest itself has SHA-256
`d589cfab621e5518abd1633cd6cf8bb6c1c397eae4184eb214e9d2d6e98e8f81`.

## Appropriate project summary

A precise description of the work is:

> Designed a provider-neutral synthetic conversation schema and a staged vLLM generation pipeline
> for natural spoken tool use. Generated 3,999 English multi-turn records with search, calculation,
> time, corrections, no-tool dialogue, and sequential calls using a self-hosted 27B FP8 teacher.
> Built a deterministic native-Qwen renderer with assistant-only loss and bucket-aware history
> composition, then trained a 17.4M-parameter BF16 LoRA adapter for Qwen3-1.7B on 17.8M rendered
> tokens. Offline evaluation across 480 seeded predictions per model improved correct tool
> decisions from 80.0% to 95.2%, concise spoken bridges from 7.6% to 94.3%, and no-tool accuracy
> from 70.0% to 100%, while also identifying teacher-forced overfitting and residual correction
> errors.

That wording distinguishes measured results from deployment claims and from capabilities the
synthetic-search setup did not test.
