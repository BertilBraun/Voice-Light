# Tool-Use LoRA Training

This runbook reproduces the completed BF16 LoRA proof of concept for the pinned
`Qwen/Qwen3-1.7B` checkpoint. It trains a PEFT adapter and does not quantize, merge, or deploy the
base model.

The measured results are recorded separately in
[Natural Spoken Tool-Use Fine-Tuning Results](tool-use-finetuning-results.md).

## Source and model pins

| Input | Pinned value |
| --- | --- |
| Canonical records | `data/tool-use/production-v2-unfiltered/records.jsonl` |
| Canonical records SHA-256 | `9d984f1be6afc4b5b8052a5d52b35ff3f267aacb14df19b520b74ad328870bd0` |
| Training implementation commit | `bf1e0b4` |
| Evaluation implementation commit | `83bbe00` |
| Base model | `Qwen/Qwen3-1.7B` |
| Base revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Precision | BF16 |

The relevant implementation is:

- `training_config.py`: exact model revision and all hyperparameters;
- `training_data.py`: balanced holdout assignment, per-pass composition, and manifests;
- `composition.py`: intact segment concatenation and bucket-aware ordering;
- `renderer.py`: native Qwen chat-template rendering and assistant-only masks;
- `training.py`: LoRA construction, optimization, checkpoints, validation, TensorBoard, and hashes;
- `training_cli.py`: clean-output command-line entry point;
- `evaluation.py` and `evaluation_cli.py`: seeded base-versus-adapter behavioral evaluation.

## Deterministic split and composition

The original synthetic split labels are replaced for this proof of concept. Exactly 80 of the 3,999
source records are selected for validation: two records from every combination of eight behavior
buckets and five speech styles. The remaining 3,919 source records are training data. There is no
separate test split in this proof of concept.

The training loader does not mutate a source segment. For each logical pass it:

1. groups records by speech style;
2. shuffles them with a deterministic pass-specific seed;
3. concatenates intact segments into histories containing 8–16 user turns;
4. prefers new behavior buckets, then a bucket different from the previous segment;
5. retries the grouping seed until every source record appears exactly once in that pass;
6. renders the result with the pinned tokenizer's native Hermes-style Qwen chat template.

All 3,919 training records therefore appear exactly once in each of 16 independently regrouped
passes: 62,704 segment appearances in total. This becomes 14,391 composed training histories. The
80 held-out records are composed once into 18 validation histories.

The canonical records, seeds, and implementation fully determine the materialized
`*-composition-plans.jsonl` and `*-composed-records.jsonl` files. Those large files are convenient
run artifacts but are not required in the long-term backup.

## Loss mask

The renderer calls the official pinned tokenizer's
`apply_chat_template(..., tools=..., enable_thinking=False)` behavior. Canonical semantic records
remain independent of the rendered Qwen protocol.

Labels retain every assistant target token:

- ordinary assistant speech;
- the audible bridge before a call;
- the structured tool-call tokens;
- post-result assistant continuation;
- later assistant turns.

System instructions, runtime tool definitions, user messages, tool-result inputs, padding, and
non-assistant protocol tokens are masked with `IGNORE_INDEX`. Rendering tests cover no-tool,
one-call, failures, multi-turn histories, sequential calls, and exact mask boundaries.

## Measured configuration

```json
{
  "random_seed": 20260729,
  "holdout_record_count": 80,
  "logical_epochs": 16,
  "minimum_user_turns": 8,
  "maximum_user_turns": 16,
  "maximum_sequence_tokens": 4096,
  "per_device_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "effective_batch_size": 16,
  "learning_rate": 0.0001,
  "warmup_ratio": 0.05,
  "maximum_gradient_norm": 1.0,
  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "save_steps": 200,
  "gradient_checkpointing": false
}
```

LoRA targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
The optimizer is fused AdamW with a cosine schedule and 5% warmup. The base model stays frozen.
The adapter has 17,432,576 trainable parameters out of 1,738,007,552 total parameters.

## Preflight

Run the deterministic corpus preflight without loading the model:

```bash
python -m app.training.tool_use.training_cli \
  data/tool-use/production-v2-unfiltered/records.jsonl \
  artifacts/tool-use-lora/qwen3-1.7b-production-v2-poc-v2-preflight \
  --source-commit bf1e0b4 \
  --prepare-only
```

Verify that the generated corpus manifest reports:

- 3,919 training and 80 validation source records;
- 14,391 training and 18 validation compositions;
- no omitted source IDs;
- 17,842,761 rendered training tokens;
- 6,856,508 assistant target tokens;
- maximum composed length 2,068 tokens, below the 4,096-token hard limit.

Remove only the verified preflight directory, then start the real run in a new output directory.
The command refuses to write into a non-empty directory, preventing accidental mixed runs.

## Training

```bash
python -m app.training.tool_use.cli validate \
  data/tool-use/production-v2-unfiltered/records.jsonl

python -u -m app.training.tool_use.training_cli \
  data/tool-use/production-v2-unfiltered/records.jsonl \
  artifacts/tool-use-lora/qwen3-1.7b-production-v2-poc-v2 \
  --source-commit bf1e0b4
```

Training must run in an exclusive approved GPU window. Do not load this model beside deployed
Voice Light services. The completed run used one 48 GiB RTX 4090 in BF16 and required 904 optimizer
steps.

Monitor:

```bash
tail -f artifacts/tool-use-lora/qwen3-1.7b-production-v2-poc-v2/metrics.jsonl
tensorboard \
  --logdir artifacts/tool-use-lora/qwen3-1.7b-production-v2-poc-v2/tensorboard
```

`metrics.jsonl` is append-only and records the pre-training baseline, every optimizer step, and one
validation measurement after every logical pass. TensorBoard contains `loss/train`,
`loss/validation`, `learning_rate`, and the serialized run configuration.

## Behavioral evaluation

Validation loss alone did not select the best behavior in this experiment. Evaluate saved
checkpoints with actual generation at production settings:

```bash
python -u -m app.training.tool_use.evaluation_cli \
  data/tool-use/production-v2-unfiltered/records.jsonl \
  artifacts/tool-use-lora/qwen3-1.7b-production-v2-poc-v2/adapter \
  artifacts/tool-use-lora/qwen3-1.7b-production-v2-poc-v2/evaluation-final \
  --batch-size 8
```

The evaluator uses `enable_thinking=False`, `max_new_tokens=256`, sampling at temperature 0.6 and
top-p 0.9, and seeds 17, 29, and 43. It loads the base model once, evaluates it, then attaches the
candidate adapter to the same model. It never loads a second base model concurrently.

The held-out evaluation points are:

- every expected assistant tool call;
- every immediate assistant continuation after a tool result;
- the first assistant response in every no-tool record.

Metrics cover parser validity, tool decision, exact name, argument schema, bridge presence and
length, no-tool behavior, and post-result continuation.

## Retention policy

Keep:

- the canonical 3,999-record source corpus and its hash manifest;
- exact training config and corpus manifest;
- complete `metrics.jsonl` and TensorBoard events;
- `training-summary.json`;
- final adapter and the step-200 comparison checkpoint;
- base, step-200, and final behavioral reports and predictions;
- a new SHA-256 manifest for the reduced backup;
- the Git commits containing generation, rendering, training, and evaluation code.

Do not retain the 132 MB materialized training conversations, composition-plan files, duplicate
top-level tokenizer copy, or checkpoints 400/600/800 in the reduced backup. They are deterministic
derivatives or empirically dominated checkpoints.

Generated corpora and model weights remain outside Git. Commit small manifests, measured reports,
documentation, and source code only.
