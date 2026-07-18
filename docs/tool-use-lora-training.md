# Tool-use LoRA training

The pilot trains a BF16 LoRA adapter on the pinned `Qwen/Qwen3-1.7B` checkpoint. It does not
quantize or merge the base model.

Run the deterministic corpus preflight without loading the model:

```text
python -m app.training.tool_use.training_cli \
  data/tool-use/production-v2-unfiltered/records.jsonl \
  artifacts/tool-use-lora/pilot-01 \
  --prepare-only
```

Remove the preflight output directory, then omit `--prepare-only` to train. The command refuses to
write into a non-empty directory so a retry cannot silently mix artifacts from different runs.

The pilot configuration is defined in `training_config.py`:

- exact Qwen checkpoint revision;
- BF16 base weights with rank-16, alpha-32 LoRA on all attention and MLP projections;
- two logical source-data passes;
- 8–16 user turns per composed history;
- effective batch size 16;
- assistant-only loss;
- 4,096-token hard limit with no truncation;
- cosine schedule at `1e-4` with 5% warmup.

For every logical pass, source segments are shuffled and regrouped. A composition keeps each source
segment intact, uses one speech style and split, and preferentially draws each next segment from a
new behavior bucket. The second logical pass changes both grouping and order. Validation uses one
fixed composition pass and test records are never loaded into the optimizer.

The output contains the exact config, source-data hash, composition plans, rendered-corpus
statistics, adapter checkpoints, per-step metrics, tokenizer files, GPU/VRAM measurements, Git
commit, and SHA-256 hashes. Canonical source records and model weights remain outside Git.
