# Synthetic Tool-Use Generation Runbook

This runbook prepares English natural voice-tool conversations for the pinned Voice Light
Qwen3-1.7B target. It does not train or deploy a model.

## Architecture

The generator runs many independent conversation state machines concurrently against a vLLM
OpenAI-compatible endpoint. Each model request returns one JSON-schema-constrained object:

1. Generate a spoken user turn from a deterministic scenario.
2. Generate either final assistant speech or short speech plus one tool call.
3. On a call, append the canonical assistant call and its result.
4. Ask the model for the next assistant step using the expanded public history.
5. Repeat for approved sequential calls and later user turns.

`calculate` uses a restricted local arithmetic parser. `get_time` returns a deterministic seeded
timestamp. `search` makes another request to the same Qwen teacher and returns its concise answer as
the synthetic tool result. No live web search occurs.

The canonical JSONL includes system text, runtime tool definitions, audible assistant text,
structured calls, stable call IDs, typed results, later turns, provenance, split assignment, and
audit metadata. Tool syntax is never placed in audible text.

## GPU checkpoint

Do not load this teacher beside deployed Voice Light services. Use only the separately approved
rental.

The official `Qwen/Qwen3.6-27B-FP8` repository is approximately 30.9 GB before runtime overhead.
The selected Vast listings report 48 GB VRAM. Confirm at startup with `nvidia-smi` that the
container actually exposes at least 40 GB usable CUDA memory. If it does not, stop; do not silently
substitute a different checkpoint or quantization.

Pin the exact model revision in both the server command and generator manifest. The placeholder
below must be replaced with that inspected revision:

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --revision MODEL_REVISION \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key VOICE_LIGHT_LOCAL \
  --language-model-only \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --generation-config vllm
```

Use vLLM 0.19 or newer for Qwen3.6. Structured response generation does not require native
auto-tool-choice or a tool-call parser because the generator requests its own typed semantic
objects. The client defaults to Qwen's documented non-thinking sampling settings: temperature 0.7,
top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, and repetition penalty 1.

The 27B FP8 checkpoint fits a reported 48 GB RTX 4090, but vLLM's CUDA-graph profiler can exceed
the remaining headroom. The measured serving smoke test therefore used eager execution and an 85%
memory allocation, leaving about 11.5 GiB for the KV cache without a startup OOM.

## Prepare scenarios

From the repository root:

```bash
python -m app.training.tool_use.cli plan \
  data/tool-use/scenarios-4000.jsonl \
  --count 4000 \
  --seed 20260718
```

The scenario plan is deterministic and safe to create without a GPU. `/data/` is ignored by Git.
The natural-v2 planner makes direct questions a minority, samples requests, context-first turns,
fragments, reactions, and self-repairs, and delays the first tool need in about 43% of multi-turn
tool conversations. Adverse outcomes are tool-specific and rare; multi-call turns target a 0.5%
adverse rate rather than treating failure as routine.

## Twenty-record serving smoke test

Run this before the manually reviewed calibration:

```bash
python -m app.training.tool_use.cli generate \
  data/tool-use/scenarios-4000.jsonl \
  data/tool-use/records.jsonl \
  --failures data/tool-use/failures.jsonl \
  --manifest data/tool-use/run-manifests.jsonl \
  --request-log data/tool-use/vllm-requests.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key VOICE_LIGHT_LOCAL \
  --model Qwen/Qwen3.6-27B-FP8 \
  --model-revision MODEL_REVISION \
  --quantization fp8 \
  --time-reference 2026-07-18T12:00:00+00:00 \
  --limit 20 \
  --concurrency 64
```

The records, failures, run manifests, and raw vLLM request/response journal are append-only. The
journal never records the API-key header. Re-running the command skips completed `scenario_id`
values; failed scenarios may be retried on a later invocation.

Choose one timezone-aware `--time-reference` at the beginning of the dataset run and keep it fixed
for every checkpoint and retry. Each successful `get_time` result is deterministically sampled from
the 365 days preceding that anchor, with a varied UTC offset. The anchor is retained in both record
provenance and run manifests.

Validate and summarize:

```bash
python -m app.training.tool_use.cli validate data/tool-use/records.jsonl
python -m app.training.tool_use.cli summarize data/tool-use/records.jsonl
```

Inspect all 20 conversations and record:

- actual GPU model and usable VRAM;
- exact model and vLLM revisions;
- peak VRAM and any out-of-memory event;
- accepted records per hour;
- prompt and completion throughput;
- requests and retries per accepted record;
- tool-plan adherence;
- bridge naturalness and phrase repetition;
- result-grounding and failure recovery;
- malformed or leaked protocol text.

## One-hundred-record calibration checkpoint

Only after the serving smoke test passes, rerun with `--limit 100`. Review all 100 records manually.
Do not continue to 2,500 or begin training until the user has reviewed the samples and measured
statistics.

Keep generated JSONL and model artifacts out of Git. Commit only schemas, scripts, small fixtures,
configuration, manifests without secrets, and reproducibility documentation.
