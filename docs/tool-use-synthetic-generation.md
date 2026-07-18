# Synthetic Tool-Use Dataset

This document records the completed English synthetic-data run used for the natural spoken
tool-use LoRA proof of concept. The generator and canonical schema remain reusable; the measured
run described here is `production-v2-unfiltered`.

## Outcome

The final run planned 4,000 source segments and produced 3,999 valid canonical records. One scenario
ended in an HTTP read timeout and was not retried because the user accepted the 3,999-record corpus.
Generation took 2 hours, 49 minutes, and 2 seconds on the separately rented RTX 4090 node.

| Property | Measured value |
| --- | ---: |
| Teacher | `Qwen/Qwen3.6-27B-FP8` |
| Teacher revision | `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` |
| Quantization | Official FP8 checkpoint |
| Prompt revision | `constructive-advice-v19` |
| Generator source commit | `0b807fc` |
| Planned scenarios | 4,000 |
| Accepted records | 3,999 |
| Terminal failures | 1 `ReadTimeout` |
| Structurally rejected candidates retried | 3 |
| Teacher requests | 28,643 |
| Prompt tokens | 32,196,968 |
| Completion tokens | 1,474,233 |
| Elapsed time | 10,142 seconds |
| Accepted records/hour | 1,419 |
| Requests/accepted record | 7.16 |

The marketplace listing was approximately USD 0.61/hour at the time. That implies approximately
USD 1.72 of GPU rental for the measured generation window, excluding storage and bandwidth. This
is a retrospective estimate from the listing price, not a provider invoice.

## What was generated

The source dataset is deliberately balanced across eight behavior families and five speech styles.
The missing sequential-tools record is the single timed-out scenario.

| Behavior family | Records |
| --- | ---: |
| Natural conversation | 500 |
| Drafting | 500 |
| Brainstorming/decision support | 500 |
| Search | 500 |
| Calculate | 500 |
| Get time | 500 |
| Sequential tools | 499 |
| Correction/recovery | 500 |

Each speech style—ASR fragment, casual, clean, formal, and self-repair—has 800 records except
`clean`, which has 799. The corpus contains:

- 11,810 user messages;
- 15,308 assistant messages;
- 3,498 structured tool calls;
- 1,741 `search`, 999 `calculate`, and 758 `get_time` calls;
- 3,248 successful, 84 failed, 83 empty, and 83 timed-out tool results;
- 2,718 medium and 1,281 long source segments.

The source records are short segments. The training loader creates longer histories by concatenating
intact segments; those materialized histories are derived data rather than a second canonical
dataset.

## Architecture

The generator runs many independent conversation state machines concurrently against a local vLLM
OpenAI-compatible endpoint. Each request returns a JSON-schema-constrained semantic object:

1. Generate a spoken user turn from a deterministic scenario.
2. Generate either assistant speech or speech plus one structured tool call.
3. Stop the conversational rollout when a tool call is produced.
4. Execute or synthesize the tool result and append both the assistant call and linked result.
5. Ask the teacher for the next assistant step using the expanded history.
6. Repeat for sequential calls and later user turns.

The tools deliberately have minimal runtime schemas:

```text
search(query: string) -> string
calculate(expression: string) -> string
get_time() -> string
```

`calculate` uses a restricted deterministic arithmetic parser. `get_time` deterministically samples
a timezone-aware timestamp from the 365 days before the fixed run anchor, preventing the model from
learning one constant timestamp. `search` makes another request to the same Qwen teacher and uses
its concise answer as a synthetic result. It does not perform live web retrieval, so the corpus
teaches causal result use and grounding to supplied text, not factual search accuracy.

The canonical JSONL stores system content, runtime tool definitions, audible assistant text,
structured calls, stable call IDs, typed outcomes, later turns, provenance, split metadata, and
audit hashes. It never uses token IDs or a pre-rendered Qwen chat string as its source of truth, and
tool protocol markup is never stored as audible content.

The implementation is divided by responsibility:

- `scenario.py`: behavior buckets, speech styles, weighted axes, and deterministic plans;
- `protocol.py`: typed teacher structured-output boundaries;
- `prompts.py`: user, assistant, search-oracle, quality, and grounding prompts;
- `generation.py`: staged causal rollout and tool-result injection;
- `tools.py`: deterministic calculator and seeded time;
- `vllm_client.py`: concurrent JSON-schema-constrained teacher requests and request journal;
- `schema.py`: frozen provider-neutral canonical records;
- `cli.py`: planning, generation, validation, statistics, review selection, and composition;
- `viewer/`: local browser review of canonical conversations.

## Serving and generation commands

The teacher was isolated from deployed Voice Light services. The 27B FP8 model required the
reported 48 GiB 4090 variant; eager vLLM execution avoided CUDA-graph startup pressure.

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key VOICE_LIGHT_LOCAL \
  --language-model-only \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --generation-config vllm
```

The equivalent deterministic scenario command is:

```bash
python -m app.training.tool_use.cli plan \
  data/tool-use/production-v2-unfiltered/scenarios-4000.jsonl \
  --count 4000 \
  --seed 20260729 \
  --profile bucket_calibration
```

The measured run used vLLM structured output, 128 concurrent conversation state machines, a fixed
time anchor, and semantic audits disabled:

```bash
python -m app.training.tool_use.cli generate \
  data/tool-use/production-v2-unfiltered/scenarios-4000.jsonl \
  data/tool-use/production-v2-unfiltered/records.jsonl \
  --failures data/tool-use/production-v2-unfiltered/failures.jsonl \
  --manifest data/tool-use/production-v2-unfiltered/run-manifests.jsonl \
  --request-log data/tool-use/production-v2-unfiltered/vllm-requests.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key VOICE_LIGHT_LOCAL \
  --model Qwen/Qwen3.6-27B-FP8 \
  --model-revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09 \
  --quantization fp8 \
  --time-reference 2026-07-18T12:00:00+00:00 \
  --concurrency 128
```

“Unfiltered” means the teacher-based taste, naturalness, and grounding rejection passes were
disabled after manual review showed that their rejection decisions were counterproductive.
Typed-schema validation, call/result ordering, deterministic tool checks, maximum spoken-turn
length, and protocol-leak checks remained active. The three rejected candidates in the manifest
were structural generation attempts that were retried, not subjective quality filtering.

## Calibration history

The generator was iterated through small review runs before scale-up. Important changes included:

- more natural user utterance forms and less question-only dialogue;
- assistant responses capped at 100 spoken words;
- longer conversations assembled from coherent short segments;
- eight balanced behavior buckets and five speech styles;
- automatic search after a correction instead of repeatedly asking permission;
- constructive advice rather than blindly endorsing unhealthy or counterproductive behavior;
- rare rather than routine tool failures;
- semantic rejection disabled after reviewing its false positives.

The relevant implementation history runs from `2cfb6cb` through `0b807fc`. The final production
prompt is identified independently in every record as `constructive-advice-v19`.

## Retained source artifacts

Generated datasets remain outside Git. The complete local source bundle is:

```text
data/tool-use/production-v2-unfiltered/
  scenarios-4000.jsonl
  records.jsonl
  failures.jsonl
  run-manifests.jsonl
  vllm-requests.jsonl
  generation.log
  SHA256SUMS
```

The canonical records SHA-256 is
`9d984f1be6afc4b5b8052a5d52b35ff3f267aacb14df19b520b74ad328870bd0`.
The scenario-plan SHA-256 is
`84951e1b8d7b54513d5ef741b6921c1172dfef998a6d290a0f562ecb77c81d70`.
The raw request journal is useful for audit and prompt reconstruction but is not required by the
training loader.

Validate any retained copy with:

```bash
sha256sum --check data/tool-use/production-v2-unfiltered/SHA256SUMS
python -m app.training.tool_use.cli validate \
  data/tool-use/production-v2-unfiltered/records.jsonl
python -m app.training.tool_use.cli summarize \
  data/tool-use/production-v2-unfiltered/records.jsonl
```

See [the LoRA runbook](tool-use-lora-training.md) for deterministic composition and training, and
[the experiment report](tool-use-finetuning-results.md) for measured model behavior.
