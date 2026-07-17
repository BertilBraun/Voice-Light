# Natural Spoken Tool Use: Dataset and Fine-Tuning Plan

Status: Phase 1 design completed; local Phase 2 generation tooling approved and in preparation

Date: 2026-07-17
Target currently deployed by Voice Light: `Qwen/Qwen3-1.7B` at revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

No dataset generation, training, model loading, service change, artifact merge, or deployment was
performed for this phase.

## Executive recommendation

Run a deliberately staged pilot before broadening either the tools or training method:

1. Calibrate self-hosted `Qwen/Qwen3.6-27B` as the teacher in a separate rented GPU environment.
   Prefer Qwen's official FP8 checkpoint on at least 40 GB usable VRAM. The selected rental
   listings report 48 GB; verify the memory exposed inside the container before loading the model.
   Generate exactly 100 structured records only after the FP8 serving path is stable, and inspect
   every record before selecting it for the full pilot.
2. Target 4,000 English-only canonical conversations, generated in checkpoints rather than one
   uninterrupted run. Include no-tool, one-call, two-sequential-call, and a very small
   three-sequential-call slice. Treat 2,500 accepted records as the minimum useful intermediate
   checkpoint, not the final target.
3. Advertise only three deliberately simple tools: `search(query: string)`,
   `calculate(expression: string)`, and `get_time()` with no arguments. Each rendered tool result
   is one string. Canonical provenance and outcome metadata remain typed outside that rendered
   string.
4. Train a BF16 LoRA SFT adapter for the pinned Qwen3-1.7B checkpoint on a separate 24 GB GPU.
   Keep the adapter as the reproducible source artifact; build a merged BF16 candidate only after
   the adapter passes offline evaluation. Use QLoRA only if the approved training allocation is
   limited to roughly 12 GB. Do not use preference optimization in the pilot.
5. Generate records as staged causal rollouts. Each teacher request produces only the next
   assistant speech/action. The controller stops at a call, executes `calculate` deterministically,
   returns a seeded synthetic value for `get_time`, or asks the same Qwen teacher to answer the
   search query as a synthetic-search oracle. It appends the result and then asks the teacher for
   the next assistant step. The assistant step sees the search result only after its call.
6. Gate scale-up on held-out tool choice, call validity, bridge quality, grounding, failure
   recovery, multi-turn behavior, and ordinary no-tool regressions. Evaluate with the exact
   production generation configuration and multiple seeds.
7. Treat private model context, durable audible history, and the execution/audit journal as three
   different stores. Runtime persistence work belongs after training evidence, not in the data
   pilot.

This is a recommendation for the six user decisions at the end of this document, not a decision
already taken.

## Evidence from the current repository

The current worktree is detached at clean `master` commit `255cb7d`. It does not contain unrelated
changes from the other checkout.

The working runtime already establishes the correct first-round causal order:

```text
audible bridge
  -> private structured call completes
  -> typed registry validation
  -> asynchronous handler
  -> typed result committed to generation-local model context
  -> tool-disabled second Qwen invocation
  -> audible continuation in the same TTS/playback generation
```

The relevant implementation is split across:

- `app/compute/voice/qwen_worker.py`: pinned tokenizer template, production sampling, and streaming
  Hermes parsing;
- `app/compute/voice/hermes_tool_parser.py`: incremental separation of audible text from private
  tool syntax, with a deliberate one-call limit;
- `app/compute/voice/llm_worker_protocol.py`: typed worker messages and events;
- `app/compute/voice/tools.py`: one deterministic weather schema, validation, execution, and typed
  outcomes;
- `app/compute/voice/session.py`: tool anchors, cancellation, timeout, continuation, TTS continuity,
  and durable playback acknowledgement;
- `app/compute/voice/conversation.py`: the separate audible and generation-local model message
  representations.

The tests already cover delimiter splits, malformed and duplicate calls, unknown tools, invalid
arguments, handler failure, timeout, cancellation-resistant late results, buffering cancellation,
one continuous TTS/playback turn, and exclusion of tool markup from speech and public history.

Recent repository history is also useful evidence. Prompt-only bridge reinforcement was added in
`fd5224a`; attempts to manufacture or enforce the bridge below the model were subsequently reverted
(`36f928d`/`8c22671` and `e12d02a`/`255cb7d`). That is consistent with the observed problem being a
learned output convention rather than a missing parser or orchestration feature.

### The current persistence gap

During a tool turn, `generation.model_messages` correctly holds:

```text
user
assistant(audible bridge + structured call)
tool(typed outcome linked by call ID)
```

After playback, however, later user turns rebuild model context from `self.conversation`, which
contains only audible user/assistant text. The structured call and result are therefore not
available to a later generation. The audible history is behaving correctly; it must not be expanded
to contain tool JSON. A separate private conversation context and an execution journal are needed
later.

## Protocol and training-stack findings

Qwen recommends Hermes-style tool use for Qwen3. Its official function-calling guide shows the
normal learned convention that assistant content is empty alongside calls, explains that tool
results are appended before another model response, and warns that malformed calls still require
application validation:
[Qwen function-calling guide](https://qwen.readthedocs.io/en/stable/framework/function_call.html).

The pinned Qwen3 model is Apache-2.0, has 1.7 billion parameters and a 32,768-token context window,
and requires Transformers 4.51 or newer:
[Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B).
Voice Light currently uses Transformers 5.13, so the model architecture and chat-template support
are compatible. The same model card recommends temperature 0.7/top-p 0.8/top-k 20 in non-thinking
mode. Production currently uses temperature 0.6/top-p 0.9 with no explicit top-k. Evaluation must
first preserve production settings; changing them would be a separate measured decision.

Transformers recommends passing structured messages to `apply_chat_template`, tokenizing through
the template directly to avoid duplicate special tokens, and supplying generation prompts only at
inference:
[Transformers chat-template documentation](https://huggingface.co/docs/transformers/chat_templating).

TRL supports conversational SFT, tool definitions, PEFT adapters, and assistant-only loss. Its
training-template utility specifically supports Qwen3 and can add non-emitting
`{% generation %}` markers while preserving the rendered prefix:
[TRL SFTTrainer](https://huggingface.co/docs/trl/main/sft_trainer) and
[TRL training chat-template utility](https://huggingface.co/docs/trl/chat_template_utils).
Phase 2 must prove that the patched training template produces exactly the same token IDs as the
pinned native template before trusting its assistant mask.

LoRA freezes base weights and learns small low-rank updates
([LoRA paper](https://arxiv.org/abs/2106.09685)). PEFT explicitly supports a QLoRA-style
`target_modules="all-linear"` configuration
([PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora)).
QLoRA uses a frozen 4-bit base with trainable LoRA weights and introduced NF4, double quantization,
and paged optimizers
([QLoRA paper](https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf);
[bitsandbytes 4-bit documentation](https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit)).

DPO is appropriate when there are real chosen/rejected preferences, not merely a desired structured
target ([DPO paper](https://arxiv.org/abs/2305.18290)). Current TRL supports tool-bearing preference
records ([TRL DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)). ORPO combines an SFT
signal with an odds-ratio penalty and avoids a reference model, but the current TRL implementation
is marked experimental
([ORPO paper](https://aclanthology.org/2024.emnlp-main.626.pdf);
[TRL trainer index](https://huggingface.co/docs/trl/index)). Neither is justified until SFT errors
show a stable preference problem such as repetitive bridges or style tradeoffs that corrected
demonstrations alone do not solve.

## Teacher/provider decision

### Important terms constraint

Using a proprietary frontier API to produce data for a Qwen fine-tune is not automatically
permitted merely because the customer owns the output:

- OpenAI's current services agreement restricts using output to develop competing AI models, with
  narrow listed exceptions that do not cover fine-tuning Qwen:
  [OpenAI Services Agreement](https://openai.com/policies/services-agreement/).
- Anthropic's commercial terms restrict access used to build a competing product, including
  training competing AI models, unless expressly approved:
  [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms).
- Gemini's current terms state that the services may not be used to develop models that compete
  with Gemini:
  [Gemini API Additional Terms](https://ai.google.dev/gemini-api/terms).

This document is not legal advice, but the safe engineering choice is to avoid those providers as
teachers unless the user's applicable agreement or written provider approval clearly authorizes the
use. Their privacy controls do not cure a use restriction.

### Options

| Teacher | License/terms posture | Expected quality | Privacy | Rough pilot generation cost |
| --- | --- | --- | --- | ---: |
| Self-hosted Qwen3.6-27B FP8 | Official Apache-2.0 checkpoint; output generated in our allocation | Recommended quality-preserving quantized candidate; Qwen reports nearly identical metrics to the original model | Prompts/results stay in the chosen GPU account and artifact store | Checkpoint is 30.9 GB before runtime/KV overhead; use at least 40 GB usable VRAM |
| Self-hosted Qwen3.6-27B GPTQ/AWQ 4-bit | Community quantization of the Apache-2.0 model | Cheapest 24 GB experiment; quality and Qwen3.6/vLLM compatibility must be measured, not assumed | Same as above | Approximately 14-18 GB weights, leaving limited but sufficient room for short contexts on a 24 GB 4090 |
| Self-hosted Qwen3.6-35B-A3B | Apache-2.0 model | Faster sparse alternative, but Qwen's published results generally trail the dense 27B model | Same as above | Lower active-compute cost, but BF16 weights still need a large allocation |
| Self-hosted Mistral Small 3.1 24B Instruct | Apache-2.0 model; output generated in our allocation | Strong JSON, function calling, natural dialogue; cross-family diversity | Prompts/results stay in the chosen GPU account and artifact store | 12-40 GPU-hours; about $5-$45 on a 24 GB marketplace GPU, excluding storage/engineering |
| Self-hosted gpt-oss-120b | Apache-2.0; 80 GB GPU | Stronger critic/structured-output challenger | Same as above | About $35-$120 for 12-40 hours at current 80 GB on-demand rates |
| Frontier commercial API | Provider-specific and potentially prohibited for this use without approval | Likely strongest schema adherence and critique | Provider receives prompts/outputs; retention and region depend on account/endpoint | Technically about $20-$60 for the proposed pilot, but not recommended without written terms clearance |
| Human-authored only | Clear ownership if contributors assign/license work | Highest control, slowest and least diverse | Local | Mostly labor rather than token/GPU cost |

The recommended Qwen teacher's model card describes Apache-2.0 licensing, vLLM compatibility, and
tool calling:
[Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B).
Qwen's official FP8 checkpoint is 30.9 GB and describes performance as nearly identical to the
original:
[Qwen3.6-27B-FP8 model card](https://huggingface.co/Qwen/Qwen3.6-27B-FP8).
The Mistral model card describes Apache-2.0 licensing, JSON output, native function calling, and
24 GB quantized deployment:
[Mistral Small 3.1 model card](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503).
The alternative model cards are
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and
[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b).
Current published on-demand examples range from a 24 GB RTX 4090 around $0.34-$1.15/hour to a
48 GB L40S around $0.99/hour and 80 GB GPUs around $1.39-$2.99/hour, but availability changes:
[RunPod pricing](https://www.runpod.io/pricing).

The revised generation estimate assumes 4,000 accepted records, roughly 6-14 million total
generated tokens across staged assistant steps, synthetic-search results, selective critiques, and
20-40% rejection/retry overhead. A planning range is 8-30 GPU hours on a 4090-class 4-bit
deployment because each conversation requires several causally ordered requests. These are
engineering estimates, not measurements. Continuous batching across many independent conversation
state machines should keep the GPU busy. The 20- and 100-record calibrations must report measured
accepted records/hour, generated tokens/second, number of requests per record, retry rate, schema
validity, semantic rejection rate, and projected total cost before the approved pilot continues.

Use vLLM's JSON-schema structured output for the complete semantic record, with thinking disabled.
This solves syntax, not judgment: vLLM explicitly documents that constrained tool calling
guarantees a parseable schema-conforming call, not a high-quality one
([vLLM tool-calling documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)).
The calibration, deterministic validators, and human review therefore remain required.

For comparison only, GPT-5.4 Batch currently lists $2.50 per million input tokens and $15 per
million output tokens and supports structured outputs
([GPT-5.4 model page](https://developers.openai.com/api/docs/models/gpt-5.4));
OpenAI Batch returns within 24 hours at a 50% discount
([Batch API FAQ](https://help.openai.com/en/articles/9197833-batch-api-faq)).
Those prices explain the technical API estimate but do not grant permission to train Qwen.

## Tool-surface decision

### Approved initial tool registry

The initial registry contains three small tools with uniform, minimal arguments:

```text
search(query: string[1..240]) -> string
calculate(expression: string[1..160]) -> string
get_time() -> string
```

The executor, not the model, chooses the search provider, result count, recency behavior, timeouts,
redirects, content limits, credentials, and summarizer. The string returned by `search` should be a
short evidence-grounded summary suitable for a 1.7B continuation model. URLs and source identity
may be embedded in the string when the scenario needs citations; the canonical audit metadata
also retains structured source provenance.

`calculate` accepts only numeric literals, parentheses, and an allowlisted set of arithmetic
operators. It is not Python. `get_time` has no arguments and returns the configured Voice Light
local time, including date and UTC offset, in one stable format. Questions about another
location's time should use `search` in this minimal milestone rather than complicating `get_time`.

External text is untrusted data and may contain prompt injection. The continuation model must be
instructed and evaluated to use evidence, ignore instructions in results, cite uncertainty, and
never turn retrieved text into expanded authority.

### Small explicit typed registry

A registry containing narrow functions is the selected direction. Advantages are:

- each call is schema-validatable and capability-limited;
- results and failures can be typed separately;
- permissions, timeouts, and audit records are tool-specific;
- examples can vary the advertised subset at runtime;
- the 1.7B model sees a small classification problem instead of an open-ended programming task.

Costs are additional prompt tokens and tool-choice confusion when definitions overlap. Keep the
advertised set to roughly one to three relevant tools for the pilot and hold out tool names and
definition combinations during evaluation.

### Arbitrary Python

Do not implement arbitrary Python as the first tool. It combines unrestricted expressiveness,
weakly bounded resource use, difficult argument validation, filesystem/network/secret exposure,
and a much harder code-generation task for a 1.7B model. A natural-language promise such as "only
do calculations" is not a security boundary.

OWASP recommends minimizing tool functionality and permissions and avoiding open-ended extensions
such as shell commands:
[OWASP Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/).
Its prompt-injection guidance also recommends tool-specific validation, least privilege, and
treating remote content as untrusted:
[OWASP Prompt Injection Prevention](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html).

If computation becomes necessary, add a separate capability-limited expression evaluator:
parsed numeric expressions only, an allowlisted AST, no imports, no attribute access, no
filesystem/network/process APIs, bounded input/output, and hard CPU/memory/time limits in an
isolated process. That is a new explicit tool, not "Python with a good prompt."

## Canonical provider-neutral schema

The source of truth is versioned semantic JSONL. It contains no token IDs, rendered chat string,
Hermes tags, or provider message quirks. All serialized-boundary models are frozen Pydantic models
with `extra="forbid"`. Closed sets use `StrEnum`; message, outcome, JSON value, and parameter-schema
variants are discriminated unions.

### Main types

```text
ToolDialogueRecord
  schema_version: literal "voice-light.tool-dialogue/v1"
  record_id: string
  system_instruction: string
  tools: tuple[ToolDefinition, ...]
  messages: tuple[UserMessage | AssistantMessage | ToolResultMessage, ...]
  metadata: RecordMetadata

ToolDefinition
  name: string
  description: string
  latency: immediate | latency_bearing
  parameters: ObjectParameterSchema
  result_description: string

ObjectParameterSchema
  properties: tuple[NamedParameter, ...]
  additional_properties: false

NamedParameter
  name: string
  required: bool
  schema: StringParameter | IntegerParameter | NumberParameter |
          BooleanParameter | ArrayParameter | ObjectParameterSchema

AssistantMessage
  kind: "assistant"
  message_id: string
  audible_text: string
  tool_calls: tuple[CanonicalToolCall, ...]

CanonicalToolCall
  call_id: string
  tool_name: string
  arguments: ObjectValue

ToolResultMessage
  kind: "tool_result"
  message_id: string
  call_id: string
  outcome: SuccessOutcome | FailureOutcome | TimeoutOutcome | CancelledOutcome
  citations: tuple[Citation, ...]

SemanticValue
  NullValue | BooleanValue | IntegerValue | NumberValue | StringValue |
  ArrayValue | ObjectValue

ObjectValue
  kind: "object"
  fields: tuple[NamedValue, ...]
```

Using ordered `NamedParameter` and `NamedValue` tuples avoids raw dictionary state inside the
application. The renderer alone converts these structures to the string-keyed JSON objects required
by a provider API.

`RecordMetadata` contains:

- `generation`: authored/synthetic/imported, teacher identifier and immutable revision, serving
  stack revision, prompt-template revision, seed, generation parameters, raw response SHA-256,
  critic identifiers, and parent record IDs;
- `scenario`: family, tool-need class, outcome class, dialogue length band, reference/correction
  axes, speech/ASR style, language tag, and parameter-source IDs;
- `split`: train/validation/test, split policy revision, leakage group ID, held-out family IDs, and
  held-out combination IDs;
- `audit`: creator, creation time, source/license identifiers and URLs, intended dataset license,
  privacy/PII review status, human-review status, reviewer IDs, and content hash.

### Cross-record invariants

- IDs are nonempty and unique within their scope.
- Every call name exists in the record's runtime tool definitions.
- Arguments validate against the canonical parameter schema.
- Every tool result references exactly one earlier unresolved call.
- A result cannot precede its call or resolve the same call twice.
- An assistant continuation after a result is a new assistant message.
- Sequential calls use separate assistant/result cycles; calls are never encoded as audible text.
- A latency-bearing call has a nonempty, one-sentence bridge of at most eight spoken words.
- A no-tool assistant message has no procedural bridge.
- Audible text contains no control tags, JSON envelope, call IDs, or provider tokens.
- A cancelled or timed-out call cannot later receive success in the same record.
- Metadata hashes and split assignments are stable and reproducible.

## Representative canonical JSON

The examples use the compact canonical value representation. Metadata is intentionally present even
in small fixtures; real generated records additionally carry immutable model and prompt revisions.

### No-tool contrast

```json
{
  "schema_version": "voice-light.tool-dialogue/v1",
  "record_id": "no-tool-001",
  "system_instruction": "Speak naturally. Use supplied tools only when needed.",
  "tools": [{
    "name": "search",
    "description": "Search for current or externally verifiable information.",
    "latency": "latency_bearing",
    "parameters": {
      "type": "object",
      "properties": [{
        "name": "query",
        "required": true,
        "schema": {"type": "string", "description": "Concise search query", "min_length": 1, "max_length": 240}
      }],
      "additional_properties": false
    },
    "result_description": "A concise grounded result string."
  }],
  "messages": [
    {"kind": "user", "message_id": "u1", "text": "what's seven times eight again", "speech_style": "asr_clean"},
    {"kind": "assistant", "message_id": "a1", "audible_text": "Seven times eight is fifty-six.", "tool_calls": []}
  ],
  "metadata": {
    "generation": {"method": "authored", "teacher": null, "prompt_revision": "manual-v1", "seed": null},
    "scenario": {"family": "hard_no_tool", "tool_need": "not_needed", "outcome": "not_applicable", "language": "en", "length_band": "short"},
    "split": {"name": "train", "policy_revision": "family-holdout-v1", "leakage_group_id": "arithmetic-basic"},
    "audit": {"source_license": "project-authored", "intended_license": "project-owned", "human_review": "approved", "content_sha256": "fixture-placeholder"}
  }
}
```

### One successful call

```json
{
  "schema_version": "voice-light.tool-dialogue/v1",
  "record_id": "one-call-001",
  "system_instruction": "Speak naturally. Use supplied tools only when needed.",
  "tools": [{
    "name": "search",
    "description": "Search for current or externally verifiable information.",
    "latency": "latency_bearing",
    "parameters": {
      "type": "object",
      "properties": [{
        "name": "query",
        "required": true,
        "schema": {"type": "string", "description": "Concise search query", "min_length": 1, "max_length": 240}
      }],
      "additional_properties": false
    },
    "result_description": "A concise grounded result string."
  }],
  "messages": [
    {"kind": "user", "message_id": "u1", "text": "who won the women's final yesterday", "speech_style": "asr_clean"},
    {
      "kind": "assistant",
      "message_id": "a1",
      "audible_text": "I'll check the latest result.",
      "tool_calls": [{
        "call_id": "call-1",
        "tool_name": "search",
        "arguments": {"kind": "object", "fields": [{
          "name": "query",
          "value": {"kind": "string", "value": "women's final result 2026-07-16"}
        }]}
      }]
    },
    {
      "kind": "tool_result",
      "message_id": "t1",
      "call_id": "call-1",
      "outcome": {
        "status": "success",
        "value": {"kind": "string", "value": "Mara Klein defeated Ana Silva 6-4, 7-5."}
      },
      "citations": [{
        "citation_id": "cite-1",
        "url": "https://example.test/final",
        "title": "Women's final result",
        "publisher": "Example Sports",
        "published_at": "2026-07-16T20:00:00Z",
        "retrieved_at": "2026-07-17T08:00:00Z"
      }]
    },
    {"kind": "assistant", "message_id": "a2", "audible_text": "Mara Klein won in straight sets, 6-4, 7-5.", "tool_calls": []}
  ],
  "metadata": {
    "generation": {"method": "synthetic", "teacher": "approved-teacher@revision", "prompt_revision": "realizer-v1", "seed": 17},
    "scenario": {"family": "current_result", "tool_need": "required", "outcome": "success", "language": "en", "length_band": "short"},
    "split": {"name": "train", "policy_revision": "family-holdout-v1", "leakage_group_id": "current-result-tennis"},
    "audit": {"source_license": "synthetic-no-external-text", "intended_license": "project-owned", "human_review": "pending", "content_sha256": "fixture-placeholder"}
  }
}
```

### Timeout/failure

```json
{
  "schema_version": "voice-light.tool-dialogue/v1",
  "record_id": "timeout-001",
  "system_instruction": "Speak naturally. Never invent a pending result.",
  "tools": [{
    "name": "search",
    "description": "Search for current information.",
    "latency": "latency_bearing",
    "parameters": {
      "type": "object",
      "properties": [{
        "name": "query",
        "required": true,
        "schema": {"type": "string", "description": "Search query", "min_length": 1, "max_length": 240}
      }],
      "additional_properties": false
    },
    "result_description": "A concise grounded result string."
  }],
  "messages": [
    {"kind": "user", "message_id": "u1", "text": "is the north line delayed right now", "speech_style": "asr_fragment"},
    {
      "kind": "assistant",
      "message_id": "a1",
      "audible_text": "Let me check the live status.",
      "tool_calls": [{
        "call_id": "call-1",
        "tool_name": "search",
        "arguments": {"kind": "object", "fields": [{
          "name": "query",
          "value": {"kind": "string", "value": "North Line live service status"}
        }]}
      }]
    },
    {
      "kind": "tool_result",
      "message_id": "t1",
      "call_id": "call-1",
      "outcome": {"status": "timeout", "message": "Search exceeded five seconds."},
      "citations": []
    },
    {"kind": "assistant", "message_id": "a2", "audible_text": "I couldn't reach the live status, so I can't confirm a delay.", "tool_calls": []}
  ],
  "metadata": {
    "generation": {"method": "authored", "teacher": null, "prompt_revision": "manual-v1", "seed": null},
    "scenario": {"family": "live_status_failure", "tool_need": "required", "outcome": "timeout", "language": "en", "length_band": "short"},
    "split": {"name": "validation", "policy_revision": "family-holdout-v1", "leakage_group_id": "transit-timeout"},
    "audit": {"source_license": "project-authored", "intended_license": "project-owned", "human_review": "approved", "content_sha256": "fixture-placeholder"}
  }
}
```

### Multi-turn follow-up and correction

```json
{
  "schema_version": "voice-light.tool-dialogue/v1",
  "record_id": "follow-up-001",
  "system_instruction": "Use prior structured results and resolve corrections naturally.",
  "tools": [{
    "name": "search",
    "description": "Search for current information.",
    "latency": "latency_bearing",
    "parameters": {
      "type": "object",
      "properties": [{
        "name": "query",
        "required": true,
        "schema": {"type": "string", "description": "Search query", "min_length": 1, "max_length": 240}
      }],
      "additional_properties": false
    },
    "result_description": "A concise grounded result string."
  }],
  "messages": [
    {"kind": "user", "message_id": "u1", "text": "when does the modern art museum close tonight", "speech_style": "asr_clean"},
    {
      "kind": "assistant",
      "message_id": "a1",
      "audible_text": "I'll check tonight's hours.",
      "tool_calls": [{
        "call_id": "call-1",
        "tool_name": "search",
        "arguments": {"kind": "object", "fields": [{
          "name": "query",
          "value": {"kind": "string", "value": "Modern Art Museum closing time Berlin 2026-07-17"}
        }]}
      }]
    },
    {
      "kind": "tool_result",
      "message_id": "t1",
      "call_id": "call-1",
      "outcome": {
        "status": "success",
        "value": {"kind": "string", "value": "The Berlin Modern Art Museum closes at 20:00 tonight."}
      },
      "citations": [{
        "citation_id": "cite-1",
        "url": "https://example.test/museum-hours",
        "title": "Visitor hours",
        "publisher": "Berlin Modern Art Museum",
        "published_at": null,
        "retrieved_at": "2026-07-17T10:00:00Z"
      }]
    },
    {"kind": "assistant", "message_id": "a2", "audible_text": "It closes at eight tonight.", "tool_calls": []},
    {"kind": "user", "message_id": "u2", "text": "sorry I meant the one in Hamburg", "speech_style": "repair"},
    {
      "kind": "assistant",
      "message_id": "a3",
      "audible_text": "I'll check the Hamburg museum instead.",
      "tool_calls": [{
        "call_id": "call-2",
        "tool_name": "search",
        "arguments": {"kind": "object", "fields": [{
          "name": "query",
          "value": {"kind": "string", "value": "Modern Art Museum Hamburg closing time 2026-07-17"}
        }]}
      }]
    },
    {
      "kind": "tool_result",
      "message_id": "t2",
      "call_id": "call-2",
      "outcome": {
        "status": "success",
        "value": {"kind": "string", "value": "The Hamburg Modern Art Museum closes at 18:00 tonight."}
      },
      "citations": [{
        "citation_id": "cite-2",
        "url": "https://example.test/hamburg-hours",
        "title": "Opening hours",
        "publisher": "Hamburg Modern Art Museum",
        "published_at": null,
        "retrieved_at": "2026-07-17T10:01:00Z"
      }]
    },
    {"kind": "assistant", "message_id": "a4", "audible_text": "The Hamburg one closes at six tonight.", "tool_calls": []},
    {"kind": "user", "message_id": "u3", "text": "and is that later than yesterday", "speech_style": "pronoun_follow_up"},
    {"kind": "assistant", "message_id": "a5", "audible_text": "I only have tonight's hours, not yesterday's.", "tool_calls": []}
  ],
  "metadata": {
    "generation": {"method": "synthetic", "teacher": "approved-teacher@revision", "prompt_revision": "realizer-v1", "seed": 23},
    "scenario": {"family": "correction_and_follow_up", "tool_need": "required", "outcome": "success", "language": "en", "length_band": "medium"},
    "split": {"name": "test", "policy_revision": "family-holdout-v1", "leakage_group_id": "venue-correction-hours"},
    "audit": {"source_license": "synthetic-no-external-text", "intended_license": "project-owned", "human_review": "approved", "content_sha256": "fixture-placeholder"}
  }
}
```

### Sequential two-call example

This deliberately simple shape is included in pilot training.

```json
{
  "schema_version": "voice-light.tool-dialogue/v1",
  "record_id": "two-call-001",
  "system_instruction": "Use a second tool only when its input depends on the first result.",
  "tools": [
    {
      "name": "get_time",
      "description": "Get the current local date and time.",
      "latency": "immediate",
      "parameters": {
        "type": "object",
        "properties": [],
        "additional_properties": false
      },
      "result_description": "Local date and time with UTC offset."
    },
    {
      "name": "calculate",
      "description": "Evaluate a basic arithmetic expression.",
      "latency": "immediate",
      "parameters": {
        "type": "object",
        "properties": [{
          "name": "expression",
          "required": true,
          "schema": {"type": "string", "description": "Basic arithmetic expression", "min_length": 1, "max_length": 160}
        }],
        "additional_properties": false
      },
      "result_description": "The calculation result as a string."
    }
  ],
  "messages": [
    {"kind": "user", "message_id": "u1", "text": "how many hours until six this evening", "speech_style": "asr_clean"},
    {
      "kind": "assistant",
      "message_id": "a1",
      "audible_text": "I'll check the time first.",
      "tool_calls": [{
        "call_id": "call-1",
        "tool_name": "get_time",
        "arguments": {"kind": "object", "fields": []}
      }]
    },
    {
      "kind": "tool_result",
      "message_id": "t1",
      "call_id": "call-1",
      "outcome": {"status": "success", "value": {"kind": "string", "value": "2026-07-17T14:30:00+02:00"}},
      "citations": []
    },
    {
      "kind": "assistant",
      "message_id": "a2",
      "audible_text": "It's two thirty. I'll calculate that.",
      "tool_calls": [{
        "call_id": "call-2",
        "tool_name": "calculate",
        "arguments": {"kind": "object", "fields": [{
          "name": "expression",
          "value": {"kind": "string", "value": "(18 * 60 - (14 * 60 + 30)) / 60"}
        }]}
      }]
    },
    {
      "kind": "tool_result",
      "message_id": "t2",
      "call_id": "call-2",
      "outcome": {"status": "success", "value": {"kind": "string", "value": "3.5 hours"}},
      "citations": []
    },
    {"kind": "assistant", "message_id": "a3", "audible_text": "There are three and a half hours until six.", "tool_calls": []}
  ],
  "metadata": {
    "generation": {"method": "authored", "teacher": null, "prompt_revision": "manual-v1", "seed": null},
    "scenario": {"family": "dependent_time_calculation", "tool_need": "multiple_required", "outcome": "success", "language": "en", "length_band": "medium"},
    "split": {"name": "test", "policy_revision": "family-holdout-v1", "leakage_group_id": "dependent-time-arithmetic"},
    "audit": {"source_license": "project-authored", "intended_license": "project-owned", "human_review": "approved", "content_sha256": "fixture-placeholder"}
  }
}
```

## Qwen rendering and loss masking

The Qwen adapter is a pure boundary transformation:

1. Convert canonical parameter schemas to JSON Schema dictionaries.
2. Convert semantic values to JSON-compatible values.
3. Convert canonical messages to standard system/user/assistant/tool messages. Assistant messages
   keep `audible_text` in `content` and structured calls in `tool_calls`; tool outcomes serialize
   as deterministic JSON associated with the call ID.
4. Invoke the pinned tokenizer's
   `apply_chat_template(messages, tools=tools, enable_thinking=False, tokenize=True)`.
5. At inference add `add_generation_prompt=True`; for complete SFT records use
   `add_generation_prompt=False`.

The canonical file never stores the rendered text. The manifest stores the tokenizer ID, immutable
revision, adapter revision, and hashes needed to reproduce it.

Assistant-only labels retain:

- every audible bridge token;
- every structured tool-call token emitted within the assistant span;
- every ordinary assistant response;
- every post-result continuation;
- every later assistant response in a multi-turn history.

They mask:

- system instructions;
- runtime tool definitions;
- user/ASR text;
- tool-result payloads, failures, and citations;
- padding.

The implementation should ask TRL for a Qwen3 training-compatible template containing generation
markers, then assert its rendered token IDs are identical to the pinned native template. If that
assertion fails, Phase 2 must stop and use an explicit token-span collator rather than silently
training the wrong protocol.

Required deterministic tests:

- canonical JSON serialize/parse equality for every union variant;
- rejection of extra fields and invalid call/result sequences;
- exact expected Qwen message objects for all five examples;
- native-versus-training template token equality;
- snapshot renders for no-tool, one-call, failure, follow-up, correction, empty-result, conflicting
  citations, and optional two-call records;
- label masks include bridge/call/final assistant tokens and exclude tool results;
- no `<tool_call>` or provider syntax exists in canonical `audible_text`;
- tools and messages round-trip without changing call IDs or typed values;
- truncation either preserves a complete causal suffix or rejects the record; it never leaves an
  orphan call/result.

## Coverage-driven synthetic generation

### Architecture

```text
authored ScenarioFamily
  + conditionally sampled axes
  + deterministic tool definitions/results
  -> ScenarioSpec (typed)
  -> teacher semantic realization (typed, no protocol markup)
  -> deterministic validators
  -> selective typed critic
  -> validators again
  -> de-duplication
  -> stratified human review
  -> immutable canonical JSONL + manifest
```

The teacher never writes Hermes strings. It receives a typed scenario and returns one typed
semantic step at a time. Deterministic code owns IDs, tool definitions, timestamps, split
assignment, and rendering.

Tool outcomes use a controlled synthetic strategy:

- `calculate`: deterministic restricted-expression execution supplies the exact result;
- `get_time`: a seeded timestamp generator supplies a valid date, time, and UTC offset;
- `search`: a separate request to the same Qwen teacher answers the query directly with a concise,
  plausible result;
- the next assistant request receives only the public history plus the newly appended result.

Search results need not be verified against the real world for this behavioral SFT. The purpose is
to teach the student to wait for and use the returned string, not to create a factual search
benchmark. The staged history boundary prevents the continuation from seeing a result before the
call. Tool-result tokens remain masked during SFT.

### Staged causal rollout

Do not ask the teacher for an entire conversation containing future tool results in one request.
Use a state machine:

```text
ScenarioSpec with an expected simple tool plan
  -> generate or realize next user utterance
  -> generate AssistantStep as schema-constrained JSON
     -> final speech: commit the assistant step
     -> speech + call: commit the assistant step and pause
  -> validate the call against the expected plan and runtime tool schema
  -> execute deterministic tool or synthetic-search oracle
  -> append call-linked string result
  -> generate the next AssistantStep from the now-expanded public history
  -> repeat up to the approved call limit
```

`AssistantStep` is a discriminated union between `final_response` and `tool_action`. A
`tool_action` contains audible text and exactly one call. The request ends at that typed object, so
no fragile token-level stopping rule is necessary. IDs are assigned by deterministic code, not the
teacher.

This mirrors Voice Light's intended runtime semantics: bridge and call are generated before the
result exists; the result is appended with its call ID; the continuation sees both. For throughput,
run many conversation state machines concurrently. A conversation waiting for a tool result goes
to the back of the queue while vLLM batches ready states from other records.

For synthetic `search`, use a separate prompt and visibility boundary:

```text
input: query
output: one short, plausible search-result string
```

The assistant-step prompt receives the resulting string only after the call. This tests and teaches
causal grounding even though no real search engine participates.

### Scenario families

- hard no-tool: arithmetic, stable knowledge, summarization of user-provided text, ordinary
  conversation, opinions, and questions answerable from the existing history;
- one-call current lookup: schedules, current status, recent result, price, availability, current
  office/venue hours, and externally verifiable facts;
- clarification before tool: missing location/date/entity, ambiguous pronoun, and similar names;
- successful retrieval: one source, multiple agreeing sources, source with missing fields, and
  empty hits;
- adverse retrieval: handler failure, timeout, cancellation, malformed result fixture, stale source,
  conflicting sources, and uncertain recency;
- multi-turn grounding: pronoun follow-up, ellipsis, correction, changed constraint, comparison to a
  prior result, and request for provenance;
- tool contrasts: same intent with and without a relevant supplied tool, distractor tools, renamed
  equivalent tools, and tool unavailable;
- ordinary spoken behavior: ASR-like fragments, repairs, casing/punctuation loss, concise direct
  speech, and varied formality;
- dependent sequential research, only if multiple rounds are approved.

### Orthogonal axes and conditional sampling

Axes include tool availability, expected sequential tool plan, tool need, result status, latency
band, number of user turns, reference type, correction location, source agreement, time
sensitivity, user register, ASR noise, bridge intent, answer length, language, and domain. Their
valid cross-product is the scenario design space, not the emitted dataset. A seeded stratified
sampler selects distinct compatible combinations from that space, caps common combinations, and
holds out selected combinations for evaluation. Exhausting the literal Cartesian product would
overweight implausible combinations and create an artificial training distribution.

Examples of incompatibility constraints:

- no-tool records cannot have a result status or procedural bridge;
- timeout records cannot contain grounded result facts;
- source-conflict records require at least two citations;
- pronoun follow-ups require an unambiguous earlier entity;
- correction axes require at least two user turns;
- a second call requires a first result that supplies an argument needed by the second;
- immediate tools do not require a latency bridge;
- English ASR-fragment transformations cannot be blindly reused for another language.

Start with roughly 30-50 authored family templates and 5-12 parameterized semantic frames per
family. Weighted conditional sampling, entity/date/value banks, optional turns, tool-subset
variation, and teacher paraphrase can create thousands of distinct records without pretending that
every combination is equally natural.

### Proposed pilot distribution

| Dimension | Proposed weight |
| --- | ---: |
| Hard no-tool / ordinary dialogue | 35% |
| Tool-needed success | 40% |
| Empty, failure, timeout, or conflict | 15% |
| Clarification before any call | 10% |
| Short, 2-4 dialogue messages | 25% |
| Medium, 5-8 dialogue messages | 55% |
| Long, 9-14 dialogue messages | 20% |
| English | 100% |

Follow-ups and corrections cut across the tool-needed categories. The English-only recommendation
matches the currently deployed English Nemotron ASR. Multilingual Qwen capability alone is not a
reason to train language paths the deployed speech stack cannot exercise reliably.

The tool-round distribution is:

| Tool path per assistant response | Proposed weight |
| --- | ---: |
| No call | 35% |
| One call | 45% |
| Two sequential calls | 18% |
| Three sequential calls | 2% |

Sequential means one call, its result, a short natural intermediate assistant turn, then the next
call. Do not bundle parallel calls into one assistant message for this milestone. Favor simple
chains such as `search -> calculate`, `get_time -> calculate`, or a refined second `search`.
Three-call chains are deliberately rare and must pass stricter human review. The Apollo elapsed
seconds example is useful as an evaluation stress case, but is too complex to be a common training
pattern.

The proposed 4,000-record build proceeds as a 20-record quantization/serving smoke test, 100
manually reviewed calibration records, an intermediate 2,500-record accepted checkpoint, and then
1,500 more only if coverage and quality statistics remain healthy. The 100-record run determines
real generation time. A provisional 8-30 RTX 4090 GPU-hour range includes staged generation and
selective critique but excludes human review.

### Teacher prompts

The realizer system prompt should contain:

```text
Return exactly one SemanticDialogueDraft matching the supplied schema.
Use only facts in ScenarioSpec and ToolOutcomeSpec.
Do not emit XML, chat-template tokens, tool-call tags, or JSON inside audible_text.
Keep speech natural for a voice conversation.
If an assistant makes a latency-bearing call, audible_text is one sentence and at most
eight spoken words, communicates the next action, and states no pending result.
Use a tool only when ScenarioSpec marks it needed and only from available_tools.
After a result, answer directly from that result. Preserve uncertainty, conflict, empty
results, timeout, and cancellation. Resolve references and corrections from history.
```

The critic receives the scenario, record, and a typed rubric and returns only
`CritiqueReport`:

```text
accepted: bool
violations: tuple[RubricViolation, ...]
suggested_record: optional SemanticDialogueDraft
```

Use critique only for high-risk families, low validator confidence, and a random audit slice.
Do not pay or compute for universal self-critique until it demonstrates higher human acceptance.
For difficult cases, generate two independent drafts and choose only when deterministic grounding
and a rubric critic agree.

### Validators

Deterministic validators enforce schema, call references, tool/argument validity, outcome order,
word/sentence limits, protocol leakage, pre-result fact leakage, source/citation identity,
grounding against supplied facts, failure non-fabrication, split membership, hashes, and phrase
frequency.

Grounding checks should compare normalized typed facts rather than ask a model whether an answer
"sounds grounded." For open paraphrases, a critic may flag candidates, but targeted human review
remains the authority.

Bridge controls:

- cap any exact normalized bridge at 0.5% of tool-call records;
- cap a bridge intent/lemma family at 2%;
- reject generic filler that adds no action signal;
- ban success promises and result-shaped facts before the result;
- report bridge entropy, top phrases, and domain/style distribution.

### De-duplication and splits

Assign split groups before realization. Hold out whole scenario families and selected cross-axis
combinations, such as:

- unseen tool name with a familiar schema;
- familiar domain with a new failure type;
- correction plus timeout;
- pronoun follow-up after conflicting sources;
- no-tool intent with a tempting distractor tool.

Never randomly split rows after paraphrasing. All records derived from the same semantic frame,
entity set, teacher parent, or paraphrase cluster share one leakage group. Use normalized exact
hashes, token n-gram MinHash, and semantic review to reject near duplicates across groups.
Target 80% train, 10% validation, and 10% locked test by leakage group, while keeping rare failure
families sufficiently represented. Human-review 100% of the locked test set, all sequential-call
records, and a stratified 10% of train/validation for the pilot.

## Fine-tuning plan

### Method comparison

| Method | Expected memory for 1.7B | Advantages | Risks | Pilot decision |
| --- | ---: | --- | --- | --- |
| Full BF16 SFT with AdamW | Roughly 24-35+ GB after weights, gradients, optimizer states, master weights, and activations | Maximum capacity; simple deployment artifact | Unnecessary cost and greater forgetting risk for a narrow behavior | No |
| BF16 LoRA SFT | Roughly 8-14 GB depending on sequence length, batch, checkpointing, and kernels | No quantization error; small reproducible adapter; easy ablation | Needs more memory than QLoRA; merge step later | Recommended on separate 24 GB GPU |
| NF4 QLoRA SFT | Roughly 5-10 GB | Safest fit for a separate 12 GB allocation | Quantized training path adds complexity and possible quality variance | Fallback |
| DPO after SFT | Often requires policy and reference/adapters plus preference records | Directly learns chosen versus rejected bridges/styles | No evidence yet that SFT is insufficient; synthetic preferences can amplify teacher bias | Later only |
| ORPO | Reference-free combined preference/SFT objective | Lower memory than DPO reference path | Experimental stack and still requires credible rejected examples | Later only |

These are planning ranges, not measurements. Phase 3 must log peak allocated/reserved VRAM and
actual throughput.

### Proposed LoRA pilot

- Base: exact production checkpoint and tokenizer revision.
- Precision: BF16 base and adapter compute on an approved Ampere-or-newer 24 GB GPU.
- Targets: all linear layers as a starting ablation; rank 16, alpha 32, dropout 0.05.
- Sequence length: 2,048 initially; separately report rejected/truncated records. Move to 4,096
  only if medium/long histories are materially truncated.
- Optimizer: AdamW or paged AdamW only if measurement justifies it.
- Learning rate search: `5e-5`, `1e-4`, and `2e-4` on a small development subset; no silent default.
- Epochs: at most two full pilot epochs, with evaluation at fixed optimizer-step intervals.
- Effective batch: 16 sequences as a starting target, adjusted for measured VRAM.
- Gradient checkpointing: on if needed; log its throughput cost.
- Packing: off in the first correctness run. Consider it only after mask and boundary tests pass.
- Seeds: at least three evaluation seeds; one training seed in the pilot, a second only if results
  are near the acceptance boundary.
- Artifact: adapter safetensors, tokenizer/config references, training config, environment lock,
  dataset manifest hash, logs, and checkpoint hashes. A merged BF16 artifact is derived and hashed
  later, not the only retained output.

At 1.5-2.5 million rendered tokens and two epochs, the pilot processes about 3-5 million tokens.
On a separate RTX 4090/L40S-class GPU, a reasonable planning range is 1-4 hours per hyperparameter
run and 6-20 total GPU-hours including smoke tests and failed configurations. Actual sequence
length, attention kernels, checkpointing, and dataloader behavior dominate; this must be measured.

### Why SFT should be first

The desired output is explicit and demonstrable: speak a bounded bridge, emit a valid call, use a
typed result, and continue. SFT directly places probability mass on those assistant tokens and is
the simplest test of whether the 1.7B checkpoint can learn the shape. Preference optimization adds
value only if held-out errors become relative-quality decisions that cannot be fixed with better
demonstrations, for example two valid bridges where one is substantially less repetitive or more
natural.

## Evaluation plan

### Offline generation configuration

Evaluate:

1. exact production settings: `enable_thinking=False`, `max_new_tokens=256`,
   `do_sample=True`, temperature `0.6`, top-p `0.9`, and the production system instruction;
2. the same settings over at least five fixed seeds per locked example;
3. a deterministic diagnostic mode only for attribution, clearly separated from product metrics;
4. base versus adapter and, only later, adapter versus merged artifact.

Do not tune on the locked test set. Keep tool execution deterministic in model-shape evaluation and
run a separate live-search robustness suite with recorded, licensed result fixtures.

### Proposed acceptance gates

| Metric | Pilot gate |
| --- | ---: |
| Canonical schema, referential, and renderer validity | 100% |
| Audible protocol/JSON/call-ID leakage | 0 |
| Valid call among tool-call attempts | at least 98% |
| Tool-needed recall | at least 95% |
| Hard no-tool false-call rate | at most 3% |
| Required latency bridge present | at least 95% |
| Bridge one-sentence/eight-word compliance | at least 98% |
| Unsupported result claim before result | 0 in locked deterministic checks and human audit |
| Successful-result grounded answer | at least 95% |
| Failure/timeout non-fabricating recovery | at least 95% |
| Multi-turn reference/correction success | at least 90% |
| Exact bridge phrase maximum frequency | at most 0.5% in generated corpus; report model outputs separately |
| Ordinary no-tool quality regression | no more than 2 percentage points on the locked regression rubric |
| Human naturalness | at least 85% acceptable; adapter preferred to base in at least 60% of non-tied bridge comparisons |

Report Wilson or bootstrap confidence intervals by scenario group rather than presenting small-set
percentages as precise facts. Also report latency to first audible token, tool-call start, parse
failure, completion length, repetition, language drift, and seed variance.

The pilot proceeds to scale-up only if it passes protocol leakage, grounding, no-tool false-call,
and call-validity gates. Marginal naturalness alone does not justify a larger dataset.

## Runtime history, commitment, and journal design

The later runtime design should have three explicit owners:

```text
PrivateModelConversation
  validated user messages
  assistant audible content + structured calls
  typed tool outcomes
  later model-visible turns

DurableAudibleHistory
  user turns accepted by session policy
  only assistant words proven played by browser boundaries

ToolExecutionJournal
  generation/user-turn/transcript anchors
  invocation and call IDs
  requested arguments
  lifecycle timestamps
  typed outcome
  cancellation/invalidation/wasted status
```

Commit semantics:

- Commit the private assistant message only after the first invocation completes and the parsed call
  is validated as belonging to the current generation.
- Commit the private typed result when execution finishes and every generation/user-turn/transcript/
  invocation/call anchor is still current, before starting the continuation.
- Preserve those private messages for later user turns even though the audible history contains
  only played speech.
- Continue to advance durable assistant history only from browser-proven word boundaries or full
  playback completion.
- Never place tool JSON, citations, call IDs, or unplayed assistant text in durable audible history.
- Journal cancellation independently. Generation replacement invalidates the anchor and late
  results may update only the journal as discarded/wasted; they cannot enter model context or
  trigger a continuation.

Multiple sequential rounds require replacing the single `ActiveToolExecution` with an
ordered collection or round state machine, allowing tools again after a result, extending parser
policy beyond its current second-call rejection, and keeping one TTS/playback generation across
each gap. That materially affects parser, worker protocol, session lifecycle, token accounting,
latency instrumentation, cancellation, overlap during silent gaps, and tests. The approved data
pilot teaches sequential rounds, but runtime integration remains a later separately validated
phase.

## Repository impact and staged migration

### Phase 2, after design approval

Proposed new modules:

```text
app/training/tool_use/
  schema.py                 canonical frozen Pydantic models and unions
  parameter_schema.py       typed JSON-schema-compatible subset
  values.py                 typed semantic JSON values
  validation.py             record and corpus validators
  scenarios.py              typed families, axes, weights, constraints
  generation.py             approved teacher adapter and retry/critique flow
  split.py                  leakage grouping and held-out assignment
  deduplication.py          exact and near-duplicate controls
  renderers/
    base.py                 renderer protocol and rendered batch types
    qwen.py                 pinned Qwen message/template adapter
  data.py                   JSONL loader and assistant-only collator
  config.py                 frozen generation/training/evaluation configs
  cli.py                    validate, generate, render, inspect, and report commands

tests/training/tool_use/
  test_schema.py
  test_validation.py
  test_scenarios.py
  test_split.py
  test_qwen_renderer.py
  test_loss_mask.py
  fixtures/                 only tiny representative records
```

Add a dedicated optional Linux training dependency group for pinned compatible `trl`, `peft`,
`accelerate`, `datasets`, and optionally `bitsandbytes`; do not add training libraries to the live
runtime dependency path.

The repository already ignores `/data/`. Add `/artifacts/` and an explicit documented artifact
manifest policy before producing checkpoints. Commit schemas, configs, prompt templates, tiny
fixtures, reports, and hashes. Do not commit generated corpora, raw teacher output, model weights,
adapters, merged weights, optimizer states, or private review exports.

Phase 2 stops after the approved pilot records are generated, validated, summarized, and sampled
for review. Training still requires a new explicit approval.

### Phase 3, after training approval

- provision only the approved separate allocation/window;
- record GPU, driver, CUDA, package lock, dataset hash, prompt hash, base revision, config, seed,
  logs, checkpoints, and SHA-256 hashes;
- run smoke, pilot, and held-out evaluation;
- run existing Voice Light unit/regression suites without starting or replacing deployed services;
- report measured results before any integration or merge decision.

### Phase 4, after integration approval

- introduce persistent private model conversation and the execution journal;
- adapt runtime tool types from the one-weather closed union to an approved typed registry;
- integrate the chosen adapter or merged artifact;
- preserve the current audible-history contract;
- extend to multiple rounds only if separately approved;
- validate in an exclusive window so a second Qwen is never loaded beside deployed services;
- do not deploy until explicitly requested.

Affected existing tests include at least:

- `test_hermes_tool_parser.py`: second-call policy, delimiter and post-call content;
- `test_qwen_worker.py`: template/rendering, system instruction, and adapter load;
- `test_models.py`: typed protocol messages and invocation behavior;
- `test_tools.py`: generic registry, search schema/result/citations, permissions, failures;
- `test_session.py`: private context across later turns, result commitment, stale results,
  cancellation, timeout, TTS continuity, history boundaries, backchannels during tool gaps, and
  optional repeated rounds;
- browser playback tests if multiple silent tool gaps change generation-completion behavior.

## Decision status after the first checkpoint

1. **Teacher/provider: approved with quantization experiment.** Self-hosted
   `Qwen/Qwen3.6-27B` through vLLM with thinking disabled and JSON-schema output. Prefer the
   official FP8 checkpoint when at least 40 GB usable VRAM is available. The chosen listings report
   48 GB, which must be verified inside the container before the 20-record smoke test and the 100
   manually reviewed calibration records.
2. **Fine-tuning method/checkpoint: approved in principle.** BF16 LoRA SFT on the exact pinned
   `Qwen/Qwen3-1.7B` revision, adapter retained as source of truth, with a merged BF16 candidate
   built only after evaluation. QLoRA is only a constrained-memory fallback. No DPO/ORPO in the
   pilot.
3. **Tool surface/milestone: approved with revision.** Runtime-supplied `search(query)`,
   `calculate(expression)`, and `get_time()`, each returning one rendered string. No arbitrary
   Python. Search provenance remains typed in the canonical audit data even though the model sees a
   string result.
4. **Rounds: approved with revision.** Include sequential multi-round records. Proposed
   distribution: 35% no call, 45% one call, 18% two calls, and 2% three calls. Each call and its
   result remain in private model history.
5. **Dataset and thresholds: partially approved.** English only and the 25%/55%/20%
   short/medium/long distribution are approved in principle. Revised proposal: 4,000 accepted
   records, reached through 100 and 2,500-record checkpoints, with the prior 80%/10%/10%
   family/combination-held-out splits and acceptance gates.
6. **Compute/service window: approved for generation preparation.** Start with one of the
   approximately EUR 0.40/hour RTX 4090 listings reporting 48 GB and do not use an H100. Verify
   actual CUDA-visible memory before loading official FP8. The 20-record smoke test measures
   compatibility; the 100-record calibration measures throughput and quality before any larger
   run. Deployed Voice Light inference remains running and untouched.

7. **Synthetic tool execution: approved.** Generate each conversation as a staged causal rollout.
   Execute `calculate` exactly, seed `get_time`, and use another invocation of the same Qwen teacher
   as the synthetic-search oracle. Append every assistant call and call-linked string result before
   requesting the next assistant step. Do not perform live web search in the initial dataset.

After these answers, Phase 2 may implement only the approved schema/tooling and approved pilot
generation. It must stop again with quality statistics and samples before any training run.
