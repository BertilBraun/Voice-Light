# Teacher-led tool-use generation experiment

The `teacher_led_tool_use` scenario profile tests whether the larger teacher can create coherent
tool-use conversations with substantially less turn planning.

Each scenario supplies only:

- one loose conversation brief;
- one speech style;
- four user turns;
- the available `search`, `calculate`, and `get_time` tools;
- a maximum of two causal tool calls for each user turn.

The teacher chooses every concrete user utterance, follow-up, assistant response, tool decision,
and tool argument. No expected tool is assigned to an individual turn. When the teacher chooses a
call, the controller executes it and appends its result before asking for the next assistant step.
This preserves causal grounding while allowing the teacher to decide the conversational path.

Semantic audits remain optional. The default experiment uses only structured-output parsing,
canonical schema validation, safe calculation, bounded turn length, and causal call/result order.
Generated metadata describes the calls that actually occurred rather than a predetermined plan.

Create a small scenario set:

```powershell
uv run python -m app.training.tool_use.cli plan `
  data\tool-use\teacher-led-smoke\scenarios.jsonl `
  --count 20 `
  --seed 20260719 `
  --profile teacher_led_tool_use
```

Generate records against the rented vLLM node:

```powershell
uv run python -m app.training.tool_use.cli generate `
  data\tool-use\teacher-led-smoke\scenarios.jsonl `
  data\tool-use\teacher-led-smoke\records.jsonl `
  --failures data\tool-use\teacher-led-smoke\failures.jsonl `
  --manifest data\tool-use\teacher-led-smoke\run-manifests.jsonl `
  --request-log data\tool-use\teacher-led-smoke\vllm-requests.jsonl `
  --base-url http://127.0.0.1:8000/v1 `
  --api-key VOICE_LIGHT_LOCAL `
  --model Qwen/Qwen3.6-27B-FP8 `
  --model-revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09 `
  --quantization fp8 `
  --time-reference 2026-07-19T12:00:00+00:00 `
  --concurrency 20
```

Do not enable `--semantic-audits` for the first sample experiment. Review the raw records so the
teacher's unaided strengths and failure modes remain visible.
