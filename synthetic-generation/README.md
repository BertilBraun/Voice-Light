# Qwen3-TTS Synthetic Conversation Experiments

Small local proof of concept for generating, comparing, manually aligning, and exporting synthetic two-speaker conversational interactions.

The generation pipeline is backend-independent. Qwen VoiceDesign is the first real backend; a fake backend is included for local tests and pipeline checks.

Case authoring rules live in [CASE_DESIGN_GUIDE.md](CASE_DESIGN_GUIDE.md). Use that guide
when asking an LLM to write new batches.

## Remote GPU Generation

```powershell
uv sync
uv run python -m src.generate_samples --experiment examples_v1 --variants 3
```

For the Qwen backend, install the optional Qwen runtime extra on the GPU node:

```powershell
uv sync --extra qwen
```

The Qwen project currently documents a Python 3.12 environment for its package. This proof of concept targets Python 3.11 for the provider-independent tooling, with Qwen isolated behind the optional backend dependency.

With explicit cases and backend:

```powershell
uv run python -m src.generate_samples `
  --backend qwen-voice-design `
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign `
  --cases interaction_cases.json `
  --experiment examples_v1 `
  --variants 3 `
  --seed 42
```

Package for transfer:

```powershell
uv run python -m src.package_experiment outputs/examples_v1
```

This creates `outputs/examples_v1.zip`.

Render every successful A/B variant combination as WAV mixes:

```powershell
uv run python -m src.export_mixes outputs/examples_v1
```

This writes `outputs/examples_v1/mixes/*.wav`, one JSON sidecar per mix, and
`outputs/examples_v1/mixes/index.json`.

## Local Alignment

After extracting an experiment zip:

```powershell
uv run python -m src.launch_aligner outputs/examples_v1
```

Custom port and initial case:

```powershell
uv run python -m src.launch_aligner outputs/examples_v1 --port 8123 --case corrective_interruption
```

The browser opens a URL such as:

```text
http://localhost:8000/web/aligner.html?manifest=/experiments/examples_v1/experiment.json
```

Audio is loaded through HTTP URLs, never arbitrary local filesystem paths.

## Backend Configuration

Use CLI flags:

```powershell
uv run python -m src.generate_samples --backend qwen-voice-design --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

Or a config file:

```json
{
  "backend": "qwen-voice-design",
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "options": {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "flash_attention": true
  }
}
```

Backend-specific options can also be repeated:

```powershell
--backend-option device=cuda:0 --backend-option flash_attention=true
```

Secrets such as hosted API keys must come from environment variables.

## ElevenLabs v3

Set the API key in the shell that runs generation:

```powershell
$env:ELEVENLABS_API_KEY = "..."
```

Run v3 with separate voices for each speaker:

```powershell
uv run python -m src.generate_samples `
  --backend elevenlabs `
  --model eleven_v3 `
  --cases interaction_cases_mixed_v2.json `
  --experiment elevenlabs_v3_mixed_v1 `
  --variants 3 `
  --seed 200 `
  --backend-option speaker_a_voice_id=<voice-id-a> `
  --backend-option speaker_b_voice_id=<voice-id-b> `
  --backend-option stability=0.4 `
  --backend-option similarity_boost=0.75 `
  --backend-option style=0.3 `
  --backend-option output_format=mp3_44100_128
```

The backend records the original text/instruction and the prepared text sent to ElevenLabs. It does not infer or add v3 audio tags. For `eleven_v3`, write any desired tags directly in `speaker_a.text` or `speaker_b.text`, because that exact text is sent to the API.

Version 6 one-of-each-type batch:

```powershell
uv run python -m src.generate_samples `
  --backend elevenlabs `
  --model eleven_v3 `
  --cases interaction_cases_elevenlabs_v3_context_v6.json `
  --experiment elevenlabs_v3_context_v6 `
  --variants 2 `
  --seed 600 `
  --auto-place `
  --backend-option speaker_a_voice_id=<voice-id-a> `
  --backend-option speaker_b_voice_id=<voice-id-b> `
  --backend-option output_format=mp3_44100_128

uv run python -m src.export_mixes outputs/elevenlabs_v3_context_v6
```

## LLM Case Prompt Template

```text
Create N short two-speaker synthetic conversation cases.

Each complete interaction should be approximately 5-10 seconds.

Return only valid JSON matching the supplied experiment schema.

For each case provide only:
- Speaker A text
- Speaker B text
- exactly one placement rule

Follow CASE_DESIGN_GUIDE.md exactly.
Speaker A and B will be generated independently.

The interaction must remain understandable when the tracks are overlaid.
For interruptions, Speaker A must stop at the interrupted/decompleted point.
For completions, do not use ellipses or trailing-off punctuation.

Use explicit ElevenLabs v3 bracket tags directly in the spoken text only when needed.
Do not add case_id, title, description, tags, alignment notes, or instructions unless specifically asked.
```

Placement rules:

```json
{
  "placement": {
    "type": "backchannel",
    "anchor_text": "I moved the budget review to next week",
    "delay_ms": 300
  }
}
```

```json
{
  "placement": {
    "type": "interruption",
    "anchor_text": "send the draft by Wednesday morning",
    "mode": "at_anchor_end",
    "lead_ms": 0
  }
}
```

```json
{
  "placement": {
    "type": "completion",
    "pause": "short"
  }
}
```

```json
{
  "placement": {
    "type": "internal_pause",
    "anchor_text": "the restart happened right after the backup job finished",
    "pause": "medium"
  }
}
```

Completion pause defaults:
- `short`: random `200-500 ms`
- `medium`: random `500-1000 ms`
- `long`: random `1000-2000 ms`

Each placement object is a discriminated Pydantic union. The generated manifest records the matching typed `placement_output` on successful variants, for example a completion output records `speaker_a_end_seconds`, `sampled_delay_ms`, and `speaker_b_start_seconds`.

Internal pause placement measures the actual ASR gap inside Speaker A after `anchor_text`; it does not align Speaker B.

Successful generated WAVs are trimmed in place to detected speech bounds before manifest writing. The manifest records `original_duration_seconds`, `trimmed_leading_seconds`, `trimmed_trailing_seconds`, `audio_onset_seconds`, and `audio_offset_seconds`, so downstream placement can treat each track as starting at actual speech time zero.

Minimal example:

```json
{
  "experiment_id": "clean_cases_v1",
  "description": "Clean placement-driven cases",
  "cases": [
    {
      "speaker_a": {
        "text": "I moved the budget review to next week because the finance team said they needed two more days to check the vendor numbers."
      },
      "speaker_b": {
        "text": "[quietly] Mm-hm."
      },
      "placement": {
        "type": "backchannel",
        "anchor_text": "I moved the budget review to next week",
        "delay_ms": 300
      }
    },
    {
      "speaker_a": {
        "text": "[conversational, focused] I can send the draft by Wednesday morning, if-"
      },
      "speaker_b": {
        "text": "[quickly, correcting] Thursday afternoon is safer."
      },
      "placement": {
        "type": "interruption",
        "anchor_text": "by Wednesday morning",
        "mode": "at_anchor_end",
        "lead_ms": 0
      }
    },
    {
      "speaker_a": {
        "text": "The export finished cleanly."
      },
      "speaker_b": {
        "text": "Great, I'll send it to the client."
      },
      "placement": {
        "type": "completion",
        "pause": "short"
      }
    },
    {
      "speaker_a": {
        "text": "The restart happened right after the backup job finished, so I waited, checked the logs again, and then opened a ticket."
      },
      "speaker_b": {
        "text": "Okay."
      },
      "placement": {
        "type": "internal_pause",
        "anchor_text": "the backup job finished",
        "pause": "medium"
      }
    }
  ]
}
```

## Development Validation

```powershell
uv run ruff format
uv run ruff check --fix
uv run pytest
```
