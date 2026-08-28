---
license: apache-2.0
language:
  - en
pretty_name: Voice-Light Synthetic Audio
tags:
  - audio
  - synthetic-data
  - speech
  - turn-taking
---

# Voice-Light Synthetic Audio

English-only synthetic conversational speech for training and evaluating streaming
turn-taking models. The corpus focuses on end-of-turn prediction, continuation holds,
short backchannels, interruptions, and response timing.

The dataset contains user-side FLAC speech units plus typed conversation plans,
rendering provenance, quality ledgers, and deterministic reconstruction metadata.
Assistant speech is represented as a time-varying speaking-probability input rather
than an audible track.

## Layout

Each versioned run contains:

- `prompts.json`: semantic conversation plans;
- `references/`: Qwen VoiceDesign reference clips and their manifest;
- `units/`: trimmed CosyVoice speech units, their render manifest, and deterministic
  download shards;
- `audit-prompts.json` and `audit-complete.json`: complete, unfiltered quality flags;
- `provenance.json`: model revisions, source-code revision, and file checksums.

The `builder/` directory contains source snapshots for the revisions recorded by
the run provenance. `voice-light-source.tar.gz` is the V4 snapshot, while later
runs whose compiler changed use a revision-qualified archive. Extract the archive
matching the run's `source_code_revision`, or use a Voice-Light checkout at that
revision.

Rendered conversations and dense 20-second targets are reproducible build products and
are intentionally not stored as canonical audio. The included builder code reconstructs
conversation timelines, varies synthetic assistant durations, selects training crops,
and derives hard targets at 80 ms without using ASR or VAD as ground truth.

## Rebuild training samples

Use an immutable dataset commit and the included Voice-Light source revision:

```bash
python -m app.local.synthetic_generation.cli prepare-hub-training \
  --revision <dataset-commit-sha> \
  --run v4 \
  --run v5 \
  --output synthetic-training \
  --crop-variants 4
```

This downloads the canonical prompt set, render manifest, and roughly 64 MB unit archives, then
verifies and extracts the trimmed FLACs. It reconstructs each user-side conversation, samples
balanced 20-second views, and writes the standard Voice-Light Parquet training contract. All
descendants of one conversation retain one split.

## Intended use

This is research data for Voice-Light's streaming turn-taking adapter. Synthetic
training results should still be evaluated on held-out real conversations and external
causal turn-taking benchmarks. The voices and conversations are generated; they should
not be treated as recordings of real people or naturally occurring interactions.

Generation uses Qwen3-TTS VoiceDesign references and CosyVoice zero-shot synthesis.
See each run's provenance and quality ledger for exact revisions and known artifacts.
