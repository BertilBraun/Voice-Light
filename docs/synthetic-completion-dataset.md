# Synthetic completion dataset

## Scope

This pilot trains the user-channel completion decision, not a synthetic conversation model. Each
generated utterance has one mono user WAV and deterministic annotations derived from that WAV. It
does not contain an assistant track, artificial background noise, ASR-derived ground truth, or
synthetic validation/test splits.

The corpus and every provider invocation are English-only. Provider support for other languages is
irrelevant to this pipeline and no language selector is exposed in its prompt schema.

The two semantic labels are:

- `hold`: every internal silent interval lasting at least 500 ms, represented by its start time and
  complete duration;
- `end_of_turn`: the final speech offset after trailing generated silence is removed.

The annotator uses non-overlapping 20 ms RMS frames and a conservative threshold of the greater of
0.001 RMS and one percent of peak frame RMS. This is approximately a -40 dB relative gate and
avoids treating quiet phonemes as silence. Labels never depend on ASR or VAD output.

Twenty-second training examples are currently stored as window plans rather than duplicated audio.
Each hold and end-of-turn boundary gets a deterministic view whose anchor is uniformly placed from
4.0 to 19.2 seconds. The plan records source bounds and zero padding, so audio windows and 80 ms
frame targets can be materialized later without changing label timing.

## Prompt design

`generate_completion_prompts.py` uses `Qwen/Qwen3-4B-Instruct-2507` to create validated structured
prompts. Text is English-only, 55–115 words, a complete conversational turn, and contains two or
three natural pause opportunities. VoiceDesign instructions vary perceived age, vocal character,
English accent or dialect, pace, emotion, pitch, energy, and pause delivery while requiring clean,
close-mic voiced speech. Invalid JSON, invalid word counts, duplicate text, and noise-prone voice
directions such as breathy, hushed, or whispered delivery are discarded and retried.
The `balanced` delivery profile covers multiple speaking rates. The `brisk_engaged` profile accepts
90–130 words while requesting 110–130, compiles an explicit 200–240 spoken words-per-minute
instruction, keeps one planned 500–800 ms
hold, and varies English accents, pitch, age presentation, and vocal weight without reflective or
soft-spoken delivery. The prompt artifact records the profile, exact model revision, runtime, and
seed.

## TTS candidates

| Model | License | Languages and control | 12 GB pilot role |
| --- | --- | --- | --- |
| Qwen3-TTS 1.7B VoiceDesign | Apache-2.0 | Ten languages, natural-language voice design, batched generation | Primary recommendation. Batch 8 fits and provides the strongest prompt-level voice diversity. |
| Chatterbox Multilingual V3 0.5B | MIT | 23+ languages, reference voice, exaggeration, CFG, temperature | Quality/diversity comparison. The official API is single-item, so reference encoding is included in throughput. |
| VoXtream2 0.5B | CC-BY-4.0 weights | English output, translingual reference voices, dynamic speaking rate | Fast streaming comparison with multiple official reference voices and rate variants. |
| Kyutai TTS 1.6B (actually 1.8B) | CC-BY-4.0 weights | English/French, supplied voice embeddings, efficient batching | Follow-up candidate already integrated into Voice-Light; useful if the first three pass listening review. |

The pilot pins Qwen VoiceDesign revision
`5ecdb67327fd37bb2e042aab12ff7391903235d3`, Chatterbox model revision
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` with source revision
`5de7a54aa4e5e2baadb0182dde554908b48b85c2`, and VoXtream2 model revision
`49addec130217e8e9e82a6f49437c315c5c851fc` with source revision
`8ec2d62159dae4716ae7058827244a962d40603c`.
The Chatterbox environment pins `setuptools==80.9.0` because Perth 1.0.1 still imports
`pkg_resources`, which is absent from newer setuptools releases.
VoXtream2 additionally requires the system `espeak-ng` package for English phonemization.

Primary sources: [Qwen3-TTS model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign),
[Chatterbox repository](https://github.com/resemble-ai/chatterbox),
[VoXtream2 model card](https://huggingface.co/voxtream2/model), and
[Kyutai TTS model card](https://huggingface.co/kyutai/tts-1.6b-en_fr).

## Artifacts and commands

Every provider directory contains:

- `audio/*.wav`: trimmed mono PCM16 speech;
- `utterances.jsonl`: prompt, boundaries, detector configuration, exact provider/model/runtime
  provenance, controls, timing, RTF, and reference-audio hashes where applicable;
- `windows.jsonl`: deterministic 20-second window recipes;
- `run.json`: aggregate elapsed generation, audio duration, hold count, sample count, and peak CUDA
  allocation;
- `validation.json`: machine quality-gate results.

Typical generation commands are:

```text
python -m app.local.synthetic_generation.generate_completion_qwen ... --batch-size 8 --run-seconds 7200
python -m app.local.synthetic_generation.generate_completion_chatterbox ... --run-seconds 7200
python -m app.local.synthetic_generation.generate_completion_voxtream ... --variants-per-prompt 8 --run-seconds 7200
python -m app.local.synthetic_generation.validate_completion_corpus --corpus <provider-dir> --output <provider-dir>/validation.json
```

The validator rejects missing or empty audio, stereo output, sample-rate or duration mismatches,
utterances outside 20–60 seconds, low-energy audio, excessive clipping, and utterances without a
500 ms hold. Passing it does not establish naturalness. Before scaling, a blinded human review must
compare randomly ordered provider samples and reject providers with noise, voice artifacts,
hallucinated content, unnatural pausing, or truncated endings.

## Training and evaluation plan

Group splits by prompt family, topic, and reference-voice lineage so near-duplicate text or acoustic
identity cannot cross splits. Keep the existing real validation and test splits unchanged. Run the
same training budget and seed grid for:

1. real-only baseline;
2. synthetic-only training, evaluated only on real validation/test and external causal benchmarks;
3. mixed real/synthetic training at several sampling ratios;
4. synthetic pretraining followed by real-only fine-tuning.

Select models on the existing real validation split. The operational release gate remains the real
held-out test split plus external causal benchmarks, with separate hold-pause and end-of-turn recall,
false-yield rate, calibration, and latency. Synthetic validation is useful for pipeline regression
only and must never substitute for the real gate.

Acoustic augmentation is intentionally deferred until clean provider output passes listening review.
If added later, augmentation should remain mild, run after boundary creation, preserve exact sample
count, and be evaluated as separate clean and augmented conditions rather than contaminating the
base corpus.
