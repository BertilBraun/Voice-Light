# Synthetic two-party conversation pipeline

Status: runnable pilot, 25 August 2026.

## Decision

Use `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` with `qwen-tts==0.1.1` as the primary
offline batch renderer. It has the best fit for this dataset because its Apache-2.0 weights support
free-form voice design, multilingual generation, and natural-language delivery control. Keep it in
an isolated Python 3.12 CUDA environment, as the official project recommends, and record the exact
Hugging Face revision in every event source. The model repository is 4.52 GB; budget a 16 GB CUDA
GPU for the pilot and measure peak allocation rather than treating that estimate as a guarantee.
FlashAttention 2 is recommended by Qwen to reduce memory. The imported proof of concept measured
single-utterance RTF 1.93-2.01 without production batching, so a larger job should batch compatible
requests and report both utterances/second and generated-audio-hours/GPU-hour.

Use Chatterbox Multilingual V3 as the first cross-model diversity check. Use VoXtream2 when
throughput or explicit speaking-rate sweeps matter more than instruction-rich voice design. Kokoro
and Piper are useful CPU baselines, but neither should dominate the corpus because their narrower
expressive range would create an obvious synthetic domain signature. Kyutai remains valuable for
English/French streaming and has a working Voice-Light integration, but its current project adapter
uses one hard-coded voice and therefore needs voice-pool work before dataset generation.

### Candidate matrix

| Candidate | Current release/model | License | Hardware and throughput | Diversity and control | Pilot role |
|---|---|---|---|---|---|
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | `qwen-tts==0.1.1`; 1.7B VoiceDesign, 4.52 GB repository | Apache-2.0 code and weights | CUDA; 16 GB pilot budget; measured here at RTF 1.93-2.01 before batching | Multilingual; free-form voice and delivery instructions; sampling controls | Primary quality/diversity renderer |
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | `chatterbox-tts==0.1.7`; Nano 110M; Multilingual V3 500M | MIT | Nano documents 3x real time on an 8-core CPU; larger models support CPU/MPS/CUDA but require local measurement | V3 has 23+ languages, prompt voice cloning, exaggeration and CFG controls | Cross-model diversity and CPU smoke tests |
| [VoXtream2](https://huggingface.co/herimor/voxtream2) | `voxtream>=0.2`; 0.5B | MIT code, CC-BY-4.0 weights | Official card reports 4x real time and 74 ms first packet on a consumer GPU; Voice-Light's older VoXtream benchmark allocated about 2.1 GB | Zero-shot 3-10 s voice prompt; dynamic speaking-rate control; primarily English evidence | Fast controlled-rate renderer and domain ablation |
| [Kyutai TTS](https://huggingface.co/kyutai/tts-1.6b-en_fr) | `kyutai/tts-1.6b-en_fr` (card clarifies 1.8B actual parameters) | CC-BY-4.0 weights; MIT/Apache runtimes | CUDA PyTorch, Rust server, or Apple MLX; model card reports up to 75x batched throughput per compute-time unit, not single-stream RTF | English/French; supplied voice embeddings; streaming word input; current Voice-Light adapter has one voice | Existing integration; use after voice-pool support |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | v1.0, 82M | Apache-2.0 weights; verify each voice attribution | CPU-friendly and small; benchmark locally on the target CPU | Five released languages and 44 voices; limited interaction control | Cheap CPU baseline |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) | `piper-tts==1.4.2` | GPL-3.0 runtime; each voice has its own license | Fast ONNX CPU runtime | Broad per-language voice catalog; pace via `length_scale`; low expressive control | Throughput floor only after voice-license audit |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | F5-TTS v1 Base | MIT code, CC-BY-NC pretrained weights | CUDA is practical; measure locally | Strong zero-shot cloning; English/Chinese base and community language variants | Exclude from redistributable default because weights are non-commercial |
| [Fish Speech](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) | current main | Fish Audio Research License | CUDA; measure locally | Expressive multilingual cloning | Exclude: commercial use needs a separate license and outputs may not train another foundation model |

These are model/runtime requirements, not output-license conclusions. Every generation batch must
record model, model revision, runtime version, weight license, source-text provenance, voice consent
or voice-pack license, and the intended dataset license. Legal review remains necessary before a
public or commercial corpus release.

## Canonical representation

`SyntheticConversationPlan` is the label authority. A plan contains exactly one user and one
assistant, a deterministic seed, a 20-second duration, and a discriminated timeline of:

- floor-holding speech with an explicit `completion`, `continuation`, or `none` boundary;
- explicit pauses linked to the planned continuation event;
- non-floor backchannels;
- interruption attempts or successful floor takes linked to the interrupted event.

Disfluencies live in the spoken text and are additionally marked on a speech event. Overlap is the
intersection of planned event intervals. Response latency is the next floor event's start minus the
prior completed event's end. No ASR or VAD output participates in ground truth. They may be used
only for acoustic quality control, such as detecting a failed or silent TTS result.

The case generator should sample semantic plans before TTS. Recommended pilot strata are:

| Stratum | Share | Timing distribution |
|---|---:|---|
| Clean completed handoff | 30% | response latency: truncated log-normal, median 350 ms, p95 1.2 s, range 80-2,000 ms |
| Continuation/internal pause | 25% | 150-1,500 ms, oversample 400-900 ms hard negatives |
| Backchannel | 15% | onset 150-450 ms after an acknowledgeable phrase; 150-700 ms duration |
| Successful interruption | 12% | onset 100-500 ms before current planned speech end; overlap 80-500 ms |
| Unsuccessful interruption | 5% | brief overlap while the current speaker retains the floor |
| Cooperative overlap/completion | 5% | next turn starts 50-350 ms before completion |
| Disfluency/repair | 8% | fillers, repetitions, cut-offs, self-corrections, and resumed phrases |

Sample these within topic, language, speaker, voice, pace, and lexical-length strata. Keep a stable
plan ID and seed so a case can be rerendered across voices/models without changing labels.

## Rendering and labels

Each voiced event is synthesized independently, then duration-fitted into its planned interval.
The renderer creates independent user and assistant WAV channels. Only after placement does it
apply gain, additive noise, synthetic reverb, microphone band limits, mu-law quantization, and
directional cross-talk. This order preserves label time. Reverb tails and cross-talk intentionally
do not alter semantic floor labels.

`materialize_training_sample` compiles the plan directly into the existing 250 frames at 80 ms:

- `assistant_has_floor` and `p_user_has_floor` come from planned floor event occupancy;
- `turn_completion` and `continuation_pause` are paired labels at explicit user boundaries;
- `p_user_yield`, `p_assistant_backchannel`, `non_floor_feedback`, and `floor_take` come from
  semantic event roles;
- the four future-activity targets are exact planned user-floor occupancy over their horizons.

All non-applicable sparse auxiliary targets use the existing `-1` mask. The corpus builder writes
the canonical `MaterializedTrainingSample` Arrow schema, Zstandard Parquet shard, independent audio
paths, `corpus.json`, `synthetic.json`, and a render manifest with per-event hashes and TTS
provenance. Synthetic assignments are always `train`; the builder never creates synthetic
validation or test rows.

## Commands

Validate one plan:

```powershell
.\.venv\Scripts\python.exe -m app.local.synthetic_generation.cli validate-plan `
  .\synthetic-generation\pilot\plan.json
```

Build the checked-in CPU/SAPI smoke pilot:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\synthetic-generation\pilot\generate_sapi_sources.ps1

.\.venv\Scripts\python.exe -m app.local.synthetic_generation.cli build-corpus `
  --request .\synthetic-generation\pilot\request.json `
  --output .\synthetic-generation\pilot\output
```

The SAPI artifacts prove pipeline mechanics with locally available speech; they are not the model
recommendation and must not be used for quality conclusions.

## GPU generation job

Run the Qwen job on an NVIDIA host with at least 16 GB VRAM, at least 16 GB free disk, CUDA, and an
isolated Python 3.12 environment:

```powershell
uv venv --python 3.12 .venv-qwen-tts
uv pip install --python .venv-qwen-tts qwen-tts==0.1.1 soundfile
uv pip install --python .venv-qwen-tts flash-attn --no-build-isolation

.venv-qwen-tts\Scripts\python.exe -m src.generate_samples `
  --backend qwen-voice-design `
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign `
  --cases interaction_cases_elevenlabs_v3_context_v6.json `
  --experiment qwen_timeline_pilot_v1 `
  --variants 4 `
  --seed 260825
```

Run that command from `synthetic-generation`. Before scaling, adapt the generated variants into
event sources for at least 30 plans (roughly 120 voiced events), record the immutable model
revision, peak VRAM, failures, RTF, audio hours/GPU-hour, and blinded naturalness/semantic-fit
ratings. Do not use `--auto-place`: author planned event intervals and duration-fit the successful
tracks through this pipeline.

## Evaluation and training experiment

Freeze the current real validation/test corpus revisions and external causal benchmark inventory
before training. Synthetic validation is diagnostic only. Use identical optimizer budgets and
three seeds for:

1. real-only baseline;
2. synthetic-only training, evaluated only on real validation and external benchmarks;
3. mixed batches at 10%, 25%, and 50% synthetic, stratified by completion class;
4. synthetic pretraining followed by real-only fine-tuning at 10%, 25%, 50%, and 100% of the real
   training step budget.

Select checkpoints using real validation completion AUROC/AP and operating-point false-take versus
missed-handoff cost. Report real-test results once after locking the winner. Also report causal
latency, calibration, clean/overlap/backchannel/interruption slices, source-dataset slices, and
synthetic-model/voice holdouts. Reject scaling if synthetic data improves synthetic metrics but
degrades the pinned real gate, external causal benchmarks, calibration, or false-take rate.

Before a 10k-window job, pass these gates on 30-100 pilot plans:

- no missing/duplicate voiced-event sources or changed source hashes;
- exact 20.0-second, 16 kHz independent channels and 250 label frames;
- no label derivation from ASR/VAD and no synthetic validation/test assignments;
- manual blinded review for intelligibility, interaction plausibility, timing fit, speaker
  separation, artifacts, and voice diversity;
- scenario/latency histogram targets within tolerance and no plan/text leakage into real
  validation/test or external benchmarks;
- complete model, voice, license, consent, seed, and augmentation provenance.
