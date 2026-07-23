# Turn-Taking Adapter Training Plan

## Decision

Train a causal adapter on the frozen encoder states of
[`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b).
It is the best fit for the English-first product because it is a native cache-aware streaming
FastConformer-RNNT, supports 80, 160, 560, and 1,120 ms operating points without retraining, and
exposes encoder hidden states and streaming caches in Transformers 5.13. Use 160 ms
(`lookahead_tokens=1`) for the first product experiment and compare 80 and 560 ms during evaluation.

Pin the checkpoint revision and retain its NVIDIA Open Model License with release artifacts. If the
product becomes multilingual, rerun the same adapter experiment on
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).
NVIDIA recommends the English checkpoint for English-only use.

## Why This Backbone

| Model | Temporal behavior | Adapter access | Decision |
| --- | --- | --- | --- |
| Nemotron streaming 0.6B | Native non-overlapping cache-aware chunks; selectable bounded lookahead | Per-layer encoder states, attention K/V cache, convolution cache, RNNT state | Train on this |
| Parakeet TDT 0.6B | Excellent batch ASR; documented streaming path is buffered with overlap | Encoder accessible, but training/runtime timing differs | Transcript baseline |
| Canary 1B | FastConformer plus autoregressive attention decoder; primarily offline/long-form | Encoder accessible, but no native subsecond stream | Quality reference |
| WhisperX large-v3 | Repeated Whisper windows plus later wav2vec alignment | Features exist, but timestamps and online state are different pipelines | Robustness reference |
| Qwen3-ASR | Streaming text path, but weaker timestamp/feature-tap contract | Less direct for a frozen acoustic adapter | Later challenger |

Nemotron has two different recurrent dimensions:

1. The FastConformer encoder is not an RNN. During streaming, bounded self-attention K/V and causal
   convolution tails are cached. A new non-overlapping chunk reuses those activations, so prior audio
   is not encoded again. Right context determines algorithmic lookahead. This is the adapter's
   acoustic/prosodic input.
2. The RNNT prediction network is recurrent over emitted non-blank tokens. It represents label
   history and combines it with the current encoder frame in the joint network. It helps incremental
   transcription, but must not be the adapter's only signal because silence, pitch, lengthening, and
   non-lexical vocalizations often occur before a token is emitted.

The adapter introduces a third, task-specific state: a unidirectional GRU carried between chunks.
Both the ASR caches and GRU state must persist for a session, detach at truncated-BPTT boundaries,
and reset on a stream discontinuity. Offline whole-file extraction is useful for pipeline smoke
tests but is not a valid streaming-latency or causality result. The architecture follows the
stateful-cache mechanism in [Stateful Conformer](https://arxiv.org/abs/2312.17279) and the efficient
encoder design in [Fast Conformer](https://arxiv.org/abs/2305.05084).

## Manifest And Sample Contract

The intended materialized training corpus will be a private Hugging Face dataset;
it has not been published by this repository. Once a prepared corpus is available,
the trainer can load Parquet shards at startup and download only the user-side FLAC
needed by each sampled crop. `hf_hub_download` provides a persistent,
content-addressed disk cache and cross-process download locking; pass
`--hub-cache-directory` to place that cache on the training volume. The corpus
format uses 20-second crops, 250 80-ms frames, and `-1.0` to mask individual
targets. The JSONL manifest format below remains available through `--manifest`
for local experiments.

### Intended source preparation

The first planned sources are [Mundo TurnBench dev](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev)
and [MagicHub Multi-stream Spontaneous Conversation Training Datasets (English)](https://magichub.com/datasets/multi-stream-spontaneous-conversation-training-datasets_english/).
They map to generic local roots, not source-specific identifiers:

| Dataset ID | Preparation | Annotation handling |
| --- | --- | --- |
| `dataset_2` | Prepare the locally obtained MagicHub archive into generic samples and speaker tracks. | Preserve only approved provenance needed for processing; ASR enrichment is deferred until compute is available. |
| `dataset_3` | Import Mundo source FLAC tracks and the supplied annotation package after access is available. | Normalize the supplied transcript and timing annotations into the training schema; do not run ASR for the initial import. |

Raw source data, derived audio, transcripts, annotations, credentials, and private training
artifacts remain outside Git and are not published by this repository. Maintain private
provenance for every materialized sample, including its source URL, source version, and applicable
attribution or handling requirements. Before importing or using either source, check its current
access, license, attribution, privacy, and handling terms; this document does not determine what
those terms permit.

Use conversation-disjoint JSONL manifests. Never put windows from one conversation or speaker into
different splits. Every line validates as `TurnTakingSample` in
`app/training/turn_taking/schema.py` and has this shape:

```json
{
  "sample_id": "candor-0001-speaker-a-0042",
  "conversation_id": "candor-0001",
  "target_speaker_id": "speaker-a",
  "target_audio_path": "audio/candor-0001-speaker-a.wav",
  "annotation_reference_audio_path": "audio/candor-0001-speaker-b.wav",
  "sample_rate_hz": 16000,
  "context_start_seconds": 120.0,
  "decision_start_seconds": 124.0,
  "decision_end_seconds": 140.0,
  "assistant_speech_spans": [
    {"start_seconds": 126.2, "end_seconds": 131.8}
  ],
  "decisions": [
    {
      "time_seconds": 132.4,
      "yield_probability": 0.72,
      "primary_reliability": null,
      "event_distribution": {
        "turn_completion": 0.64,
        "continuation_pause": 0.20,
        "backchannel": 0.08,
        "interruption": 0.03,
        "other": 0.05
      },
      "event_reliability": null,
      "future_user_activity": [false, false, true, true]
    }
  ],
  "words": [],
  "source_dataset": "CANDOR",
  "source_license": "record exact dataset grant/version"
}
```

The normal item is 20 seconds: four seconds of burn-in followed by 16 supervised seconds. Audio is
mono float32 at 16 kHz. Window endpoints and all annotations use seconds from the original recording.
Training targets are aligned to emitted encoder frames; the current prototype initially rasterizes
at 80 ms and nearest-aligns to the returned encoder length. `decisions` contains one HOLD/YIELD
target for every supervised frame; the single entry above is abbreviated for readability. Manifest
loading fails if any supervised frame lacks a primary target. Event distributions remain nullable
because that auxiliary loss applies only at event candidates.

`annotation_reference_audio_path` is label-generation evidence only. Its waveform must never enter
the ASR or adapter. The binary `assistant_speech_spans` state is different: it is a known runtime
condition produced by assistant playback control, so it is rasterized across the full window and
fed directly to the adapter. This lets identical short user utterances be interpreted differently
while the assistant is speaking without letting the model hear future partner audio. Mixed
recordings still require source separation. Keep source, license, annotation origin, and confidence
on every sample. Mask unreliable labels instead of inventing them.

## Labels

Supervise the primary target densely after burn-in. The primary target is one soft scalar,
`yield_probability`: zero means the user holds the floor and one means the user yields it. Ordinary
user speech is HOLD. Ordinary silence or other-speaker time is YIELD. Backchannel-like user speech
stays mostly YIELD because it does not claim the floor. A gap between two user segments uses the
continuous same-turn merge belief: `p(HOLD)=merge_confidence` and
`p(YIELD)=1-merge_confidence`. It is not the assistant playback state.

Keep target probability and reliability independent. `yield_probability=0.4` describes an ambiguous
HOLD/YIELD identity. `primary_reliability=0.4` says the annotation itself is weak. A null reliability
means the current data has no independent measurement; the baseline uses the explicitly configured
neutral weight `unmeasured_reliability_weight=1.0` and reports those examples separately. It never
copies target confidence into reliability.

The materialized corpus provides five independent auxiliary targets. They are
not mutually exclusive: a backchannel may occur during a pause. Train the
corresponding five logits with masked binary cross-entropy:

- turn completion, continuation pause, assistant backchannel, non-floor feedback, and floor take;
- Hard future user-activity targets for `[0, 200)`, `[200, 500)`, `[500, 1000)`, and
  `[1000, 1500)` milliseconds. A null bin is masked.

Do not add current user VAD, silence duration, assistant playback as a target, nested endpoint
horizons, or time-to-shift heads initially. They are redundant or create shortcut risk. Known
assistant playback remains an input condition. A known future
continuation is valid HOLD supervision even when the pause is long; lower reliability only when the
underlying timing/identity annotation is unreliable, not merely because the example is difficult.

Future activity borrows the central idea of [Voice Activity Projection](https://arxiv.org/abs/2205.09812):
learn what both participants are likely to do rather than treating silence as a decision. The
ASR-integrated precedent jointly optimized ASR and turn-taking and reported that distinguishing
thinking pauses from completion was critical
([Chang et al., 2022](https://arxiv.org/abs/2208.13321)). We freeze ASR first to keep the experiment
cheap and attributable; joint fine-tuning is a later ablation.

Use a 100-200 ms zero-loss uncertainty band around boundaries after the first annotation audit.
Auxiliary gaps and response latency may later use masked `log1p` Smooth-L1 heads. They are omitted
from the first code path until those measurements exist reliably.

## Data Scale And Sampling

The plumbing pilot needs at least 100 hours, 20,000 unique shifts, 20,000 hard pause/hold negatives,
5,000 backchannels, and 2,000 each interruptions and overlaps. It validates learning and data
quality; it is not a production claim.

The v1 target is 300-600 hours of speaker-separated natural conversation with these unique event
counts before augmentation or repeated windows:

| Category | Minimum unique events |
| --- | ---: |
| Genuine turn shifts | 100,000 |
| Intra-turn holds/pauses | 100,000 (25,000 at least 500 ms) |
| Listener backchannels | 40,000 |
| Interruption attempts | 20,000 |
| Cooperative overlaps | 20,000 |
| Competitive overlaps | 20,000 |
| Laughter/non-speech vocalizations | 20,000 |
| Ordinary active/silent contexts | 100,000 |

Compose batches as roughly 25% shift, 25% hard hold, 15% backchannel, 15% overlap/interruption, and
20% background. Count a source event once per epoch; augmentations are not new unique samples. Cap a
single conversation or speaker at 1% of sampled windows. Reserve 10% of conversations for validation
and 10% for test, and retain EasyCom/DiPCo as cross-domain tests when CANDOR is the main training
source. Bootstrap confidence intervals by conversation, never frame.

## Augmentation

The implemented first pass uses random gain (-12 to +6 dB), additive noise (5-30 dB SNR), and
short packet dropout. Extend it with measured room impulse responses, mild EQ/band limiting,
telephony codecs, low-probability clipping, and realistic assistant echo at -30 to -10 dB.

Speed perturbation from 0.9-1.1 is allowed only when every word, event, and future horizon is warped
with the waveform. Never mask across a labeled boundary, reverse audio, splice unrelated turns, or
crop away the interval used to build a future target. Keep validation/test clean and add fixed noise,
reverb, codec, and echo robustness suites.

## Adapter Architecture

Tap encoder layers 6, 12, 18, and 24. Each 1,024-dimensional stream receives its own LayerNorm and
32-dimensional projection. Concatenate the four taps, fuse to 64 with SiLU and dropout 0.1, then
apply two residual causal depthwise-separable convolution blocks (kernel 5, dilations 1 and 2).
Append the known binary assistant-speaking state and feed the resulting 65 dimensions to a
single-layer unidirectional GRU with 64 hidden units. Three linear heads produce one YIELD logit,
five event logits, and four future-activity logits. The default adapter has about 183,000 trainable
parameters while retaining all four encoder taps and both causal convolution blocks.

Run the frozen encoder in evaluation mode, detach taps, and optimize only the adapter. The current
Transformers wrapper exposes hidden states without forward hooks. Production streaming extraction
must carry the returned attention and convolution caches; that stateful replay is the next runtime
integration gate.

## Optimization And Duration

- AdamW, learning rate `3e-4`, betas `(0.9, 0.98)`, weight decay `0.01`.
- Effective batch 16 sequences (physical batch 4, accumulation 4), each with 16 supervised seconds.
- Linear warmup for 2,000 optimizer steps, cosine decay to `3e-5`, global gradient clipping at 1.0.
- BF16 encoder/adapter where supported and FP32 loss computation on the production GPU.
- Smoke test: 100 steps. Pilot: 10,000 steps. Full maximum: 50,000 optimizer steps.
- Validate every 1,000 steps; stop after eight validations without improvement, but not before 15,000.

Fifty thousand optimizer steps expose about 3,556 supervised hours: roughly 12 passes over 300 hours
or six over 600 hours. Define iteration counts as optimizer steps, not microbatches. Compute and log
the actual epoch count from the filtered manifest because event balancing changes samples per epoch.

The loss is `1.0 soft HOLD/YIELD BCE + 0.25 auxiliary BCE + 0.25 future-activity BCE`.
Normalize every task by its own reliability-weighted valid targets. Derive any sampling or class
weights from the training manifest after the first audit; do not silently turn target ambiguity into
sample reliability.

Run with:

```powershell
uv run python -m app.training.turn_taking.cli .\artifacts\turn-taking\adapter.pt `
  --hub-cache-directory D:\voice-light-hf-cache --max-steps 100
```

## Evaluation Gates

Report macro F1 and per-class PR-AUC; horizon AUROC, PR-AUC, Brier score, and calibration error; and
per-event F1. At the policy level report false starts/hour, missed shifts/hour, cutoff rate during
holds by pause duration, boundary latency p50/p90/p95, backchannel confusion, and interruption/yield
latency. Tune thresholds and three-frame hysteresis on validation only.

Replay held-out conversations through the real cache API at 80, 160, and 560 ms. Report adapter real
time factor, p95 compute/chunk, peak memory, and offline-versus-streaming probability drift. Slice by
dataset, speaker, SNR, channel, accent/dialect where licensed, turn length, pause length, overlap,
backchannel, and ASR WER. Compare against the repository's silence/VAD, transcript-gap, Smart Turn,
LiveKit, and VAP baselines. A model is not ready based on frame accuracy alone; it must improve the
latency-versus-false-cutoff Pareto curve on locked conversation-level test data.

## Primary References

- [Nemotron Speech Streaming English model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [Transformers Nemotron streaming documentation](https://huggingface.co/docs/transformers/main/model_doc/nemotron_asr_streaming)
- [NVIDIA cache-aware streaming model documentation](https://docs.nvidia.com/nemo-framework/user-guide/26.02/nemotoolkit/asr/models.html)
- [NVIDIA streaming inference documentation](https://docs.nvidia.com/nemo/speech/nightly/asr/inference.html)
- [Stateful Conformer with Cache-based Inference](https://arxiv.org/abs/2312.17279)
- [Fast Conformer](https://arxiv.org/abs/2305.05084)
- [Turn-Taking Prediction for Natural Conversational Speech](https://arxiv.org/abs/2208.13321)
- [Voice Activity Projection](https://arxiv.org/abs/2205.09812)
- [Parakeet TDT 0.6B v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Canary 1B v2 model card](https://huggingface.co/nvidia/canary-1b-v2)
- [Whisper paper](https://arxiv.org/abs/2212.04356) and
  [official implementation](https://github.com/openai/whisper)
