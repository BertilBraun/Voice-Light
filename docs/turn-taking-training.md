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

Use conversation-disjoint JSONL manifests. Never put windows from one conversation or speaker into
different splits. Every line validates as `TurnTakingSample` in
`app/training/turn_taking/schema.py` and has this shape:

```json
{
  "sample_id": "candor-0001-speaker-a-0042",
  "conversation_id": "candor-0001",
  "target_speaker_id": "speaker-a",
  "target_audio_path": "audio/candor-0001-speaker-a.wav",
  "reference_audio_path": "audio/candor-0001-speaker-b.wav",
  "sample_rate_hz": 16000,
  "context_start_seconds": 120.0,
  "decision_start_seconds": 124.0,
  "decision_end_seconds": 140.0,
  "events": [
    {
      "event_id": "shift-91",
      "event_type": "turn_shift",
      "start_seconds": 132.4,
      "end_seconds": 132.56,
      "confidence": 1.0,
      "source": "human"
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
at 80 ms and nearest-aligns to the returned encoder length.

`reference_audio_path` is label-generation evidence only. It must never enter the ASR or adapter.
Otherwise the model can hear the other speaker begin and mistake detection for prediction. Mixed
recordings require source separation or samples whose cutoff precedes partner onset. Keep source,
license, annotation origin, and confidence on every sample/event. Mask unreliable labels instead of
inventing them.

## Labels

The main mutually exclusive frame policy is:

- `wait`: target speaker has the floor, including a hesitation or intra-turn pause.
- `take_turn`: a genuine transition-relevance point/turn shift.
- `may_backchannel`: a short listener response that does not claim the floor.
- `yield`: the other participant is trying to take the floor while the agent is speaking.

Auxiliary heads prevent the main classifier from collapsing every silence into an endpoint:

- Future target and partner activity for `[0, 0.5)`, `[0.5, 1.0)`, and `[1.0, 2.0)` seconds.
- Nested `turn_ends_within_0.5s`, `turn_ends_within_1s`, and `turn_ends_within_2s` targets.
- Time-to-next-shift buckets: up to 250 ms, 250-500 ms, 500 ms-1 s, 1-2 s, or over 2 s.
- Backchannel, interruption, cooperative overlap, competitive overlap, laughter, and non-speech
  vocalization multi-label targets.

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
128-dimensional projection. Concatenate the four taps, fuse to 256 with SiLU and dropout 0.1, then
apply two residual causal depthwise-separable convolution blocks (kernel 5, dilations 1 and 2) and a
two-layer unidirectional GRU with 256 hidden units. Independent linear heads produce the targets
above. The adapter is only a few million parameters and has bounded convolutional cost plus constant
recurrent memory.

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

The loss is `1.0 policy CE + 0.5 future BCE + 0.5 horizon BCE + 0.25 time-bucket CE + 0.2 event BCE
+ 0.05 horizon-monotonicity`. Normalize every task by its own confidence-weighted valid frames. Derive
clipped class weights from the training manifest after the first audit.

Run with:

```powershell
uv run python -m app.training.turn_taking.cli .\data\turn-taking\train.jsonl `
  .\artifacts\turn-taking\adapter.pt --max-steps 100
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
