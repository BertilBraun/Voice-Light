# Voice Model Benchmark Results

Results collected on 2026-07-16 on one NVIDIA RTX 4090. GPU models were run sequentially with
the Voice Light compute service stopped unless stated otherwise.

## Language Model

Models:

- `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.

Each result contains 18 generations across six single-turn and multi-turn conversational cases.
Both models used BF16, SDPA, thinking disabled, temperature `0.6`, top-p `0.9`, and a 96-token
limit. The benchmark measured the actual first generated token separately from the first printable
complete word.

| Model | Prompt | First token p50 | First word p50 | First word p90 | Total p50 | Words p50 | Tokens/s p50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | Current | 44 ms | 72 ms | 101 ms | 984 ms | 25 | 34.8 |
| Qwen3 0.6B | Cooperative listener | 32 ms | 73 ms | 103 ms | 842 ms | 24 | 34.7 |
| Qwen3 1.7B | Current | 45 ms | 81 ms | 101 ms | 1,703 ms | 50 | 35.4 |
| Qwen3 1.7B | Cooperative listener | 31 ms | 86 ms | 101 ms | 796 ms | 26 | 35.1 |
| Qwen3 1.7B | Spoken conversation | 32 ms | 86 ms | 112 ms | 992 ms | 27 | 34.4 |

The 0.6B model reduced first-word latency by only about 9-14 ms and did not improve steady
throughput. Both models have 28 transformer layers, so the smaller width does not remove the
serial-depth and framework costs that dominate this batch-one workload.

The 0.6B outputs were materially worse. Common failures included shallow paraphrasing, generic
support-agent language, repeated `Let me know if...` endings, poor repair after a correction, and
weak multi-turn advancement. Prompting shortened its answers but did not recover the lost semantic
or social quality.

The cooperative-listener prompt cut the 1.7B median response from 50 to 26 words without changing
first-word latency materially. The spoken-conversation prompt also shortened responses, but
occasionally produced task restatement and visible deliberation such as `Let me think...`. It
should not be used unchanged.

Recommendation:

- Keep Qwen3 1.7B.
- Refine and live-test the cooperative-listener prompt.
- Do not switch to 0.6B for a roughly 10 ms first-word improvement.
- Evaluate response length and conversational behavior separately from first-token latency.

Run the benchmark with:

```bash
uv run python -m deployment.compute.benchmark_llm \
  --model Qwen/Qwen3-1.7B \
  --prompt-style cooperative-listener \
  --runs 3
```

## Text-To-Speech

### Kyutai TTS 1.6B

Isolated Voice Light benchmark, five warm runs:

| Metric | Result |
|---|---:|
| First word to first PCM, median | 469 ms |
| First chunk wall time, median | 485 ms |
| Real-time factor | 0.285-0.293 |
| Model steps before first PCM | 19 |
| Model load | 13.4 s |

The live integrated service previously measured approximately 640-654 ms from first word to first
PCM, leaving roughly 150-180 ms of service scheduling and integration overhead beyond the isolated
model.

### VoXtream2

Pinned implementation: `herimor/voxtream` at
`8ec2d62159dae4716ae7058827244a962d40603c`.

Official benchmark:

| Mode | First packet | Real-time factor |
|---|---:|---:|
| Eager | 102 ms | 0.270 |
| Compiled | 94 ms | 0.224 |

Fixed-prompt incremental-word benchmark:

| Delay between available words | Wall first packet | Model-internal first packet | RTF |
|---|---:|---:|---:|
| 0 ms | 113 ms | 56 ms | 0.217 |
| 20 ms | 112 ms | 56 ms | 0.215 |
| 50 ms | 132 ms | 76 ms | 0.217 |
| 100 ms | 189 ms | 133 ms | 0.219 |

VoXtream2 genuinely consumes additional text while emitting audio. Its public generator accepts a
word generator and proceeds with three-phoneme lookahead. It therefore maps directly onto Voice
Light's current `add_word()` and `finish_input()` synthesis boundary.

The fixed-prompt result is the relevant production shape because Voice Light uses one preselected
voice. A worker integration still needs to measure prompt-cache preparation, cancellation,
word-boundary quality, short-utterance behavior, and sustained stability.

### Qwen3-TTS 12Hz 0.6B

Tested model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, speaker `Ryan`, official `qwen-tts==0.1.1`
Python API, BF16 and SDPA.

| Metric | Result |
|---|---:|
| Model load | 41.5 s |
| Peak allocated VRAM | 2,059 MB |
| Complete-call RTF | 1.66-1.73 |
| Complete-call wall time | 11.2-15.4 s |

FlashAttention was not installed, so these throughput figures are not the model's best possible
offline result. That does not change the architectural result: the released Python API accepts a
complete string, completes autoregressive generation, decodes the complete code sequence, and
returns a complete waveform. It exposes no local first-audio yield.

The saved first run for the one-word input `Right.` contained 10.88 seconds of non-silent audio,
which is unacceptable for a backchannel without further sampling and termination work.

Recommendation:

- Build a VoXtream2 worker prototype as the primary low-latency TTS experiment.
- Keep Kyutai as the current quality reference.
- Treat Qwen3-TTS as an offline quality challenger until an official incremental local API exists.
- Do not interpret Qwen's reported 97 ms architecture result as obtainable from the released
  Python wrapper.

## Expected Pipeline Impact

With Qwen3 1.7B producing a complete word in roughly 80-100 ms:

- Current isolated Kyutai implies roughly 550-585 ms from LLM start to first PCM.
- VoXtream2 implies roughly 190-230 ms from LLM start to first PCM with incremental words.
- The expected saving is approximately 350 ms before service-integration effects.

The larger remaining cost is still endpointing. Replacing TTS improves responsiveness
substantially, but does not by itself solve overlap, backchannels, floor control, or conversational
naturalness.
