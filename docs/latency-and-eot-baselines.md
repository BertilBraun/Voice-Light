# Latency Budget And End-Of-Turn Baselines

## Latency Reality Check

The optimistic component numbers are not additive in a simple way, because some work can be overlapped. Still, the hard part is not only model inference. The hard part is deciding that the user has actually yielded the turn without cutting them off during a pause.

Approximate optimistic local budget from true user end-of-turn to first playable assistant audio:

| Stage | Aggressive Target | More Realistic Early Target | Notes |
| --- | ---: | ---: | --- |
| ASR chunk finalization and stable text update | 80-160 ms | 160-300 ms | Even an 80 ms ASR chunk mode still needs chunk arrival, encoder/RNNT work, token stabilization, and policy update. |
| Turn-taking model inference | 10-70 ms | 30-100 ms | Existing local turn models can be fast; the policy wait is usually larger than inference. |
| Endpointing policy delay | 100-300 ms | 300-700 ms | This is the main tradeoff. Lower delay means more false cutoffs. |
| LLM first token | 20-100 ms | 50-200 ms | With a small warm model. Prefix caching helps conversation history, but the current user utterance still needs prefill unless it has been incrementally committed. |
| TTS first playable audio | 75-150 ms | 150-300 ms | Based on optimistic local streaming TTS claims; must be measured. |
| Playback scheduling/buffer | 20-50 ms | 30-100 ms | Depends on audio stack and buffering policy. |
| Total | 305-830 ms | 720-1,700 ms | The low end requires aggressive endpointing and strong overlap. |

The practical goal should be:

- Under 500 ms when the turn is unambiguous and the policy can be aggressive.
- 500-900 ms for reliable everyday behavior with low false cutoffs.
- More than 900 ms for ambiguous pauses where waiting is preferable to interrupting.

The correct metric is not only average response latency. It should be a Pareto curve over:

- False cutoff rate.
- Delay after true end-of-turn.
- Interruption stop latency.
- Backchannel false interruption rate.

The live pipeline logs one `voice first audio latency` record per generation. It separates:

- ASR finalization after `vad.stopped`.
- Turn commit and teardown-barrier delay.
- Qwen generation start to first text delta.
- Qwen generation start to the first complete word accepted by TTS.
- First complete word to first PCM.
- Kyutai worker tokenization, delayed-stream LM steps, Mimi decoding, and model-step count.
- Generation start to first PCM.

The browser worklet reports `playback.started` when it writes the first assistant sample into an
output render quantum. The server logs both first-audio-send to playback acknowledgement and
generation start to playback acknowledgement. This acknowledgement includes the return network
trip and should not be interpreted as a one-way playback delay.

Run the standalone Kyutai benchmark from the repository root:

```bash
uv run python -m deployment.compute.benchmark_tts --runs 5
```

The benchmark reports first-word-to-PCM latency and the same worker-internal breakdown. It queues
text immediately and therefore measures TTS rather than LLM word-formation latency.

## LLM Prefill Assessment

Prefix caching is still useful, but it probably should not be the first complex engineering bet.

For turn `n`, the stable historical prefix can be reused:

- System prompt.
- Conversation turns that have already completed.
- Tool/schema text if present.
- Possibly a conversation summary.

The current user utterance is the part that changes. If typical user turns are short, the cost of prefilling the current user text may be smaller than the cost and complexity of a custom early-prefill protocol. The first implementation should therefore prioritize a fast runtime and clean cache reuse over speculative cache surgery.

Recommended first implementation:

1. Use vLLM or SGLang with normal prefix caching.
2. Keep the prompt byte-identical across turns to maximize cache hits.
3. Do not inject unstable timestamps, confidence values, or changing metadata into the cached prefix.
4. Measure cold prompt, cached-history prompt, and cached-history-plus-current-stable-prefix separately.
5. Add incremental current-utterance prefill only if measurement shows it saves enough latency to justify the complexity.

## Swappable Model Boundaries

ASR is the hardest component to swap because the turn-taking adapter may be trained on its intermediate states.

LLM and TTS should be intentionally swappable:

- LLM interface: prompt state, generation request, streamed text tokens, cancellation.
- TTS interface: streamed text input, audio chunk output, flush, cancel, reset voice/session.
- Metrics interface: TTFT, tokens/sec, TTFA, audio real-time factor, cancellation latency, VRAM.

ASR should still use a stable interface, but the adapter will likely be model-specific:

- Audio chunks.
- Stable transcript events.
- Partial transcript events.
- Encoder/RNNT feature frames.
- Per-chunk timestamps.
- Turn-taking probabilities.

## Runnable End-Of-Turn Baselines

These are the first models worth running before training the ASR-attached adapter.

| Candidate | Input | Local? | What It Predicts | Why It Matters |
| --- | --- | --- | --- | --- |
| [Pipecat Smart Turn v3](https://github.com/pipecat-ai/smart-turn) / [HF model](https://huggingface.co/pipecat-ai/smart-turn-v3) | Raw PCM audio | Yes | Probability that the user completed their turn. | Best immediate local open baseline. Audio-native, BSD-2-Clause, fast CPU/GPU inference, open weights/training code. |
| [Pipecat Smart Turn v2](https://huggingface.co/pipecat-ai/smart-turn-v2) | Raw waveform | Yes | Single completion probability. | Older but easy baseline. Model card reports about 12 ms to analyze 8 s of audio on an NVIDIA L40S. |
| [LiveKit Turn Detector v1-mini](https://docs.livekit.io/agents/logic/turns/turn-detector/) | Raw audio | Yes through LiveKit Agents | End-of-turn decision. | Practical baseline inside LiveKit. The full v1 model is hosted; v1-mini is the local lightweight fallback. |
| [LiveKit text turn detector](https://huggingface.co/livekit/turn-detector) | Transcript text | Yes | Contextual EOU from transcribed speech. | Useful semantic-only baseline. It complements VAD but cannot see prosody. |
| [Voice Activity Projection](https://github.com/ErikEkstedt/VoiceActivityProjection) | Stereo or mono audio | Yes, research repo | Future voice activity distribution and turn-shift probabilities. | Best research baseline for richer turn dynamics, but less drop-in than Smart Turn. |
| [TEN Turn Detection](https://github.com/TEN-framework/ten-turn-detection) | Transcript text | Yes | `finished`, `unfinished`, or `wait`. | Interesting text-policy baseline because it has a third wait state. Heavier because it is Qwen2.5-7B-based. |
| [Turnsense](https://github.com/latishab/turnsense) | Transcript text | Yes | Binary EOU. | Lightweight semantic baseline based on SmolLM2-135M and ONNX, but likely weaker and STT/punctuation-sensitive. |
| [Silero VAD](https://github.com/snakers4/silero-vad) + silence hysteresis | Audio | Yes | Speech/non-speech only. | Required naive floor. Not a turn-taking model, but it gives the baseline that learned models must beat. |
| [LiveKit EOT Bench](https://github.com/livekit/eot-bench) | Benchmark harness | Yes | Evaluation, not a detector. | Useful for measuring latency vs false cutoff. Public results show endpointing delay dominates. |

Important benchmark signal: LiveKit EOT Bench reports that at a 5% false-cutoff budget, English endpointing delay is hundreds of milliseconds even for strong models. It reports 543 ms for LiveKit Turn Detector v1 and 739 ms for Smart Turn v3.2 at that operating point. At a 10% false-cutoff budget, the numbers are lower, but the user experience risk rises.

## First Baseline Plan

1. Run Silero VAD plus fixed silence as the naive floor.
2. Run Pipecat Smart Turn v3 locally on the same pause candidates.
3. Run LiveKit v1-mini locally if it can be used without coupling the whole stack to LiveKit.
4. Run LiveKit text turn detector, Turnsense, and optionally TEN on ASR transcript snapshots.
5. Run VAP on stereo or separated-channel datasets to understand richer hold/shift/backchannel behavior.
6. Evaluate with LiveKit EOT Bench and with a small manually audited dataset from CANDOR/EgoCom/EasyCom/DiPCo.
7. Use the best external baseline as the bar for the frozen-ASR adapter.
