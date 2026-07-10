# Voice Agent Research Matrix

This document tracks candidate technologies and datasets for the streaming cascaded voice agent.

The current selection criteria are:

- Local or single-machine deployment first.
- True streaming or low-latency incremental behavior.
- Ability to expose or reuse intermediate state.
- Clear licensing and reproducibility.
- Fit with a frozen-ASR turn-taking adapter.
- Modern datasets and models, not legacy defaults unless used as baselines.

## ASR Candidates

| Rank | Candidate | Fit | Streaming And Latency | Feature Access | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | [NVIDIA Nemotron Speech Streaming English 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | Best English-first ASR backbone for turn-taking adapter research. | Native cache-aware FastConformer-RNNT with 80, 160, 560, and 1120 ms chunk modes. | Strong through NeMo and Transformers `AutoModelForRNNT`; encoder/RNNT internals should be accessible. | NVIDIA Open Model License. Use 160 ms chunks for interaction tests and 560 ms chunks for quality comparison. |
| 2 | [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | Best multilingual current option. | Same cache-aware FastConformer-RNNT family, with 80, 160, 320, 560, and 1120 ms chunks. | Strong through NeMo/Transformers. | OpenMDW-1.1. Prefer the English model when English is the primary target. |
| 3 | [Qwen3-ASR 0.6B / 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | Strong modern ASR quality candidate. | Supports streaming/offline usage through vLLM. | Weak for this adapter plan because streaming does not expose timestamps and feature taps as cleanly. | Apache-2.0. Good second experiment for quality/multilingual behavior, less ideal as the frozen-ASR feature backbone. |
| 4 | [Moonshine Streaming](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) / [Moonshine](https://github.com/moonshine-ai/moonshine) | Hackable low-latency research candidate. | Sliding-window encoder with 80 ms lookahead. | Good because the stack is small and modifiable. | MIT for core English/code according to current research; verify exact model. Current Transformers path may not provide the fully efficient streaming implementation. |
| 5 | [sherpa-onnx online Zipformer / Transducer](https://github.com/k2-fsa/sherpa-onnx) | Best deployment/runtime fallback. | True online ONNX ASR with many real-time examples. | Moderate; ONNX runtime hides internals unless exports are modified. | Good for product packaging, less clean for training an adapter on intermediate features. |
| 6 | [FunASR](https://github.com/modelscope/FunASR) | Practical service ecosystem. | Has streaming models and server paths, but English-first streaming is less clean than Nemotron. | Good in toolkit form. | Stronger if Mandarin/Asian-language coverage becomes important. |

Current recommendation: start with Nemotron Speech Streaming English 0.6B. Whisper-style streaming, Parakeet, Canary, and Kaldi/Vosk-era stacks should stay as baselines or fallbacks, not the primary turn-taking backbone.

## LLM And Runtime Candidates

| Rank | Candidate | Fit | Prefix Cache / Prefill | Streaming | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) + [vLLM](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) or [SGLang](https://docs.sglang.io/) | Best first LLM experiment. | vLLM automatic prefix caching or SGLang RadixAttention reuse exact shared prefixes. | OpenAI-compatible streaming. | Apache-2.0, 32k context, GQA, broad runtime support. Run with `enable_thinking=False` or `/no_think` for low-latency voice responses. |
| 2 | [LiquidAI LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) + llama.cpp or vLLM | Speed and memory challenger. | Runtime-dependent prefix cache. | Runtime-dependent streaming. | LFM Open License v1.0. Newer than Qwen3 and potentially very fast, but may be weaker on knowledge-intensive tasks. |
| 3 | [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) + vLLM/SGLang | Quality challenger under 4B. | Inherits runtime prefix cache behavior. | OpenAI-compatible streaming through serving runtime. | Apache-2.0, 64k trained context, dual think/no-think. More VRAM and slower TTFT than 1-1.7B models. |
| 4 | [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) + vLLM/SGLang | Strong reasoning fallback. | Inherits runtime prefix cache behavior. | Runtime streaming. | MIT, 3.8B, 128k context. Likely too large for the fastest voice-loop path but useful as a quality reference. |
| 5 | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) + vLLM/llama.cpp | Mature fallback only. | vLLM/llama.cpp prefix or slot cache. | Runtime streaming. | Apache-2.0. No longer the default; useful if Qwen3 template/thinking behavior causes issues. |

Important constraint: mainstream runtime caches are exact-token prefix caches. The orchestration should treat ASR text as `stable_prefix + volatile_tail`, commit only stable tokens into the durable prefix, and accept recomputation from the divergence point when ASR revises unstable text.

The first benchmark should compare Qwen3-1.7B against LiquidAI LFM2.5-1.2B for TTFT, prefix-cache hit impact, VRAM, tokens/sec, and subjective short-turn conversation quality. [TensorRT-LLM KV cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html) is worth testing later if the prototype becomes locked to NVIDIA production serving, but vLLM/SGLang should be faster for research iteration.

## TTS Candidates

| Rank | Candidate | Fit | Streaming Mechanism | Latency Signal | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 12Hz 0.6B / 1.7B | Best current local first bet. | Designed for streaming TTS, with local package and Web UI. | Reported first-packet latency down to 97 ms. | Apache-2.0, English support, voice cloning/design/instruction control. Very new, so streaming API claims need hands-on verification. |
| 2 | [CosyVoice 2 / 3](https://github.com/FunAudioLLM/CosyVoice) | Most mature practical local option. | Text-in/audio-out bi-streaming, server paths, vLLM support. | Claimed latency as low as 150 ms. | Apache-2.0 repo, active ecosystem, 0.5B family. Use as maturity fallback if Qwen3-TTS is not usable enough. |
| 3 | [Kyutai TTS](https://kyutai.org/tts/) / [Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) | Strong architecture fit for voice agents. | Delayed streams modeling can begin audio before full text is available. | Needs measurement locally. | 1.6B, PyTorch research stack and Rust websocket server. Larger and less plug-and-play than CosyVoice. |
| 4 | [VoXtream2](https://github.com/herimor/voxtream) | Best research architecture for true incremental TTS. | Full-stream text/phoneme input to audio frames. | Reported 74 ms first-packet latency and 4x realtime. | Smaller ecosystem, but highly relevant for barge-in and incremental generation research. |
| 5 | [ZONOS2](https://github.com/Zyphra/ZONOS2) | Experimental quality/voice cloning candidate. | Local server with documented `stream: true`. | Needs measurement. | MIT, English tier-1, CUDA server, Mini-SGLang backend. New and experimental. |
| 6 | [Dia2](https://github.com/nari-labs/dia2) | Conversational speech generation candidate. | Can start from the first few words and supports realtime conversational conditioning. | Needs measurement. | Apache-2.0. English only; quality/stability may vary without conditioning/fine-tuning. |
| 7 | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) | Popular local baseline. | Public examples look more whole-text than incremental. | Low-latency voice-agent target, needs real TTFA measurement. | MIT, 350M, strong usability signal. Keep as baseline, not primary streaming engine. |
| 8 | [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) | Naturalness baseline. | Output streaming from prompt through generator. | Around 200 ms streaming claims. | Apache-2.0, vLLM-based; not clean incremental text input. |
| 9 | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Tiny sanity baseline. | Segment/generator-based. | Very fast but no hard streaming-agent TTFA target. | Apache-licensed weights, easy local usage. Not top-tier for expressive incremental TTS. |

Current recommendation: hosted APIs are out of scope for the main research stack. Start with Qwen3-TTS 0.6B on the rented NVIDIA GPU, compare immediately against CosyVoice 0.5B, then test VoXtream2 and Kyutai for research-aligned incremental behavior.

## Turn-Taking Datasets

| Rank | Dataset | Fit | Constraints | Notes |
| ---: | --- | --- | --- | --- |
| 1 | [CANDOR](https://convokit.cornell.edu/documentation/candor.html) | Best primary acquisition target. | Medium friction: BetterUp application plus TalkBank access. Need to inspect media structure. | 1,650+ dyadic stranger video-chat conversations and 850+ hours. ConvoKit exposes `start`, `stop`, `interval`, `overlap`, `backchannel`, and backchannel spans/counts. Treat timing/turn grouping as algorithmic until audited. |
| 2 | [EgoCom](https://github.com/facebookresearch/EgoCom-Dataset) | Best modern public multimodal turn-taking dataset. | Low friction via GitHub. Speaker labels may be noisy at 1-second level. | 38.5h natural multi-person egocentric conversations with synchronized stereo audio/video and human-created timestamped word transcripts. |
| 3 | [EasyCom](https://github.com/facebookresearch/EasyComDataset) | Strong public overlap/target-of-speech dataset. | Low friction via GitHub. Small enough to use mainly as eval/augmentation. | Natural group conversations in a noisy restaurant-like AR setting, with headset mics, AR glasses array, VAD, transcripts, and target-of-speech labels. |
| 4 | [CHiME-6](https://chimechallenge.github.io/chime6/overview.html) / CHiME-7/8 DASR | Strong realism and noise/overlap source. | Medium challenge/data-term friction. Multiparty and far-field, so less aligned with dyadic assistant behavior. | Real home dinner parties with speaker labels and start/end annotations. |
| 5 | [DiPCo](https://zenodo.org/records/8122551) | Clean small speaker-separated dinner-party evaluation set. | Low friction via Zenodo, CDLA-Permissive. Small. | Four-person natural English conversation, per-speaker close-talk plus far-field arrays, human-labeled transcripts. |
| 6 | [AliMeeting](https://openslr.org/119/) | Useful overlap/meeting architecture validation. | Language mismatch for English-first work. | 118.75h Mandarin meetings, high-quality transcription, far-field array and near-field headset mics. |
| 7 | [MM-F2F](https://github.com/Linyx1125/MM-F2F) | Modern turn-taking/backchannel benchmark candidate. | Licensing/provenance needs review because source is in-the-wild video; annotations mostly automatic/minimally checked. | 210h text/audio/video with word-level turn/backchannel annotations. |
| 8 | [GAP-derived Interruption Audio & Transcript](https://www.mdpi.com/2306-5729/9/9/104) | Targeted interruption evaluation set. | CC-BY-NC 4.0, mono mixed audio, timestamp imperfections acknowledged. | 200 manually annotated true interruptions from overlapping utterances. |
| 9 | [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Secondary meeting benchmark. | Older and partly scenario-elicited. | 100h multimodal meetings with close/far mics, individual/room video, transcripts, and rich annotations. |
| 10 | [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) / CUEMPATHY | Privacy-restricted clinical/counseling dialogue. | High access restrictions; likely evaluation/research only. | Real turn dynamics and emotional stakes, but not a primary acquisition target. |

Current recommendation: acquire CANDOR first, and immediately inspect EgoCom, EasyCom, and DiPCo because they are lower friction. Switchboard, Fisher, and CALLHOME should be historical sanity baselines only, not the main training substrate.

## Turn-Taking Models And Baselines

| Model / Paper | What To Borrow |
| --- | --- |
| [VAP: Voice Activity Projection](https://arxiv.org/abs/2205.09812) | Predict future voice activity jointly instead of using only silence. This naturally maps to hold, shift, backchannel, and overlap behavior. |
| [TurnGPT](https://aclanthology.org/2020.findings-emnlp.268/) | Text-only incremental turn-shift prediction from pragmatic and syntactic completeness. Useful for stable ASR token streams. |
| [PairwiseTurnGPT](https://www.semdial.org/anthology/Z24-Leishman_semdial_0002.pdf) | Two aligned speaker text streams instead of serialized dialogue. Useful when training from separated-channel data. |
| [ASR-integrated turn-taking predictor](https://ar5iv.labs.arxiv.org/html/2208.13321) | Close to the frozen-ASR adapter idea: add turn-taking targets on top of ASR representations. |
| [Acoustic + ASR decoder EOU detector](https://m.media-amazon.com/images/G/01/amazon.jobs/Endpointing2p0_V20180316.pdf) | Practical production baseline using acoustic embeddings, ASR hypothesis/decoder features, and pause safeguards. |
| [Acoustic + LLM fusion for turn/backchannel prediction](https://assets.amazon.science/95/b2/0cd8a6ce484497c31a7cf932ae3c/turn-taking-and-backchannel-prediction-with-acoustic-and-large-language-model-fusion.pdf) | Fuse acoustic turn cues with lexical/semantic context. |
| [Response-conditioned TurnGPT](https://aclanthology.org/2023.findings-acl.776.pdf) | Decide whether to take the turn based partly on candidate response fit, not only user EOU probability. |

Recommended label framing for the adapter: train multi-task labels rather than a binary EOU classifier. Use frame-level labels at 20-50 ms plus event spans: `speech_active`, `primary_floor`, `listener_backchannel`, `turn_hold`, `turn_relevance_place`, `turn_shift`, `interruption_attempt`, `interruption_success`, `competitive_overlap`, `cooperative_overlap`, `laughter`, `nonspeech_vocalization`, `silence_gap_ms`, and `response_latency_ms`. Derive soft policy labels such as `agent_should_wait`, `agent_may_backchannel`, `agent_should_take_turn`, and `agent_should_yield`.

## Open Decisions

- Does CANDOR provide speaker-separated media, mixed Zoom media, or only speaker-attributed transcripts over mixed media?
- How accurate are CANDOR timestamps and Backbiter/Audiophile-derived backchannel labels after manual audit?
- Does Qwen3-TTS local streaming actually accept incremental text and emit playable audio before the full sentence?
- Can Nemotron Speech Streaming English expose stable intermediate encoder/RNNT features without modifying too much of NeMo?
- Does Qwen3-1.7B in non-thinking mode beat LiquidAI LFM2.5-1.2B on real TTFT while preserving conversational quality?

## Proposed First Experiment Stack

Use one fully local/single-machine research stack first:

- ASR: NVIDIA Nemotron Speech Streaming English 0.6B at 160 ms and 560 ms chunk settings.
- LLM: Qwen3-1.7B on vLLM with automatic prefix caching and non-thinking mode.
- LLM challenger: LiquidAI LFM2.5-1.2B-Instruct for raw speed and memory comparison.
- TTS: Qwen3-TTS 12Hz 0.6B first, CosyVoice 0.5B fallback, then VoXtream2/Kyutai for deeper incremental streaming research.
- Turn-taking data: CANDOR acquisition first; EgoCom, EasyCom, and DiPCo immediate inspection/audit.

The first measurable comparison should report cold LLM generation, system-prompt-prefilled generation, conversation-prefilled generation, and stable-user-prefix-prefilled generation separately.

The first dataset task should build a one-hour manual audit set across CANDOR, EgoCom, EasyCom, and DiPCo to verify timestamp quality, overlap labels, backchannels, interruptions, laughter, and whether the audio is speaker-separated or only diarized/mixed.

