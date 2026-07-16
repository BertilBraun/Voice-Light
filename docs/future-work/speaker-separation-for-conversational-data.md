# Speaker Separation for Conversational Voice-Agent Data

Recover approximate per-speaker audio streams from single-channel conversations so overlaps, interruptions, backchannels, laughter, breaths, and other conversational events can become useful training signals.

## Goal

Given a mono conversation `x(t)`, estimate two participant streams `s₁(t)` and `s₂(t)`. The goal is not necessarily studio-quality isolated speech. It is to recover speaker activity and conversational vocal events well enough to create training data for turn-taking and conversational voice-agent models.

## Core Idea

Speaker separation differs from diarization:

- **Diarization** identifies who spoke when.
- **Speaker separation** reconstructs individual audio sources during overlapping speech.

Long-term speaker identity is not essential for this application. Audio can be processed in short windows, and output-channel assignments may change between snippets.

A practical pipeline is:

```text
Mono conversation
    → overlap-aware diarization or activity detection
    → candidate overlap and backchannel regions
    → two-speaker separation
    → quality and confidence filtering
    → separated snippets and speaker-activity labels
    → turn-taking model training
```

Non-overlapping speech should generally be taken directly from the original recording. Separation is most valuable around overlaps, interruptions, and short responses.

## Model Formulation

For a two-speaker mixture `x = sA + sB`, the model predicts two estimates:

```text
(ŝ1, ŝ2) = fθ(x)
```

Because output order is arbitrary, training uses permutation-invariant training (PIT).

## Permutation-Invariant Training

PIT evaluates both possible speaker assignments and selects the lower-loss one:

```text
L(PIT) = min(
  loss(ŝ1, sA) + loss(ŝ2, sB),
  loss(ŝ1, sB) + loss(ŝ2, sA)
)
```

The permutation is normally selected over the complete training crop rather than independently for every frame. This is commonly called utterance-level PIT. For two speakers, the computational overhead is negligible.

## Recommended Architecture

A strong quality-oriented design is a noncausal time-frequency separator such as TF-GridNet.

```text
Waveform
    → complex STFT
    → time-frequency feature projection
    → repeated spectral, temporal, and attention blocks
    → complex spectrogram prediction for each source
    → inverse STFT
```

The model jointly reasons across:

- frequency structure such as pitch, harmonics, and formants;
- temporal continuity;
- longer-range relationships through attention.

This is particularly useful for quiet backchannels underneath louder speech.

A practical initial configuration would use:

- mono 16 kHz audio;
- 6–12 second training crops;
- approximately 10–30 million parameters;
- two vocal outputs;
- optionally one residual output for music, noise, and non-vocal audio.

Alternative baselines include Conv-TasNet, DPRNN-TasNet, and SepFormer. Pretrained implementations should be benchmarked before training a new model.

## Output Design

For real recordings, a three-output formulation may be preferable:

```text
x → ŝ1, ŝ2, r̂
```

The residual `r` represents music, background noise, room ambience, microphone artifacts, and other non-vocal audio. PIT is applied only across the two vocal outputs. The residual output has a fixed target.

## Training Losses

A useful objective combines:

```text
L = L(PIT)
  + λ(STFT) L(MR-STFT)
  + λ(mix) L(mix)
  + λ(silence) L(silence)
```

Relevant components:

- **SI-SDR loss:** waveform reconstruction quality.
- **Multi-resolution STFT loss:** spectral and transient quality.
- **Mixture consistency:** requires separated outputs to sum approximately to the input.
- **Silence or leakage loss:** penalizes duplicated speech in an inactive output.

Mixture consistency is particularly important for preserving unfamiliar events such as laughter, breaths, coughs, and short vocalizations.

## Data Construction

Channel-separated conversational recordings provide ideal supervision. Given isolated tracks `s1` and `s2`, construct mixtures with independently sampled gains and noise:

```text
x(t) = g1 s1(t) + g2 s2(t) + n(t)
```

Fixed 50/50 and 60/40 mixtures are insufficient. Relative levels should include:

- approximately equal speakers;
- moderate imbalances of 3–8 dB;
- difficult imbalances of 8–15 dB;
- occasional extreme cases.

Training and evaluation mixtures should cover:

- single-speaker speech;
- alternating speech without overlap;
- short backchannel overlap;
- interruptions;
- extended overlap;
- simultaneous laughter or nonverbal events;
- silence controls;
- noise, reverberation, codecs, and gain variation.

The model should be trained on complete participant vocal tracks, not only transcript-aligned speech. Targets should retain breaths, laughter, hesitation sounds, coughs, partial words, and other vocal events.

## Using an Existing 100-Hour Dataset

A 100-hour channel-separated dataset is sufficient for benchmarking existing separators, building a domain-specific evaluation suite, fine-tuning a pretrained model, and testing whether mono conversational data can be mined reliably.

A small, carefully selected evaluation subset is more useful initially than processing all 100 hours. Recommended evaluation strata are:

- ordinary overlap;
- quiet backchannels;
- interruptions;
- nonverbal overlap;
- single-speaker controls;
- varied relative gains;
- codec and noise variants.

## Evaluation

Average SI-SDR alone is not sufficient. The important metrics are downstream event quality:

- backchannel recovery recall;
- interruption recovery;
- source leakage;
- duplicated-event rate;
- onset and offset error;
- overlap activity F1;
- ASR recovery after separation;
- human usability.

The most relevant summary metric is usable conversational-event recall at a fixed high precision. For example:

> At 98% precision, recover 65% of backchannels and 80% of interruption onsets within 100 ms.

This is more meaningful than a single average waveform score.

## Expected Practical Quality

Modern separation systems perform very well on matched synthetic benchmarks but degrade on arbitrary real recordings.

| Case | Expected quality |
| --- | --- |
| Small boundary overlap | Usually good |
| Clear interruption | Often good |
| Audible backchannel | Mixed to good |
| Very quiet backchannel | Unreliable |
| Simultaneous laughter | Often imperfect |
| Similar-sounding speakers | Harder |
| Music or effects | Requires residual modelling |
| Clipped or compressed overlap | Potentially unrecoverable |
| More than two speakers | Requires another formulation |

The most important errors are suppressing a quiet backchannel, leaking one speaker into the other output, duplicating the same event into both outputs, shifting event timing, and hallucinating low-energy speech.

Separated audio should therefore be treated as silver supervision rather than guaranteed ground truth.

## Recommended Use in Voice-Agent Training

The safest use is as an offline teacher.

```text
Original mono audio
    → separator + diarizer
    → confidence-weighted speaker-activity and overlap labels
    → train turn-taking model on original mono audio
```

The deployed model does not need to consume separated streams. Separation is used only to recover better labels from otherwise ambiguous mono recordings.

Useful derived targets include the probabilities of speaker A activity, speaker B activity, overlap, backchannel, and interruption. The original mixture, separated estimates, timestamps, and confidence scores should all be retained.

## Data Sources

Potential sources of channel-separated conversational audio include:

- the existing 100-hour dataset;
- Switchboard;
- Fisher telephone conversations;
- CANDOR;
- AMI meetings;
- AliMeeting;
- LibriCSS for controlled benchmarking.

Split-channel telephone and online conversations are especially useful for spontaneous speech, interruptions, and backchannels. Meeting datasets provide additional overlap diversity. Podcasts remain useful as a large unlabeled target corpus even though they usually lack isolated speaker tracks.

## Recommended Experiment

1. Select 3–10 hours from the existing separated dataset.
2. Construct realistic mixtures with controlled gains and overlap types.
3. Benchmark pretrained Conv-TasNet, DPRNN, SepFormer, and TF-GridNet systems.
4. Measure event recovery, leakage, timing accuracy, and ASR quality.
5. Manually inspect difficult backchannels and interruptions.
6. Fine-tune the strongest model on the remaining in-domain data.
7. Apply it selectively to mono podcasts or other conversational recordings.
8. Retain only high-confidence separated segments or labels.

## Conclusion

Speaker separation is a sensible data-collection strategy when treated as selective, confidence-filtered silver-label generation. It is not reliable enough to convert arbitrary mono conversations into universally clean multitrack ground truth, but it may recover overlaps, interruptions, and audible backchannels from large conversational corpora.

The first step should be benchmarking existing pretrained separators on the available channel-separated data rather than training a new architecture immediately.
