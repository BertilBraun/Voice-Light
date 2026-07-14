class PcmPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.inputSampleRate = options.processorOptions.inputSampleRate;
    this.inputSamplesPerOutputSample = this.inputSampleRate / sampleRate;
    this.chunks = [];
    this.queuedSampleCount = 0;
    this.sourcePosition = 0;
    this.sentencePlayback = new Map();
    this.generationId = -1;
    this.cancelledGenerationId = -1;
    this.endedGenerationId = -1;
    this.port.onmessage = ({ data }) => {
      if (data.type === "clear") {
        this.chunks = [];
        this.queuedSampleCount = 0;
        this.sourcePosition = 0;
        this.sentencePlayback.clear();
        this.cancelledGenerationId = Math.max(this.cancelledGenerationId, data.generationId);
        this.endedGenerationId = -1;
      } else if (data.type === "audio" && data.generationId > this.cancelledGenerationId) {
        if (data.generationId > this.generationId) {
          this.chunks = [];
          this.queuedSampleCount = 0;
          this.sourcePosition = 0;
          this.sentencePlayback.clear();
          this.generationId = data.generationId;
          this.endedGenerationId = -1;
        }
        if (data.generationId === this.generationId) {
          const samples = new Int16Array(data.pcm);
          this.chunks.push({ samples, sentenceId: data.sentenceId });
          this.queuedSampleCount += samples.length;
        }
      } else if (data.type === "sentence" && data.generationId === this.generationId) {
        const playback = this.sentencePlayback.get(data.sentenceId) ?? { playedSamples: 0 };
        playback.totalSamples = data.totalSamples;
        this.sentencePlayback.set(data.sentenceId, playback);
        this.reportSentenceProgress(data.sentenceId);
      } else if (data.type === "end" && data.generationId === this.generationId) {
        this.endedGenerationId = data.generationId;
        this.reportCompletionIfDrained();
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
      if (this.availableSampleCount() === 0) break;
      const lowerIndex = Math.floor(this.sourcePosition);
      const fraction = this.sourcePosition - lowerIndex;
      const lowerSample = this.sampleAt(lowerIndex);
      const upperSample = this.sampleAt(lowerIndex + 1) ?? lowerSample;
      output[outputIndex] = (lowerSample * (1 - fraction) + upperSample * fraction) / 0x8000;
      this.sourcePosition += this.inputSamplesPerOutputSample;
      const consumedSamples = Math.floor(this.sourcePosition);
      if (consumedSamples > 0) {
        this.consumeSamples(consumedSamples);
        this.sourcePosition -= consumedSamples;
      }
    }
    this.reportCompletionIfDrained();
    return true;
  }

  availableSampleCount() {
    return this.queuedSampleCount;
  }

  sampleAt(index) {
    let remainingIndex = index;
    for (const chunk of this.chunks) {
      if (remainingIndex < chunk.samples.length) return chunk.samples[remainingIndex];
      remainingIndex -= chunk.samples.length;
    }
    return undefined;
  }

  consumeSamples(count) {
    let remainingCount = count;
    while (remainingCount > 0 && this.chunks.length > 0) {
      const chunk = this.chunks[0];
      const consumedCount = Math.min(remainingCount, chunk.samples.length);
      this.recordPlayedSamples(chunk.sentenceId, consumedCount);
      if (remainingCount < chunk.samples.length) {
        this.chunks[0] = {
          samples: chunk.samples.subarray(remainingCount),
          sentenceId: chunk.sentenceId,
        };
        this.queuedSampleCount -= remainingCount;
        return;
      }
      remainingCount -= chunk.samples.length;
      this.queuedSampleCount -= chunk.samples.length;
      this.chunks.shift();
    }
  }

  recordPlayedSamples(sentenceId, sampleCount) {
    const playback = this.sentencePlayback.get(sentenceId) ?? { playedSamples: 0 };
    playback.playedSamples += sampleCount;
    this.sentencePlayback.set(sentenceId, playback);
    this.reportSentenceProgress(sentenceId);
  }

  reportSentenceProgress(sentenceId) {
    const playback = this.sentencePlayback.get(sentenceId);
    if (!playback?.totalSamples || playback.playedSamples === playback.reportedSamples) return;
    playback.reportedSamples = playback.playedSamples;
    this.port.postMessage({
      type: "sentence.progress",
      generationId: this.generationId,
      sentenceId,
      playedSamples: Math.min(playback.playedSamples, playback.totalSamples),
      totalSamples: playback.totalSamples,
    });
  }

  reportCompletionIfDrained() {
    if (
      this.generationId < 1 ||
      this.endedGenerationId !== this.generationId ||
      this.availableSampleCount() !== 0
    ) return;
    this.port.postMessage({ type: "playback.complete", generationId: this.generationId });
    this.endedGenerationId = -1;
  }
}

registerProcessor("pcm-playback", PcmPlaybackProcessor);
