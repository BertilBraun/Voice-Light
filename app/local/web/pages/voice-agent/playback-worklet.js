class PcmPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.inputSampleRate = options.processorOptions.inputSampleRate;
    this.inputSamplesPerOutputSample = this.inputSampleRate / sampleRate;
    this.chunks = [];
    this.queuedSampleCount = 0;
    this.sourcePosition = 0;
    this.generationId = -1;
    this.cancelledGenerationId = -1;
    this.endedGenerationId = -1;
    this.port.onmessage = ({ data }) => {
      if (data.type === "clear") {
        this.chunks = [];
        this.queuedSampleCount = 0;
        this.sourcePosition = 0;
        this.cancelledGenerationId = Math.max(this.cancelledGenerationId, data.generationId);
        this.endedGenerationId = -1;
      } else if (data.type === "audio" && data.generationId > this.cancelledGenerationId) {
        if (data.generationId > this.generationId) {
          this.chunks = [];
          this.queuedSampleCount = 0;
          this.sourcePosition = 0;
          this.generationId = data.generationId;
          this.endedGenerationId = -1;
        }
        if (data.generationId === this.generationId) {
          const chunk = new Int16Array(data.pcm);
          this.chunks.push(chunk);
          this.queuedSampleCount += chunk.length;
        }
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
      if (remainingIndex < chunk.length) return chunk[remainingIndex];
      remainingIndex -= chunk.length;
    }
    return undefined;
  }

  consumeSamples(count) {
    let remainingCount = count;
    while (remainingCount > 0 && this.chunks.length > 0) {
      const chunk = this.chunks[0];
      if (remainingCount < chunk.length) {
        this.chunks[0] = chunk.subarray(remainingCount);
        this.queuedSampleCount -= remainingCount;
        return;
      }
      remainingCount -= chunk.length;
      this.queuedSampleCount -= chunk.length;
      this.chunks.shift();
    }
  }

  reportCompletionIfDrained() {
    if (this.endedGenerationId !== this.generationId || this.availableSampleCount() !== 0) return;
    this.port.postMessage({ type: "playback.complete", generationId: this.generationId });
    this.endedGenerationId = -1;
  }
}

registerProcessor("pcm-playback", PcmPlaybackProcessor);
