class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = options.processorOptions.targetSampleRate;
    this.sourcePosition = 0;
    this.previousSample = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input || input.length === 0) return true;
    const ratio = sampleRate / this.targetSampleRate;
    const outputSamples = [];
    while (this.sourcePosition < input.length) {
      const lowerIndex = Math.floor(this.sourcePosition);
      const fraction = this.sourcePosition - lowerIndex;
      if (lowerIndex === input.length - 1 && fraction > 0) break;
      const lowerSample = lowerIndex < 0 ? this.previousSample : input[lowerIndex];
      const upperIndex = lowerIndex + 1;
      const upperSample = upperIndex < input.length ? input[upperIndex] : lowerSample;
      const sample = lowerSample * (1 - fraction) + upperSample * fraction;
      outputSamples.push(Math.max(-1, Math.min(1, sample)) * 0x7fff);
      this.sourcePosition += ratio;
    }
    this.sourcePosition -= input.length;
    this.previousSample = input[input.length - 1];
    const pcm = Int16Array.from(outputSamples);
    if (pcm.length > 0) this.port.postMessage(pcm.buffer, [pcm.buffer]);
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
