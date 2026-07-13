class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = options.processorOptions.targetSampleRate;
    this.sourceOffset = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input || input.length === 0) return true;
    const ratio = sampleRate / this.targetSampleRate;
    const outputLength = Math.floor((input.length - this.sourceOffset) / ratio);
    const pcm = new Int16Array(Math.max(0, outputLength));
    for (let index = 0; index < pcm.length; index += 1) {
      const position = this.sourceOffset + index * ratio;
      const lower = Math.floor(position);
      const fraction = position - lower;
      const sample = input[lower] * (1 - fraction) + input[Math.min(lower + 1, input.length - 1)] * fraction;
      pcm[index] = Math.max(-1, Math.min(1, sample)) * 0x7fff;
    }
    this.sourceOffset = this.sourceOffset + pcm.length * ratio - input.length;
    if (pcm.length > 0) this.port.postMessage(pcm.buffer, [pcm.buffer]);
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
