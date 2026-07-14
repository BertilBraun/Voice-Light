class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input || input.length === 0) return true;
    const pcm = new Int16Array(input.length);
    for (let index = 0; index < input.length; index += 1) {
      pcm[index] = Math.max(-1, Math.min(1, input[index])) * 0x7fff;
    }
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
