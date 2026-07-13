class PcmPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = [];
    this.chunkOffset = 0;
    this.generationId = -1;
    this.port.onmessage = ({ data }) => {
      if (data.type === "clear") {
        this.chunks = [];
        this.chunkOffset = 0;
        this.generationId = data.generationId;
      } else if (data.type === "audio" && data.generationId >= this.generationId) {
        if (data.generationId > this.generationId) {
          this.chunks = [];
          this.chunkOffset = 0;
          this.generationId = data.generationId;
        }
        this.chunks.push(new Int16Array(data.pcm));
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    let outputOffset = 0;
    while (outputOffset < output.length && this.chunks.length > 0) {
      const chunk = this.chunks[0];
      const count = Math.min(output.length - outputOffset, chunk.length - this.chunkOffset);
      for (let index = 0; index < count; index += 1) output[outputOffset + index] = chunk[this.chunkOffset + index] / 0x8000;
      outputOffset += count;
      this.chunkOffset += count;
      if (this.chunkOffset === chunk.length) {
        this.chunks.shift();
        this.chunkOffset = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-playback", PcmPlaybackProcessor);
