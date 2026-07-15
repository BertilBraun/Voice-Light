class PcmPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.inputSampleRate = options.processorOptions.inputSampleRate;
    this.inputSamplesPerOutputSample = this.inputSampleRate / sampleRate;
    this.chunks = [];
    this.queuedSampleCount = 0;
    this.sourcePosition = 0;
    this.playedSampleCount = 0;
    this.acknowledgedTextOffset = 0;
    this.boundaries = [];
    this.generationId = -1;
    this.cancelledGenerationId = -1;
    this.endedGenerationId = -1;
    this.port.onmessage = ({ data }) => this.handleMessage(data);
  }

  handleMessage(data) {
    if (data.type === "clear") {
      this.stopGeneration(data.generationId);
    } else if (data.type === "audio" && data.generationId > this.cancelledGenerationId) {
      if (data.generationId > this.generationId) this.startGeneration(data.generationId);
      if (data.generationId === this.generationId) {
        const samples = new Int16Array(data.pcm);
        const expectedStartSample = this.playedSampleCount + this.queuedSampleCount;
        if (data.startSample !== expectedStartSample) return;
        this.chunks.push(samples);
        this.queuedSampleCount += samples.length;
      }
    } else if (data.type === "boundary" && data.generationId > this.cancelledGenerationId) {
      if (data.generationId > this.generationId) this.startGeneration(data.generationId);
      if (data.generationId === this.generationId) {
        this.boundaries.push({ textOffset: data.textOffset, startSample: data.startSample });
        this.boundaries.sort((left, right) => left.startSample - right.startSample);
        this.reportCrossedBoundaries();
      }
    } else if (data.type === "end" && data.generationId === this.generationId) {
      this.endedGenerationId = data.generationId;
      this.reportCompletionIfDrained();
    }
  }

  startGeneration(generationId) {
    this.chunks = [];
    this.queuedSampleCount = 0;
    this.sourcePosition = 0;
    this.playedSampleCount = 0;
    this.acknowledgedTextOffset = 0;
    this.boundaries = [];
    this.generationId = generationId;
    this.endedGenerationId = -1;
  }

  stopGeneration(generationId) {
    let playedSampleCount = 0;
    let textOffset = 0;
    if (generationId === this.generationId) {
      this.reportCrossedBoundaries();
      playedSampleCount = this.playedSampleCount;
      textOffset = this.acknowledgedTextOffset;
      this.chunks = [];
      this.queuedSampleCount = 0;
      this.sourcePosition = 0;
      this.boundaries = [];
      this.endedGenerationId = -1;
    }
    this.cancelledGenerationId = Math.max(this.cancelledGenerationId, generationId);
    this.port.postMessage({
      type: "playback.stopped",
      generationId,
      playedSampleCount,
      textOffset,
    });
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
      if (this.queuedSampleCount === 0) break;
      const lowerIndex = Math.floor(this.sourcePosition);
      const fraction = this.sourcePosition - lowerIndex;
      const lowerSample = this.sampleAt(lowerIndex);
      const upperSample = this.sampleAt(lowerIndex + 1) ?? lowerSample;
      output[outputIndex] = (lowerSample * (1 - fraction) + upperSample * fraction) / 0x8000;
      this.sourcePosition += this.inputSamplesPerOutputSample;
      const consumedSampleCount = Math.floor(this.sourcePosition);
      if (consumedSampleCount > 0) {
        this.consumeSamples(consumedSampleCount);
        this.sourcePosition -= consumedSampleCount;
      }
    }
    this.reportCrossedBoundaries();
    this.reportCompletionIfDrained();
    return true;
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
    let remainingCount = Math.min(count, this.queuedSampleCount);
    this.playedSampleCount += remainingCount;
    this.queuedSampleCount -= remainingCount;
    while (remainingCount > 0 && this.chunks.length > 0) {
      const chunk = this.chunks[0];
      if (remainingCount < chunk.length) {
        this.chunks[0] = chunk.subarray(remainingCount);
        return;
      }
      remainingCount -= chunk.length;
      this.chunks.shift();
    }
  }

  reportCrossedBoundaries() {
    while (
      this.boundaries.length > 0 &&
      this.playedSampleCount > this.boundaries[0].startSample
    ) {
      const boundary = this.boundaries.shift();
      this.acknowledgedTextOffset = Math.max(
        this.acknowledgedTextOffset,
        boundary.textOffset,
      );
      this.port.postMessage({
        type: "boundary.progress",
        generationId: this.generationId,
        textOffset: boundary.textOffset,
        startSample: boundary.startSample,
        playedSampleCount: this.playedSampleCount,
      });
    }
  }

  reportCompletionIfDrained() {
    if (
      this.generationId < 1 ||
      this.endedGenerationId !== this.generationId ||
      this.queuedSampleCount !== 0
    ) return;
    this.port.postMessage({ type: "playback.complete", generationId: this.generationId });
    this.endedGenerationId = -1;
  }
}

registerProcessor("pcm-playback", PcmPlaybackProcessor);
