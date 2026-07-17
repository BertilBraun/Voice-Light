const PlaybackState = Object.freeze({
  IDLE: "idle",
  QUEUED: "queued",
  SPEAKING: "speaking",
  DUCKING: "ducking",
  PAUSED_BUFFERED: "paused_buffered",
  RESUMING: "resuming",
  DRAINING_TO_BOUNDARY: "draining_to_boundary",
  CANCELLED: "cancelled",
  COMPLETED: "completed",
});

const PlaybackAction = Object.freeze({
  DUCK: "duck",
  PAUSE_AT_BOUNDARY: "pause_at_boundary",
  RESUME: "resume",
  CANCEL: "cancel",
});

const PauseResult = Object.freeze({
  NOT_REQUESTED: "not_requested",
  WORD_BOUNDARY: "word_boundary",
  FORCED_SAMPLE: "forced_sample",
});

const TERMINAL_STATES = new Set([PlaybackState.CANCELLED, PlaybackState.COMPLETED]);
const MAX_RETAINED_COMMANDS = 256;

class PcmPlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.inputSampleRate = options.processorOptions.inputSampleRate;
    this.inputSamplesPerOutputSample = this.inputSampleRate / sampleRate;
    this.outputSampleRate = sampleRate;
    this.cancelledGenerationId = -1;
    this.processedCommandIds = [];
    this.commandAcknowledgements = new Map();
    this.pendingCommandIds = new Set();
    this.resetGeneration(-1, PlaybackState.IDLE);
    this.port.onmessage = ({ data }) => this.handleMessage(data);
  }

  resetGeneration(generationId, state) {
    this.chunks = [];
    this.queuedSourceSampleCount = 0;
    this.sourceFraction = 0;
    this.sourceSamplePosition = 0;
    this.renderedOutputSamplePosition = 0;
    this.acknowledgedTextOffset = 0;
    this.boundaries = [];
    this.generationId = generationId;
    this.endedGenerationId = -1;
    this.playbackStarted = false;
    this.state = state;
    this.currentGain = 1;
    this.gainRamp = undefined;
    this.pauseRequest = undefined;
    this.pausedAtBrowserTimeMs = undefined;
    this.pendingGainCommand = undefined;
    this.pendingResumeCommand = undefined;
  }

  handleMessage(data) {
    switch (data.type) {
      case "audio":
        this.enqueueAudio(data);
        break;
      case "boundary":
        this.registerBoundary(data);
        break;
      case "end":
        this.endGeneration(data.generationId);
        break;
      case "playback.command":
        this.applyCommand(data);
        break;
    }
  }

  enqueueAudio(data) {
    if (data.generationId <= this.cancelledGenerationId) return;
    if (data.generationId > this.generationId) this.replaceGeneration(data.generationId);
    if (data.generationId !== this.generationId || TERMINAL_STATES.has(this.state)) return;
    const samples = new Int16Array(data.pcm);
    const expectedStartSample = this.sourceSamplePosition + this.queuedSourceSampleCount;
    if (data.startSample !== expectedStartSample) return;
    this.chunks.push(samples);
    this.queuedSourceSampleCount += samples.length;
    if (this.state === PlaybackState.IDLE) this.state = PlaybackState.QUEUED;
  }

  registerBoundary(data) {
    if (data.generationId <= this.cancelledGenerationId) return;
    if (data.generationId > this.generationId) this.replaceGeneration(data.generationId);
    if (data.generationId !== this.generationId || TERMINAL_STATES.has(this.state)) return;
    this.boundaries.push({ textOffset: data.textOffset, startSample: data.startSample });
    this.boundaries.sort((left, right) => left.startSample - right.startSample);
    if (
      this.pauseRequest &&
      this.pauseRequest.boundarySourceSamplePosition === null &&
      data.startSample >= this.sourceSamplePosition
    ) {
      this.pauseRequest.boundarySourceSamplePosition = data.startSample;
    }
    this.reportCrossedBoundaries();
    this.pauseIfDue();
  }

  endGeneration(generationId) {
    if (generationId !== this.generationId || TERMINAL_STATES.has(this.state)) return;
    this.endedGenerationId = generationId;
    this.reportCompletionIfDrained();
  }

  replaceGeneration(generationId) {
    if (generationId <= this.generationId) return;
    if (this.generationId > 0 && !TERMINAL_STATES.has(this.state)) {
      this.cancelledGenerationId = Math.max(this.cancelledGenerationId, this.generationId);
    }
    this.resetGeneration(generationId, PlaybackState.IDLE);
  }

  applyCommand(command) {
    const cachedAcknowledgement = this.commandAcknowledgements.get(command.commandId);
    if (cachedAcknowledgement) {
      this.port.postMessage(cachedAcknowledgement);
      return;
    }
    if (this.pendingCommandIds.has(command.commandId)) return;
    if (
      command.generationId !== this.generationId ||
      command.generationId <= this.cancelledGenerationId ||
      TERMINAL_STATES.has(this.state)
    ) return;
    this.pendingCommandIds.add(command.commandId);
    switch (command.action) {
      case PlaybackAction.DUCK:
        this.applyDuck(command);
        break;
      case PlaybackAction.PAUSE_AT_BOUNDARY:
        this.applyPause(command);
        break;
      case PlaybackAction.RESUME:
        this.applyResume(command);
        break;
      case PlaybackAction.CANCEL:
        this.applyCancel(command);
        break;
      default:
        this.pendingCommandIds.delete(command.commandId);
    }
  }

  applyDuck(command) {
    if (
      this.state !== PlaybackState.SPEAKING &&
      this.state !== PlaybackState.QUEUED &&
      this.state !== PlaybackState.DUCKING
    ) {
      this.acknowledgeCommand(command, PauseResult.NOT_REQUESTED, false, false, 0);
      return;
    }
    this.state = PlaybackState.DUCKING;
    this.pendingGainCommand = command;
    this.beginGainRamp(command.targetGain, command.gainRampDurationMs);
    if (this.gainRamp === undefined) this.finishPendingGainCommand(true);
  }

  applyPause(command) {
    if (
      this.state !== PlaybackState.DUCKING &&
      this.state !== PlaybackState.SPEAKING &&
      this.state !== PlaybackState.QUEUED &&
      this.state !== PlaybackState.DRAINING_TO_BOUNDARY
    ) {
      this.acknowledgeCommand(command, PauseResult.NOT_REQUESTED, false, false, 0);
      return;
    }
    this.state = PlaybackState.DRAINING_TO_BOUNDARY;
    this.pauseRequest = {
      command,
      boundarySourceSamplePosition: command.requestedBoundarySourceSamplePosition,
      renderedOutputSampleDeadline: command.renderedOutputSampleDeadline,
    };
    this.pauseIfDue();
  }

  applyResume(command) {
    const wasPaused = this.state === PlaybackState.PAUSED_BUFFERED;
    const wasReversibleDrain =
      this.state === PlaybackState.DUCKING ||
      this.state === PlaybackState.DRAINING_TO_BOUNDARY;
    if (!wasPaused && !wasReversibleDrain) {
      this.acknowledgeCommand(command, PauseResult.NOT_REQUESTED, false, true, 0);
      return;
    }
    const pausedAgeMs = wasPaused
      ? this.browserTimeMs() - this.pausedAtBrowserTimeMs
      : 0;
    if (wasPaused && pausedAgeMs > command.maximumPausedAgeMs) {
      const discardedSourceSampleCount = this.discardQueuedAudio();
      this.state = PlaybackState.CANCELLED;
      this.cancelledGenerationId = Math.max(this.cancelledGenerationId, this.generationId);
      this.acknowledgeCommand(
        command,
        PauseResult.NOT_REQUESTED,
        false,
        true,
        discardedSourceSampleCount,
      );
      return;
    }
    if (this.pauseRequest) {
      const pauseCommand = this.pauseRequest.command;
      this.pauseRequest = undefined;
      this.acknowledgeCommand(
        pauseCommand,
        PauseResult.NOT_REQUESTED,
        this.gainRamp === undefined,
        false,
        0,
      );
    }
    this.state = PlaybackState.RESUMING;
    this.pendingResumeCommand = command;
    this.beginGainRamp(command.targetGain, command.gainRampDurationMs);
  }

  applyCancel(command) {
    const discardedSourceSampleCount = this.discardQueuedAudio();
    this.state = PlaybackState.CANCELLED;
    this.cancelledGenerationId = Math.max(this.cancelledGenerationId, command.generationId);
    this.endedGenerationId = -1;
    this.settleSupersededCommands();
    this.acknowledgeCommand(
      command,
      PauseResult.NOT_REQUESTED,
      this.gainRamp === undefined,
      false,
      discardedSourceSampleCount,
    );
  }

  settleSupersededCommands() {
    if (this.pendingGainCommand) {
      const command = this.pendingGainCommand;
      this.pendingGainCommand = undefined;
      this.acknowledgeCommand(command, PauseResult.NOT_REQUESTED, false, false, 0);
    }
    if (this.pauseRequest) {
      const command = this.pauseRequest.command;
      this.pauseRequest = undefined;
      this.acknowledgeCommand(command, PauseResult.NOT_REQUESTED, false, false, 0);
    }
    if (this.pendingResumeCommand) {
      const command = this.pendingResumeCommand;
      this.pendingResumeCommand = undefined;
      this.acknowledgeCommand(command, PauseResult.NOT_REQUESTED, false, true, 0);
    }
  }

  beginGainRamp(targetGain, durationMs) {
    const sampleCount = Math.max(1, Math.round(this.outputSampleRate * durationMs / 1000));
    if (targetGain === this.currentGain) {
      this.gainRamp = undefined;
      return;
    }
    this.gainRamp = {
      startGain: this.currentGain,
      targetGain,
      sampleCount,
      renderedSampleCount: 0,
    };
  }

  nextGain() {
    if (!this.gainRamp) return this.currentGain;
    this.gainRamp.renderedSampleCount += 1;
    const progress = Math.min(
      this.gainRamp.renderedSampleCount / this.gainRamp.sampleCount,
      1,
    );
    this.currentGain =
      this.gainRamp.startGain +
      (this.gainRamp.targetGain - this.gainRamp.startGain) * progress;
    if (progress === 1) {
      this.currentGain = this.gainRamp.targetGain;
      this.gainRamp = undefined;
      this.finishPendingGainCommand(true);
      if (this.state === PlaybackState.RESUMING && !this.pendingResumeCommand) {
        this.state = PlaybackState.SPEAKING;
      }
    }
    return this.currentGain;
  }

  finishPendingGainCommand(gainRampComplete) {
    if (!this.pendingGainCommand) return;
    const command = this.pendingGainCommand;
    this.pendingGainCommand = undefined;
    this.acknowledgeCommand(
      command,
      PauseResult.NOT_REQUESTED,
      gainRampComplete,
      false,
      0,
    );
  }

  pauseIfDue() {
    if (!this.pauseRequest || this.state !== PlaybackState.DRAINING_TO_BOUNDARY) return false;
    const boundaryPosition = this.pauseRequest.boundarySourceSamplePosition;
    const boundaryReached =
      boundaryPosition !== null && this.sourceSamplePosition >= boundaryPosition;
    const deadlineReached =
      this.renderedOutputSamplePosition >= this.pauseRequest.renderedOutputSampleDeadline;
    if (!boundaryReached && !deadlineReached) return false;
    const command = this.pauseRequest.command;
    this.pauseRequest = undefined;
    this.state = PlaybackState.PAUSED_BUFFERED;
    this.pausedAtBrowserTimeMs = this.browserTimeMs();
    this.finishPendingGainCommand(false);
    this.acknowledgeCommand(
      command,
      boundaryReached ? PauseResult.WORD_BOUNDARY : PauseResult.FORCED_SAMPLE,
      this.gainRamp === undefined,
      false,
      0,
    );
    return true;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    if (
      this.state === PlaybackState.IDLE ||
      this.state === PlaybackState.PAUSED_BUFFERED ||
      TERMINAL_STATES.has(this.state)
    ) return true;
    let producedAudio = false;
    for (let outputIndex = 0; outputIndex < output.length; outputIndex += 1) {
      if (this.pauseIfDue() || this.queuedSourceSampleCount === 0) break;
      const lowerIndex = Math.floor(this.sourceFraction);
      const fraction = this.sourceFraction - lowerIndex;
      const lowerSample = this.sampleAt(lowerIndex);
      const upperSample = this.sampleAt(lowerIndex + 1) ?? lowerSample;
      output[outputIndex] =
        ((lowerSample * (1 - fraction) + upperSample * fraction) / 0x8000) *
        this.nextGain();
      producedAudio = true;
      this.renderedOutputSamplePosition += 1;
      this.sourceFraction += this.inputSamplesPerOutputSample;
      const consumedSourceSampleCount = Math.floor(this.sourceFraction);
      if (consumedSourceSampleCount > 0) {
        this.consumeSourceSamples(consumedSourceSampleCount);
        this.sourceFraction -= consumedSourceSampleCount;
      }
      if (this.pendingResumeCommand) {
        const command = this.pendingResumeCommand;
        this.pendingResumeCommand = undefined;
        this.acknowledgeCommand(
          command,
          PauseResult.NOT_REQUESTED,
          this.gainRamp === undefined,
          false,
          0,
        );
        if (this.gainRamp === undefined) this.state = PlaybackState.SPEAKING;
      }
    }
    if (producedAudio && !this.playbackStarted) {
      this.playbackStarted = true;
      if (this.state === PlaybackState.QUEUED) this.state = PlaybackState.SPEAKING;
      this.port.postMessage({
        type: "playback.started",
        generationId: this.generationId,
        browserMonotonicTimeNs: this.browserMonotonicTimeNs(),
        renderedOutputSamplePosition: this.renderedOutputSamplePosition,
        sourceSamplePosition: this.sourceSamplePosition,
        outputSampleRate: this.outputSampleRate,
      });
    } else if (producedAudio && this.state === PlaybackState.QUEUED) {
      this.state = PlaybackState.SPEAKING;
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

  consumeSourceSamples(count) {
    let remainingCount = Math.min(count, this.queuedSourceSampleCount);
    this.sourceSamplePosition += remainingCount;
    this.queuedSourceSampleCount -= remainingCount;
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

  discardQueuedAudio() {
    const discardedSourceSampleCount = this.queuedSourceSampleCount;
    this.chunks = [];
    this.queuedSourceSampleCount = 0;
    this.sourceFraction = 0;
    this.boundaries = [];
    return discardedSourceSampleCount;
  }

  reportCrossedBoundaries() {
    while (
      this.boundaries.length > 0 &&
      this.sourceSamplePosition > this.boundaries[0].startSample
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
        playedSampleCount: this.sourceSamplePosition,
        browserMonotonicTimeNs: this.browserMonotonicTimeNs(),
        renderedOutputSamplePosition: this.renderedOutputSamplePosition,
        outputSampleRate: this.outputSampleRate,
      });
    }
  }

  reportCompletionIfDrained() {
    if (
      this.generationId < 1 ||
      this.endedGenerationId !== this.generationId ||
      this.queuedSourceSampleCount !== 0 ||
      TERMINAL_STATES.has(this.state)
    ) return;
    this.settleSupersededCommands();
    this.state = PlaybackState.COMPLETED;
    this.port.postMessage({
      type: "playback.complete",
      generationId: this.generationId,
      browserMonotonicTimeNs: this.browserMonotonicTimeNs(),
      renderedOutputSamplePosition: this.renderedOutputSamplePosition,
      sourceSamplePosition: this.sourceSamplePosition,
      outputSampleRate: this.outputSampleRate,
    });
    this.endedGenerationId = -1;
  }

  acknowledgeCommand(
    command,
    pauseResult,
    gainRampComplete,
    resumeRejected,
    discardedSourceSampleCount,
  ) {
    const acknowledgement = {
      type: "playback.acknowledgement",
      commandId: command.commandId,
      generationId: command.generationId,
      action: command.action,
      streamEpoch: command.streamEpoch,
      turnEpoch: command.turnEpoch,
      resultingState: this.state,
      browserMonotonicTimeNs: this.browserMonotonicTimeNs(),
      renderedOutputSamplePosition: this.renderedOutputSamplePosition,
      sourceSamplePosition: this.sourceSamplePosition,
      outputSampleRate: this.outputSampleRate,
      pauseResult,
      currentGain: this.currentGain,
      gainRampComplete,
      queuedSourceSampleCount: this.queuedSourceSampleCount,
      discardedSourceSampleCount,
      replayedSourceSampleCount: 0,
      skippedSourceSampleCount: 0,
      resumeRejected,
    };
    this.pendingCommandIds.delete(command.commandId);
    this.commandAcknowledgements.set(command.commandId, acknowledgement);
    this.processedCommandIds.push(command.commandId);
    while (this.processedCommandIds.length > MAX_RETAINED_COMMANDS) {
      const expiredCommandId = this.processedCommandIds.shift();
      this.commandAcknowledgements.delete(expiredCommandId);
    }
    this.port.postMessage(acknowledgement);
  }

  browserTimeMs() {
    return currentTime * 1000;
  }

  browserMonotonicTimeNs() {
    return Math.round(currentTime * 1_000_000_000);
  }
}

registerProcessor("pcm-playback", PcmPlaybackProcessor);
