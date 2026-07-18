import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const WORKLET_PATH = new URL(
  "../../app/local/web/pages/voice-agent/playback-worklet.js",
  import.meta.url,
);

class PlaybackHarness {
  constructor(sampleRate = 16_000) {
    this.sampleRate = sampleRate;
    this.messages = [];
    let Processor;
    const messages = this.messages;
    class MockAudioWorkletProcessor {
      constructor() {
        this.port = {
          onmessage: undefined,
          postMessage(message) {
            messages.push(structuredClone(message));
          },
        };
      }
    }
    this.context = vm.createContext({
      AudioWorkletProcessor: MockAudioWorkletProcessor,
      currentTime: 0,
      sampleRate,
      registerProcessor(_name, processor) {
        Processor = processor;
      },
      Set,
      Int16Array,
      Math,
      Object,
    });
    vm.runInContext(readFileSync(WORKLET_PATH, "utf8"), this.context);
    this.processor = new Processor({
      processorOptions: { inputSampleRate: sampleRate },
    });
  }

  send(message) {
    this.processor.handleMessage(message);
  }

  enqueue(generationId, startSample, samples) {
    const pcm = Int16Array.from(samples).buffer;
    this.send({ type: "audio", generationId, startSample, pcm });
  }

  boundary(generationId, textOffset, startSample) {
    this.send({ type: "boundary", generationId, textOffset, startSample });
  }

  command(commandId, generationId, action, fields = {}) {
    this.send({
      type: "playback.command",
      commandId,
      generationId,
      action,
      streamEpoch: 1,
      turnEpoch: 1,
      ...fields,
    });
  }

  process(sampleCount) {
    const output = new Float32Array(sampleCount);
    this.processor.process([], [[output]]);
    this.context.currentTime += sampleCount / this.sampleRate;
    return Array.from(output, (sample) => Math.round(sample * 0x8000));
  }

  acknowledgement(commandId) {
    return this.messages.find(
      (message) =>
        message.type === "playback.acknowledgement" &&
        message.commandId === commandId,
    );
  }
}

test("reports a word when it starts and acknowledges it at the next word", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3, 4, 5, 6]);
  harness.boundary(1, 3, 0);
  harness.boundary(1, 7, 4);

  assert.equal(
    harness.messages.filter((message) => message.type === "boundary.started").length,
    0,
  );
  harness.process(1);
  assert.deepEqual(
    harness.messages
      .filter((message) => message.type === "boundary.started")
      .map((message) => message.textOffset),
    [3],
  );
  assert.equal(
    harness.messages.filter((message) => message.type === "boundary.progress").length,
    0,
  );

  harness.process(3);
  assert.deepEqual(
    harness.messages
      .filter((message) => message.type === "boundary.started")
      .map((message) => message.textOffset),
    [3, 7],
  );
  assert.deepEqual(
    harness.messages
      .filter((message) => message.type === "boundary.progress")
      .map((message) => message.textOffset),
    [3],
  );
});

test("coalesces words with the same start sample before acknowledging playback", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3, 4, 5, 6]);
  harness.boundary(1, 1, 0);
  harness.boundary(1, 4, 0);
  harness.boundary(1, 8, 4);

  harness.process(1);
  assert.deepEqual(
    harness.messages
      .filter((message) => message.type === "boundary.started")
      .map((message) => message.textOffset),
    [1, 4],
  );
  assert.equal(
    harness.messages.filter((message) => message.type === "boundary.progress").length,
    0,
  );

  harness.process(3);
  assert.deepEqual(
    harness.messages
      .filter((message) => message.type === "boundary.progress")
      .map((message) => message.textOffset),
    [4],
  );
});

test("pauses on a known word boundary and resumes without duplicate or skipped samples", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  harness.boundary(1, 4, 4);
  const rendered = harness.process(2);

  harness.command("pause", 1, "pause_at_boundary", {
    requestedBoundarySourceSamplePosition: 4,
    renderedOutputSampleDeadline: 8,
  });
  rendered.push(...harness.process(6));
  assert.equal(harness.processor.state, "paused_buffered");
  assert.equal(harness.processor.sourceSamplePosition, 4);
  assert.equal(harness.processor.renderedOutputSamplePosition, 4);
  assert.equal(harness.acknowledgement("pause").pauseResult, "word_boundary");

  const pausedOutputPosition = harness.processor.renderedOutputSamplePosition;
  harness.process(8);
  assert.equal(harness.processor.renderedOutputSamplePosition, pausedOutputPosition);

  harness.command("resume", 1, "resume", {
    targetGain: 1,
    gainRampDurationMs: 1,
    maximumPausedAgeMs: 800,
  });
  rendered.push(...harness.process(8));
  assert.deepEqual(rendered.filter((sample) => sample !== 0), [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
  ]);
  assert.equal(harness.processor.sourceSamplePosition, 12);
  assert.equal(harness.processor.renderedOutputSamplePosition, 12);
});

test("force-pauses at the exact rendered output sample deadline", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3, 4, 5, 6]);
  harness.command("pause", 1, "pause_at_boundary", {
    requestedBoundarySourceSamplePosition: null,
    renderedOutputSampleDeadline: 3,
  });
  assert.deepEqual(harness.process(6), [1, 2, 3, 0, 0, 0]);
  assert.equal(harness.processor.state, "paused_buffered");
  assert.equal(harness.processor.renderedOutputSamplePosition, 3);
  assert.equal(harness.acknowledgement("pause").pauseResult, "forced_sample");
});

test("forced mid-word pause does not acknowledge the partial word", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3, 4, 5, 6]);
  harness.boundary(1, 3, 0);
  harness.boundary(1, 7, 4);
  harness.command("pause", 1, "pause_at_boundary", {
    requestedBoundarySourceSamplePosition: null,
    renderedOutputSampleDeadline: 3,
  });
  harness.process(4);
  assert.equal(
    harness.messages.filter((message) => message.type === "boundary.progress").length,
    0,
  );
  harness.command("resume", 1, "resume", {
    targetGain: 1,
    gainRampDurationMs: 1,
    maximumPausedAgeMs: 800,
  });
  harness.process(1);
  const progress = harness.messages.find(
    (message) => message.type === "boundary.progress",
  );
  assert.equal(progress.textOffset, 3);
  assert.equal(progress.playedSampleCount, 4);
});

test("a boundary registered while draining can satisfy the pause", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3, 4, 5, 6]);
  harness.command("pause", 1, "pause_at_boundary", {
    requestedBoundarySourceSamplePosition: null,
    renderedOutputSampleDeadline: 6,
  });
  harness.process(2);
  harness.boundary(1, 4, 4);
  harness.process(4);
  assert.equal(harness.processor.sourceSamplePosition, 4);
  assert.equal(harness.acknowledgement("pause").pauseResult, "word_boundary");
});

test("duck ramps gain and duplicate commands do not apply the ramp twice", () => {
  const harness = new PlaybackHarness(1_000);
  harness.enqueue(1, 0, Array.from({ length: 40 }, () => 10_000));
  harness.process(1);
  harness.command("duck", 1, "duck", {
    targetGain: 0.1258925,
    gainRampDurationMs: 20,
  });
  harness.process(20);
  const firstAcknowledgement = harness.acknowledgement("duck");
  assert.equal(firstAcknowledgement.gainRampComplete, true);
  assert.ok(Math.abs(firstAcknowledgement.currentGain - 0.1258925) < 1e-7);
  harness.command("duck", 1, "duck", {
    targetGain: 0.1258925,
    gainRampDurationMs: 20,
  });
  assert.equal(harness.processor.currentGain, firstAcknowledgement.currentGain);
  assert.equal(
    harness.messages.filter((message) => message.commandId === "duck").length,
    2,
  );
});

test("cancel while ducking prevents all later rendering", () => {
  const harness = new PlaybackHarness(1_000);
  harness.enqueue(1, 0, Array.from({ length: 20 }, (_, index) => index + 1));
  harness.process(2);
  harness.command("duck", 1, "duck", {
    targetGain: 0.1258925,
    gainRampDurationMs: 20,
  });
  harness.process(2);
  const stoppedAt = harness.processor.renderedOutputSamplePosition;
  harness.command("cancel", 1, "cancel");
  assert.deepEqual(harness.process(16), Array.from({ length: 16 }, () => 0));
  assert.equal(harness.processor.renderedOutputSamplePosition, stoppedAt);
  assert.equal(harness.processor.state, "cancelled");
});

test("cancel while paused discards buffered audio and rejects later audio", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3, 4, 5, 6]);
  harness.command("pause", 1, "pause_at_boundary", {
    requestedBoundarySourceSamplePosition: null,
    renderedOutputSampleDeadline: 2,
  });
  harness.process(4);
  harness.command("cancel", 1, "cancel");
  const acknowledgement = harness.acknowledgement("cancel");
  assert.equal(acknowledgement.discardedSourceSampleCount, 4);
  harness.enqueue(1, 2, [7, 8]);
  assert.equal(harness.processor.queuedSourceSampleCount, 0);
});

test("commands for replaced generations cannot affect current playback", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3]);
  harness.enqueue(2, 0, [4, 5, 6]);
  harness.command("old-cancel", 1, "cancel");
  assert.equal(harness.processor.generationId, 2);
  assert.notEqual(harness.processor.state, "cancelled");
  assert.equal(harness.acknowledgement("old-cancel"), undefined);
});

test("duplicate pause resume and cancel commands are idempotent", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2, 3, 4]);
  const pause = {
    requestedBoundarySourceSamplePosition: null,
    renderedOutputSampleDeadline: 1,
  };
  harness.command("pause", 1, "pause_at_boundary", pause);
  harness.process(2);
  harness.command("pause", 1, "pause_at_boundary", pause);
  assert.equal(harness.processor.sourceSamplePosition, 1);
  const resume = {
    targetGain: 1,
    gainRampDurationMs: 1,
    maximumPausedAgeMs: 800,
  };
  harness.command("resume", 1, "resume", resume);
  harness.process(1);
  harness.command("resume", 1, "resume", resume);
  assert.equal(harness.processor.sourceSamplePosition, 2);
  harness.command("cancel", 1, "cancel");
  const discarded = harness.acknowledgement("cancel").discardedSourceSampleCount;
  harness.command("cancel", 1, "cancel");
  assert.equal(harness.acknowledgement("cancel").discardedSourceSampleCount, discarded);
});

test("pause racing with end-of-stream completes terminally", () => {
  const harness = new PlaybackHarness();
  harness.enqueue(1, 0, [1, 2]);
  harness.command("pause", 1, "pause_at_boundary", {
    requestedBoundarySourceSamplePosition: null,
    renderedOutputSampleDeadline: 10,
  });
  harness.send({ type: "end", generationId: 1 });
  harness.process(4);
  assert.equal(harness.processor.state, "completed");
  assert.equal(
    harness.messages.filter((message) => message.type === "playback.complete").length,
    1,
  );
});

test("one generation accepts final-answer audio after a drained tool gap", () => {
  const harness = new PlaybackHarness();
  harness.boundary(1, 4, 0);
  harness.boundary(1, 9, 2);
  harness.enqueue(1, 0, [1, 2, 3, 4]);
  const rendered = harness.process(4);

  assert.equal(harness.processor.generationId, 1);
  assert.notEqual(harness.processor.state, "completed");
  assert.equal(
    harness.messages.filter((message) => message.type === "playback.complete").length,
    0,
  );

  harness.boundary(1, 15, 4);
  harness.boundary(1, 20, 6);
  harness.enqueue(1, 4, [5, 6, 7, 8]);
  harness.send({ type: "end", generationId: 1 });
  rendered.push(...harness.process(4));

  assert.deepEqual(rendered, [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(harness.processor.sourceSamplePosition, 8);
  assert.equal(harness.processor.state, "completed");
  assert.equal(
    harness.messages.filter((message) => message.type === "playback.started").length,
    1,
  );
  assert.equal(
    harness.messages.filter((message) => message.type === "playback.complete").length,
    1,
  );
  const textOffsets = harness.messages
    .filter((message) => message.type === "boundary.progress")
    .map((message) => message.textOffset);
  assert.deepEqual(textOffsets, [...textOffsets].sort((left, right) => left - right));
});

test("resume is rejected and playback is cancelled after the resumable age", () => {
  const harness = new PlaybackHarness(1_000);
  harness.enqueue(1, 0, [1, 2, 3, 4]);
  harness.command("pause", 1, "pause_at_boundary", {
    requestedBoundarySourceSamplePosition: null,
    renderedOutputSampleDeadline: 1,
  });
  harness.process(1);
  harness.process(801);
  harness.command("resume", 1, "resume", {
    targetGain: 1,
    gainRampDurationMs: 20,
    maximumPausedAgeMs: 800,
  });
  const acknowledgement = harness.acknowledgement("resume");
  assert.equal(acknowledgement.resumeRejected, true);
  assert.equal(acknowledgement.resultingState, "cancelled");
  assert.equal(harness.processor.state, "cancelled");
});
