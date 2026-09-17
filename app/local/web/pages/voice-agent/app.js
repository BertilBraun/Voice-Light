import { SpokenTextProgress } from "./spoken-text-progress.mjs";
import { PRODUCTION_VOICE_WEBSOCKET_URL } from "./public-config.mjs";
import {
  contiguousModelObservationSegments,
  INTERACTION_TIMELINE_DURATION_MS,
  modelObservationSamples,
  summarizeModelObservationCadence,
  updateInteractionEvidence as storeInteractionEvidence,
} from "./interaction-evidence.mjs";

const INPUT_SAMPLE_RATE = 16000;
const LOCAL_TIME_ZONE = Intl.DateTimeFormat().resolvedOptions().timeZone || "Etc/UTC";
const MAX_EVENT_LOG_ENTRIES = 200;
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const recordingReview = document.querySelector("#recording-review");
const recordingPlayer = document.querySelector("#recording-player");
const recordingDownload = document.querySelector("#recording-download");
const traceDownload = document.querySelector("#trace-download");
const connectionStatus = document.querySelector("#connection-status");
const sessionGuidance = document.querySelector("#session-guidance");
const vadStatus = document.querySelector("#vad-status");
const playbackStatus = document.querySelector("#playback-status");
const conversationHistory = document.querySelector("#conversation-history");
const conversationEmpty = document.querySelector("#conversation-empty");
const generatedTextToggle = document.querySelector("#generated-text-toggle");
const eventLog = document.querySelector("#event-log");
const debugAdapterStatus = document.querySelector("#debug-adapter-status");
const debugSilero = document.querySelector("#debug-silero");
const debugUserYield = document.querySelector("#debug-user-yield");
const debugTurnCompletion = document.querySelector("#debug-turn-completion");
const debugFloorTake = document.querySelector("#debug-floor-take");
const debugNonFloor = document.querySelector("#debug-non-floor");
const debugPolicyDecision = document.querySelector("#debug-policy-decision");
const debugPolicyLatency = document.querySelector("#debug-policy-latency");
const debugPredictionLatency = document.querySelector("#debug-prediction-latency");
const debugPredictionDisposition = document.querySelector("#debug-prediction-disposition");
const debugActionLatency = document.querySelector("#debug-action-latency");
const debugEvidenceCadence = document.querySelector("#debug-evidence-cadence");
const interactionTimeline = document.querySelector("#interaction-timeline");

let socket;
let microphoneStream;
let captureContext;
let playbackContext;
let playbackNode;
let stopRequested = false;
let cancelledGenerationId = -1;
let audioGenerationId = -1;
let expectedAudioSequence = 0;
let activeUserTurn;
let recordedInputChunks = [];
let recordingUrl;
let sessionTraceStartedAt;
let sessionTraceEvents = [];
let sessionTraceId;
let traceUrl;
const assistantTurns = new Map();
const intentionallyClosedSockets = new WeakSet();
const interactionEvidence = new Map();
let interactionTimelineFrame;

new ResizeObserver(scheduleInteractionTimelineDraw).observe(interactionTimeline);

class ConversationTurn {
  constructor(role, state) {
    conversationEmpty.remove();
    this.element = document.createElement("article");
    this.element.className = "conversation-turn";
    this.element.dataset.role = role;
    this.element.dataset.state = state;

    const heading = document.createElement("div");
    heading.className = "turn-heading";
    const speaker = document.createElement("span");
    speaker.className = "turn-speaker";
    speaker.textContent = role === "user" ? "You" : "Assistant";
    this.meta = document.createElement("span");
    this.meta.className = "turn-meta";
    this.state = document.createElement("span");
    this.state.className = "turn-state";
    this.latencies = document.createElement("span");
    this.latencies.className = "turn-latencies";
    this.meta.append(this.state, this.latencies);
    heading.append(speaker, this.meta);

    this.transcript = document.createElement("p");
    this.transcript.className = "turn-transcript";
    this.progress = new SpokenTextProgress();
    if (role === "assistant") {
      this.spokenTranscript = document.createElement("span");
      this.spokenTranscript.className = "turn-spoken";
      this.unspokenTranscript = document.createElement("span");
      this.unspokenTranscript.className = "turn-unspoken";
      this.transcript.append(this.spokenTranscript, this.unspokenTranscript);
    }
    this.element.append(heading, this.transcript);
    conversationHistory.append(this.element);
    this.setState(state);
    followConversationHistory(true);
  }

  setText(text) {
    const followHistory = historyIsAtEnd();
    this.progress.replaceText(text);
    this.renderText();
    followConversationHistory(followHistory);
  }

  appendText(text) {
    const followHistory = historyIsAtEnd();
    this.progress.appendText(text);
    this.renderText();
    followConversationHistory(followHistory);
  }

  setSpokenOffset(offset) {
    if (!this.spokenTranscript || offset <= this.progress.spokenOffset) return;
    const followHistory = historyIsAtEnd();
    this.progress.markSpoken(offset);
    this.renderText();
    followConversationHistory(followHistory);
  }

  acknowledgeOffset(offset) {
    this.progress.acknowledge(offset);
  }

  settleInterruptedText() {
    const followHistory = historyIsAtEnd();
    this.progress.settleInterruptedText();
    this.renderText();
    followConversationHistory(followHistory);
  }

  renderText() {
    if (!this.spokenTranscript) {
      this.transcript.textContent = this.progress.text;
      return;
    }
    this.spokenTranscript.textContent = this.progress.spokenText();
    this.unspokenTranscript.textContent = this.progress.unspokenText();
  }

  setState(state) {
    this.element.dataset.state = state;
    this.state.textContent = stateLabel(state);
  }

  setLatencies(latencies) {
    const measurements = latencies.final_vad_endpoint_to_first_audio_send_ms === null
      ? []
      : [{
          label: "response",
          value: latencies.final_vad_endpoint_to_first_audio_send_ms,
          description: "Final speech endpoint to the first released audio packet.",
        }];
    this.latencies.replaceChildren(
      ...measurements.map(({ label, value, description }) => {
        const measurement = document.createElement("span");
        measurement.className = "turn-latency";
        measurement.title = description;
        measurement.textContent = `${label} ${formatLatency(value)}`;
        return measurement;
      }),
    );
  }

  get text() {
    return this.progress.text;
  }
}

startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);
generatedTextToggle.addEventListener("change", () => {
  document.body.dataset.showGeneratedText = String(generatedTextToggle.checked);
});

async function startSession() {
  clearInputRecording();
  clearSessionTrace();
  clearConversationHistory();
  stopRequested = false;
  startButton.disabled = true;
  startButton.textContent = "Starting…";
  stopButton.disabled = false;
  setConnection("starting", "Server starting…", "Waking the server. This can take about a minute after it has scaled down.");
  try {
    socket = await openSocket(PRODUCTION_VOICE_WEBSOCKET_URL);
    setConnection("connected", "Preparing session…", "The server is connected, but the microphone is not ready yet.");
    const sessionReady = waitForSessionReady(socket);
    sendClientEvent({
      type: "session.start",
      input_sample_rate: INPUT_SAMPLE_RATE,
      local_time_zone: LOCAL_TIME_ZONE,
    });
    const ready = await sessionReady;
    await setupPlayback(ready.output_sample_rate);
    if (stopRequested) return;
    setConnection("connected", "Connecting microphone…", "Allow microphone access if your browser asks for it.");
    microphoneStream = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    if (stopRequested) {
      microphoneStream.getTracks().forEach((track) => track.stop());
      return;
    }
    await setupCapture(microphoneStream);
    if (stopRequested) return;
    startButton.textContent = "Microphone active";
    vadStatus.textContent = "ready";
    setConnection("ready", "Ready to talk", "Ready — you can speak now.");
  } catch (error) {
    await stopMedia();
    if (socket) {
      intentionallyClosedSockets.add(socket);
      socket.close();
    }
    resetControls();
    if (stopRequested) setConnection("idle", "Disconnected", "Press Start microphone to wake the server.");
    else setConnection("error", "Connection problem", error.message);
  }
}

function openSocket(endpoint) {
  return new Promise((resolve, reject) => {
    const candidate = new WebSocket(endpoint);
    let opened = false;
    socket = candidate;
    candidate.binaryType = "arraybuffer";
    candidate.addEventListener("open", () => { opened = true; resolve(candidate); }, { once: true });
    candidate.addEventListener("error", () => reject(new Error("WebSocket connection failed.")), { once: true });
    candidate.addEventListener("message", handleMessage);
    candidate.addEventListener("close", () => {
      if (!opened) reject(new Error("The server connection closed before it was ready."));
      void stopMedia().then(() => {
        finalizeInputRecording();
        finalizeSessionTrace();
      });
      resetControls();
      if (intentionallyClosedSockets.has(candidate)) return;
      if (stopRequested) setConnection("idle", "Disconnected", "Press Start microphone to wake the server.");
      else setConnection("error", "Connection closed", "The server connection closed unexpectedly. Start again to reconnect.");
    });
  });
}

function waitForSessionReady(candidate) {
  return new Promise((resolve, reject) => {
    function onMessage(event) {
      if (event.data instanceof ArrayBuffer) return;
      const message = JSON.parse(event.data);
      if (message.type === "session.ready") {
        cleanup();
        resolve(message);
      } else if (message.type === "error") {
        cleanup();
        reject(new Error(message.message));
      }
    }
    function onClose() { cleanup(); reject(new Error("The server closed before the session was ready.")); }
    function cleanup() {
      candidate.removeEventListener("message", onMessage);
      candidate.removeEventListener("close", onClose);
    }
    candidate.addEventListener("message", onMessage);
    candidate.addEventListener("close", onClose);
  });
}

async function setupCapture(stream) {
  captureContext = new AudioContext({ sampleRate: INPUT_SAMPLE_RATE });
  if (captureContext.sampleRate !== INPUT_SAMPLE_RATE) {
    throw new Error(`Browser created a ${captureContext.sampleRate} Hz capture context instead of ${INPUT_SAMPLE_RATE} Hz.`);
  }
  await captureContext.audioWorklet.addModule("./capture-worklet.js?v=2");
  const source = captureContext.createMediaStreamSource(stream);
  const captureNode = new AudioWorkletNode(captureContext, "pcm-capture", {
    processorOptions: { targetSampleRate: INPUT_SAMPLE_RATE },
  });
  const silentGain = captureContext.createGain();
  silentGain.gain.value = 0;
  console.info("Voice input capture", {
    audioContextSampleRate: captureContext.sampleRate,
    microphoneTrackSettings: stream.getAudioTracks()[0].getSettings(),
  });
  captureNode.port.onmessage = ({ data }) => {
    if (socket?.readyState !== WebSocket.OPEN) return;
    recordedInputChunks.push(data.slice(0));
    socket.send(data);
  };
  source.connect(captureNode).connect(silentGain).connect(captureContext.destination);
}

async function setupPlayback(inputSampleRate) {
  playbackContext = new AudioContext();
  await playbackContext.audioWorklet.addModule("./playback-worklet.js?v=9");
  playbackNode = new AudioWorkletNode(playbackContext, "pcm-playback", {
    outputChannelCount: [1],
    processorOptions: { inputSampleRate },
  });
  playbackNode.port.onmessage = ({ data }) => {
    if (data.type === "playback.started") {
      console.info(`Playback started for generation ${data.generationId}`, {
        generationId: data.generationId,
        clientTimeMs: performance.now(),
      });
      if (socket?.readyState === WebSocket.OPEN) {
        sendClientEvent({
          type: "playback.started",
          generation_id: data.generationId,
          browser_monotonic_time_ns: data.browserMonotonicTimeNs,
          rendered_output_sample_position: data.renderedOutputSamplePosition,
          source_sample_position: data.sourceSamplePosition,
          output_sample_rate: data.outputSampleRate,
        });
      }
      return;
    }
    if (data.type === "boundary.progress") {
      updateBoundaryProgress(data);
      return;
    }
    if (data.type === "playback.clock") {
      if (socket?.readyState === WebSocket.OPEN) {
        sendClientEvent({
          type: "playback.clock",
          generation_id: data.generationId,
          state: data.state,
          browser_monotonic_time_ns: data.browserMonotonicTimeNs,
          rendered_output_sample_position: data.renderedOutputSamplePosition,
          source_sample_position: data.sourceSamplePosition,
          queued_source_sample_count: data.queuedSourceSampleCount,
          underrun_count: data.underrunCount,
          output_sample_rate: data.outputSampleRate,
        });
      }
      return;
    }
    if (data.type === "boundary.started") {
      assistantTurns.get(data.generationId)?.setSpokenOffset(data.textOffset);
      return;
    }
    if (data.type === "playback.stopped") {
      const turn = assistantTurns.get(data.generationId);
      turn?.setSpokenOffset(data.textOffset);
      turn?.acknowledgeOffset(data.textOffset);
      turn?.settleInterruptedText();
      turn?.setState("cancelled");
      if (socket?.readyState === WebSocket.OPEN) {
        sendClientEvent({
          type: "playback.stopped",
          generation_id: data.generationId,
          text_offset: data.textOffset,
          played_sample_count: data.playedSampleCount,
          browser_monotonic_time_ns: data.browserMonotonicTimeNs,
          rendered_output_sample_position: data.renderedOutputSamplePosition,
          output_sample_rate: data.outputSampleRate,
        });
      }
      return;
    }
    if (
      data.type === "playback.complete" &&
      data.generationId > 0 &&
      socket?.readyState === WebSocket.OPEN
    ) {
      const turn = assistantTurns.get(data.generationId);
      const completeOffset = characterLength(turn?.text ?? "");
      turn?.setSpokenOffset(completeOffset);
      turn?.acknowledgeOffset(completeOffset);
      turn?.setState("complete");
      sendClientEvent({
        type: "playback.complete",
        generation_id: data.generationId,
        browser_monotonic_time_ns: data.browserMonotonicTimeNs,
        rendered_output_sample_position: data.renderedOutputSamplePosition,
        source_sample_position: data.sourceSamplePosition,
        output_sample_rate: data.outputSampleRate,
      });
      vadStatus.textContent = "ready";
      playbackStatus.textContent = "waiting";
      return;
    }
    if (data.type === "playback.acknowledgement") {
      if (socket?.readyState !== WebSocket.OPEN) return;
      sendClientEvent({
        type: "playback.acknowledgement",
        command_id: data.commandId,
        generation_id: data.generationId,
        action: data.action,
        stream_epoch: data.streamEpoch,
        turn_epoch: data.turnEpoch,
        resulting_state: data.resultingState,
        browser_monotonic_time_ns: data.browserMonotonicTimeNs,
        rendered_output_sample_position: data.renderedOutputSamplePosition,
        source_sample_position: data.sourceSamplePosition,
        output_sample_rate: data.outputSampleRate,
        pause_result: data.pauseResult,
        current_gain: data.currentGain,
        gain_ramp_complete: data.gainRampComplete,
        queued_source_sample_count: data.queuedSourceSampleCount,
        discarded_source_sample_count: data.discardedSourceSampleCount,
        replayed_source_sample_count: data.replayedSourceSampleCount,
        skipped_source_sample_count: data.skippedSourceSampleCount,
        resume_rejected: data.resumeRejected,
      });
    }
  };
  playbackNode.connect(playbackContext.destination);
  await playbackContext.resume();
}

function handleMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    if (playbackContext?.state === "suspended") void playbackContext.resume();
    const view = new DataView(event.data);
    const generationId = view.getUint32(0, true);
    const sequenceNumber = view.getUint32(4, true);
    const startSample = view.getUint32(8, true);
    if (generationId <= cancelledGenerationId) return;
    if (generationId !== audioGenerationId) {
      audioGenerationId = generationId;
      expectedAudioSequence = 0;
    }
    if (sequenceNumber !== expectedAudioSequence) return;
    expectedAudioSequence += 1;
    const pcm = event.data.slice(12);
    playbackNode.port.postMessage({ type: "audio", generationId, startSample, pcm }, [pcm]);
    return;
  }
  const message = JSON.parse(event.data);
  recordSessionTraceEvent("server", message);
  if (message.type === "session.ready") sessionTraceId = message.session_id;
  if (message.type !== "speech_understanding.debug") logEvent(message);
  if (message.type === "vad.started") vadStatus.textContent = "speaking";
  if (message.type === "vad.stopped") {
    vadStatus.textContent = "thinking";
  }
  if (message.type === "speech_understanding.debug") {
    updateInteractionEvidence(message);
  }
  if (message.type === "interaction_policy.debug") {
    debugPolicyDecision.textContent = `${message.decision} · ${message.reason}`;
    debugPolicyLatency.textContent = `${message.decision_latency_ms.toFixed(1)} ms`;
    debugPredictionLatency.textContent =
      message.first_applicable_prediction_latency_ms === null
        ? "none"
        : `${message.first_applicable_prediction_latency_ms.toFixed(1)} ms`;
  }
  if (message.type === "interaction_action.debug") {
    debugActionLatency.textContent =
      `${message.action} · ${message.onset_to_acknowledgement_ms.toFixed(1)} ms`;
  }
  if (message.type === "transcript.partial" || message.type === "transcript.final") {
    updateUserDraft(message.text);
  }
  if (message.type === "turn.committed") commitUserTurn(message.text);
  if (message.type === "llm.history") {
    console.groupCollapsed(`Audible history for generation ${message.generation_id}`);
    console.table(message.messages);
    console.log(JSON.stringify(message.messages, null, 2));
    console.groupEnd();
  }
  if (message.type === "llm.model_request") {
    const mode = message.speculative ? "speculative" : "committed";
    console.groupCollapsed(
      `Exact Qwen request for generation ${message.generation_id}, invocation ${message.invocation_index} (${mode})`,
    );
    console.table(message.messages);
    console.log(
      JSON.stringify({ messages: message.messages, tools: message.tools }, null, 2),
    );
    console.groupEnd();
  }
  if (message.type === "search.debug") {
    console.groupCollapsed(`Search trace for generation ${message.generation_id}`);
    console.table({
      provider: { durationMs: message.provider_duration_ms },
      summarizer: { durationMs: message.summarizer_duration_ms },
      total: { durationMs: message.total_duration_ms },
    });
    console.log("Query", message.query);
    console.table(message.results);
    console.groupCollapsed("Isolated Qwen summarizer request");
    console.log("System prompt", message.summarizer_system_prompt);
    console.log("User prompt", message.summarizer_user_prompt);
    console.groupEnd();
    console.log("Isolated Qwen summary / main-agent tool result", message.summary);
    console.groupEnd();
  }
  if (message.type === "assistant.text.delta") {
    const turn = assistantTurn(message.generation_id);
    turn.appendText(message.text);
    turn.setState("streaming");
    playbackStatus.textContent = "generating";
  }
  if (message.type === "assistant.audio.start") {
    if (playbackContext?.state === "suspended") void playbackContext.resume();
    assistantTurn(message.generation_id).setState("speaking");
    playbackStatus.textContent = "speaking";
  }
  if (message.type === "assistant.audio.end") {
    playbackNode.port.postMessage({ type: "end", generationId: message.generation_id });
    assistantTurn(message.generation_id).setState("speaking");
    playbackStatus.textContent = "finishing";
  }
  if (message.type === "assistant.audio.text_boundary") {
    if (message.generation_id <= cancelledGenerationId) return;
    playbackNode.port.postMessage({
      type: "boundary",
      generationId: message.generation_id,
      textOffset: message.text_offset,
      startSample: message.start_sample,
    });
  }
  if (message.type === "assistant.latency") {
    assistantTurn(message.generation_id).setLatencies(message);
  }
  if (message.type === "playback.command") {
    if (message.action === "cancel") {
      cancelledGenerationId = Math.max(cancelledGenerationId, message.generation_id);
      playbackStatus.textContent = "cancelled";
    } else if (message.action === "duck") {
      playbackStatus.textContent = "ducking";
    } else if (message.action === "pause_at_boundary") {
      playbackStatus.textContent = "pausing";
    } else if (message.action === "resume") {
      playbackStatus.textContent = "resuming";
    }
    playbackNode.port.postMessage({
      type: "playback.command",
      commandId: message.command_id,
      generationId: message.generation_id,
      action: message.action,
      issuedMonotonicTimeNs: message.issued_monotonic_time_ns,
      causalEventId: message.causal_event_id,
      causalSource: message.causal_source,
      streamEpoch: message.stream_epoch,
      turnEpoch: message.turn_epoch,
      confidence: message.confidence,
      requestedBoundarySourceSamplePosition:
        message.requested_boundary_source_sample_position,
      renderedOutputSampleDeadline: message.rendered_output_sample_deadline,
      targetGain: message.target_gain,
      gainRampDurationMs: message.gain_ramp_duration_ms,
      maximumPausedAgeMs: message.maximum_paused_age_ms,
    });
  }
  if (message.type === "assistant.cancel") {
    cancelledGenerationId = Math.max(cancelledGenerationId, message.generation_id);
    vadStatus.textContent = "ready";
    playbackStatus.textContent = "cancelled";
  }
  if (message.type === "error") {
    const generation = message.generation_id === null ? "session" : message.generation_id;
    const failureDetail = `${message.component}.${message.operation} generation=${generation} retryable=${message.retryable}: ${message.message}`;
    console.error("Voice component failure", message);
    vadStatus.textContent = "ready";
    setConnection("error", "Server error", failureDetail);
  }
}

async function stopSession() {
  stopRequested = true;
  setConnection("connected", "Stopping…", "Closing the microphone and server connection.");
  if (socket?.readyState === WebSocket.OPEN) sendClientEvent({ type: "session.stop" });
  if (socket) {
    intentionallyClosedSockets.add(socket);
    socket.close();
  }
  await stopMedia();
  finalizeInputRecording();
  finalizeSessionTrace();
  resetControls();
  setConnection("idle", "Disconnected", "Press Start microphone to wake the server.");
}

async function stopMedia() {
  microphoneStream?.getTracks().forEach((track) => track.stop());
  microphoneStream = undefined;
  if (captureContext && captureContext.state !== "closed") await captureContext.close();
  if (playbackContext && playbackContext.state !== "closed") await playbackContext.close();
  captureContext = undefined;
  playbackContext = undefined;
}

function resetControls() {
  cancelledGenerationId = -1;
  audioGenerationId = -1;
  expectedAudioSequence = 0;
  startButton.disabled = false;
  startButton.textContent = "Start microphone";
  stopButton.disabled = true;
  vadStatus.textContent = "waiting";
  playbackStatus.textContent = "waiting";
  debugSilero.textContent = "waiting";
  debugAdapterStatus.textContent = "waiting";
  debugTurnCompletion.textContent = "—";
  debugFloorTake.textContent = "—";
  debugNonFloor.textContent = "—";
  debugPolicyDecision.textContent = "waiting";
  debugPolicyLatency.textContent = "—";
}

function formatProbability(value) {
  if (value === null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function clearInputRecording() {
  recordedInputChunks = [];
  if (recordingUrl) URL.revokeObjectURL(recordingUrl);
  recordingUrl = undefined;
  recordingPlayer.removeAttribute("src");
  recordingPlayer.load();
  recordingDownload.removeAttribute("href");
  recordingReview.hidden = true;
}

function clearSessionTrace() {
  sessionTraceStartedAt = new Date().toISOString();
  sessionTraceEvents = [];
  sessionTraceId = undefined;
  if (traceUrl) URL.revokeObjectURL(traceUrl);
  traceUrl = undefined;
  traceDownload.removeAttribute("href");
}

function recordSessionTraceEvent(direction, message) {
  sessionTraceEvents.push({
    direction,
    browser_monotonic_time_ms: performance.now(),
    message,
  });
}

function sendClientEvent(message) {
  if (socket?.readyState !== WebSocket.OPEN) return;
  recordSessionTraceEvent("client", message);
  socket.send(JSON.stringify(message));
}

function finalizeSessionTrace() {
  if (traceUrl || sessionTraceEvents.length === 0) return;
  const trace = {
    schema_version: 1,
    session_id: sessionTraceId ?? null,
    started_at: sessionTraceStartedAt,
    ended_at: new Date().toISOString(),
    local_time_zone: LOCAL_TIME_ZONE,
    browser_user_agent: navigator.userAgent,
    events: sessionTraceEvents,
  };
  traceUrl = URL.createObjectURL(
    new Blob([JSON.stringify(trace, null, 2)], { type: "application/json" }),
  );
  traceDownload.href = traceUrl;
  const timestamp = new Date().toISOString().replaceAll(":", "-");
  traceDownload.download = `voice-light-session-${timestamp}.json`;
}

function finalizeInputRecording() {
  if (recordingUrl || recordedInputChunks.length === 0) return;
  const recording = createPcmWav(recordedInputChunks, INPUT_SAMPLE_RATE);
  recordingUrl = URL.createObjectURL(recording);
  recordingPlayer.src = recordingUrl;
  recordingDownload.href = recordingUrl;
  recordingDownload.download = `voice-light-input-${new Date().toISOString().replaceAll(":", "-")}.wav`;
  recordingReview.hidden = false;
}

function createPcmWav(pcmChunks, sampleRate) {
  const dataByteCount = pcmChunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataByteCount, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataByteCount, true);
  return new Blob([header, ...pcmChunks], { type: "audio/wav" });
}

function writeAscii(view, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}
function setConnection(state, text, guidance) { connectionStatus.dataset.state = state; connectionStatus.textContent = text; sessionGuidance.dataset.state = state; sessionGuidance.textContent = guidance; }
function logEvent(message) {
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()} ${message.type}`;
  eventLog.prepend(item);
  while (eventLog.childElementCount > MAX_EVENT_LOG_ENTRIES) {
    eventLog.lastElementChild.remove();
  }
}

function updateUserDraft(text) {
  if (!activeUserTurn) activeUserTurn = new ConversationTurn("user", "transcribing");
  activeUserTurn.setText(text);
}

function commitUserTurn(text) {
  updateUserDraft(text);
  activeUserTurn.setState("committed");
  activeUserTurn = undefined;
}

function assistantTurn(generationId) {
  let turn = assistantTurns.get(generationId);
  if (!turn) {
    turn = new ConversationTurn("assistant", "streaming");
    assistantTurns.set(generationId, turn);
  }
  return turn;
}

function clearConversationHistory() {
  activeUserTurn = undefined;
  assistantTurns.clear();
  conversationEmpty.hidden = false;
  conversationHistory.replaceChildren(conversationEmpty);
  interactionEvidence.clear();
  debugEvidenceCadence.textContent = "no model observations";
  debugPredictionDisposition.textContent = "—";
  scheduleInteractionTimelineDraw();
}

function updateInteractionEvidence(message) {
  const hasModelEvidence = message.turn_completion_probability !== null;
  const newestAudioTimeMs = storeInteractionEvidence(interactionEvidence, message);
  debugAdapterStatus.textContent = message.adapter_status.replaceAll("_", " ");
  if (message.observed_audio_time_ms === newestAudioTimeMs) {
    debugSilero.textContent = message.silero_speech ? "speech" : "silence";
  }
  if (hasModelEvidence) {
    debugUserYield.textContent = formatProbability(message.user_yield_probability);
    debugTurnCompletion.textContent = formatProbability(message.turn_completion_probability);
    debugFloorTake.textContent = formatProbability(message.floor_take_probability);
    debugNonFloor.textContent = formatProbability(message.non_floor_feedback_probability);
    debugPredictionDisposition.textContent = message.prediction_disposition.replaceAll("_", " ");
  }
  const points = [...interactionEvidence.values()].sort(
    (left, right) => left.audioTimeMs - right.audioTimeMs,
  );
  const cadence = summarizeModelObservationCadence(points);
  if (cadence === null) {
    debugEvidenceCadence.textContent = "no model observations";
  } else {
    const interval = cadence.medianIntervalMs === null
      ? "one sample"
      : `${cadence.medianIntervalMs} ms median`;
    debugEvidenceCadence.textContent = `${interval} · latest ${cadence.ageMs} ms behind input`;
  }
  scheduleInteractionTimelineDraw();
}

function scheduleInteractionTimelineDraw() {
  if (interactionTimelineFrame !== undefined) return;
  interactionTimelineFrame = requestAnimationFrame(() => {
    interactionTimelineFrame = undefined;
    drawInteractionTimeline();
  });
}

function drawInteractionTimeline() {
  const width = Math.max(interactionTimeline.clientWidth, 320);
  const height = interactionTimeline.clientHeight;
  const pixelRatio = window.devicePixelRatio || 1;
  const backingWidth = Math.round(width * pixelRatio);
  const backingHeight = Math.round(height * pixelRatio);
  if (interactionTimeline.width !== backingWidth) interactionTimeline.width = backingWidth;
  if (interactionTimeline.height !== backingHeight) interactionTimeline.height = backingHeight;
  const context = interactionTimeline.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const points = [...interactionEvidence.values()].sort(
    (left, right) => left.audioTimeMs - right.audioTimeMs,
  );
  const rightTimeMs = points.at(-1)?.audioTimeMs ?? INTERACTION_TIMELINE_DURATION_MS;
  const leftTimeMs = Math.max(0, rightTimeMs - INTERACTION_TIMELINE_DURATION_MS);
  const labelWidth = 82;
  const plotRight = width - 12;
  const plotWidth = plotRight - labelWidth;
  const laneHeight = 18;
  const userLaneTop = 12;
  const assistantLaneTop = 36;
  const probabilityTop = 70;
  const probabilityBottom = height - 24;

  context.font = "11px ui-monospace, monospace";
  context.fillStyle = "#64706a";
  context.textBaseline = "middle";
  context.fillText("user speech", 8, userLaneTop + laneHeight / 2);
  context.fillText("assistant", 8, assistantLaneTop + laneHeight / 2);
  for (const [label, probability] of [["1", 1], [".5", 0.5], ["0", 0]]) {
    const y = probabilityBottom - probability * (probabilityBottom - probabilityTop);
    context.strokeStyle = "#d9e0dc";
    context.beginPath();
    context.moveTo(labelWidth, y);
    context.lineTo(plotRight, y);
    context.stroke();
    context.fillText(label, labelWidth - 24, y);
  }

  const xForTime = (audioTimeMs) =>
    labelWidth + ((audioTimeMs - leftTimeMs) / INTERACTION_TIMELINE_DURATION_MS) * plotWidth;
  const frameWidth = Math.max((80 / INTERACTION_TIMELINE_DURATION_MS) * plotWidth, 1);
  for (const point of points) {
    const x = xForTime(point.audioTimeMs);
    if (x < labelWidth || x > plotRight) continue;
    if (point.sileroSpeech) {
      context.fillStyle = "#4ea878";
      context.fillRect(x - frameWidth, userLaneTop, frameWidth, laneHeight);
    }
    if (point.assistantAudible) {
      context.fillStyle = "#d4a94f";
      context.fillRect(x - frameWidth, assistantLaneTop, frameWidth, laneHeight);
    }
  }
  drawProbabilitySeries(context, points, xForTime, probabilityTop, probabilityBottom, "userYield", "#e7bf5f");
  drawProbabilitySeries(context, points, xForTime, probabilityTop, probabilityBottom, "turnCompletion", "#66e3a4");
  drawProbabilitySeries(context, points, xForTime, probabilityTop, probabilityBottom, "floorTake", "#ff796f");
  drawProbabilitySeries(context, points, xForTime, probabilityTop, probabilityBottom, "nonFloorFeedback", "#77a9ff");

  context.fillStyle = "#64706a";
  context.textBaseline = "alphabetic";
  context.fillText("−20s", labelWidth, height - 7);
  context.textAlign = "right";
  context.fillText("now", plotRight, height - 7);
  context.textAlign = "left";
}

function drawProbabilitySeries(context, points, xForTime, top, bottom, field, color) {
  const samples = modelObservationSamples(points, field);
  for (const segment of contiguousModelObservationSegments(samples)) {
    context.beginPath();
    for (const [index, sample] of segment.entries()) {
      const x = xForTime(sample.audioTimeMs);
      const y = bottom - sample.probability * (bottom - top);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.strokeStyle = color;
    context.globalAlpha = 0.55;
    context.lineWidth = 1.25;
    context.setLineDash(segment[0].disposition === "applicable" ? [] : [3, 3]);
    context.stroke();
  }
  context.globalAlpha = 1;
  context.setLineDash([]);
  for (const sample of samples) {
    const x = xForTime(sample.audioTimeMs);
    const y = bottom - sample.probability * (bottom - top);
    context.beginPath();
    context.arc(x, y, 2.5, 0, 2 * Math.PI);
    if (sample.disposition === "applicable") {
      context.fillStyle = color;
      context.fill();
    } else {
      context.strokeStyle = color;
      context.lineWidth = 1.25;
      context.stroke();
    }
  }
}

function updateBoundaryProgress(progress) {
  const turn = assistantTurns.get(progress.generationId);
  if (!turn) return;
  turn.setSpokenOffset(progress.textOffset);
  turn.acknowledgeOffset(progress.textOffset);
  if (socket?.readyState === WebSocket.OPEN) {
    sendClientEvent({
      type: "playback.progress",
      generation_id: progress.generationId,
      text_offset: progress.textOffset,
      boundary_start_sample: progress.startSample,
      played_sample_count: progress.playedSampleCount,
      browser_monotonic_time_ns: progress.browserMonotonicTimeNs,
      rendered_output_sample_position: progress.renderedOutputSamplePosition,
      output_sample_rate: progress.outputSampleRate,
    });
  }
}

function characterLength(text) {
  return Array.from(text).length;
}

function formatLatency(milliseconds) {
  if (milliseconds < 10) return `${milliseconds.toFixed(1)} ms`;
  return `${Math.round(milliseconds)} ms`;
}

function historyIsAtEnd() {
  const remainingScroll = conversationHistory.scrollHeight - conversationHistory.scrollTop - conversationHistory.clientHeight;
  return remainingScroll < 80;
}

function followConversationHistory(shouldFollow) {
  if (!shouldFollow) return;
  requestAnimationFrame(() => {
    conversationHistory.scrollTop = conversationHistory.scrollHeight;
  });
}

function stateLabel(state) {
  switch (state) {
    case "transcribing": return "transcribing";
    case "committed": return "heard";
    case "streaming": return "generating";
    case "speaking": return "speaking";
    case "complete": return "complete";
    case "cancelled": return "interrupted";
    default: throw new Error(`Unknown conversation turn state: ${state}`);
  }
}
