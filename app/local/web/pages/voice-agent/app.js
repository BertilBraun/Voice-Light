const INPUT_SAMPLE_RATE = 16000;
const ENDPOINT_STORAGE_KEY = "voice-light-compute-voice-endpoint";
const endpointInput = document.querySelector("#endpoint-url");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const connectionStatus = document.querySelector("#connection-status");
const sessionGuidance = document.querySelector("#session-guidance");
const vadStatus = document.querySelector("#vad-status");
const playbackStatus = document.querySelector("#playback-status");
const conversationHistory = document.querySelector("#conversation-history");
const conversationEmpty = document.querySelector("#conversation-empty");
const eventLog = document.querySelector("#event-log");

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
const assistantTurns = new Map();
const intentionallyClosedSockets = new WeakSet();

class ConversationTurn {
  constructor(role, state) {
    const followHistory = historyIsAtEnd();
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
    heading.append(speaker, this.meta);

    this.transcript = document.createElement("p");
    this.transcript.className = "turn-transcript";
    this.element.append(heading, this.transcript);
    conversationHistory.append(this.element);
    this.setState(state);
    followConversationHistory(followHistory);
  }

  setText(text) {
    const followHistory = historyIsAtEnd();
    this.transcript.textContent = text;
    followConversationHistory(followHistory);
  }

  appendText(text) {
    const followHistory = historyIsAtEnd();
    this.transcript.textContent += text;
    followConversationHistory(followHistory);
  }

  setState(state) {
    this.element.dataset.state = state;
    this.meta.textContent = stateLabel(state);
  }
}

endpointInput.value = new URLSearchParams(location.search).get("compute") ?? localStorage.getItem(ENDPOINT_STORAGE_KEY) ?? "";
startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);

async function startSession() {
  const endpoint = endpointInput.value.trim();
  if (!endpoint.startsWith("wss://") && !endpoint.startsWith("ws://")) {
    setConnection("error", "Invalid endpoint", "Enter a WebSocket URL beginning with ws:// or wss://.");
    return;
  }
  localStorage.setItem(ENDPOINT_STORAGE_KEY, endpoint);
  clearConversationHistory();
  stopRequested = false;
  startButton.disabled = true;
  startButton.textContent = "Starting…";
  stopButton.disabled = false;
  setConnection("starting", "Server starting…", "Waking the server. This can take about a minute after it has scaled down.");
  try {
    socket = await openSocket(endpoint);
    setConnection("connected", "Preparing session…", "The server is connected, but the microphone is not ready yet.");
    const sessionReady = waitForSessionReady(socket);
    socket.send(JSON.stringify({ type: "session.start", input_sample_rate: INPUT_SAMPLE_RATE }));
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
      void stopMedia();
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
  captureContext = new AudioContext();
  await captureContext.audioWorklet.addModule("/pages/voice-agent/capture-worklet.js");
  const source = captureContext.createMediaStreamSource(stream);
  const captureNode = new AudioWorkletNode(captureContext, "pcm-capture", { processorOptions: { targetSampleRate: INPUT_SAMPLE_RATE } });
  const silentGain = captureContext.createGain();
  silentGain.gain.value = 0;
  captureNode.port.onmessage = ({ data }) => {
    if (socket?.readyState === WebSocket.OPEN) socket.send(data);
  };
  source.connect(captureNode).connect(silentGain).connect(captureContext.destination);
}

async function setupPlayback(inputSampleRate) {
  playbackContext = new AudioContext();
  await playbackContext.audioWorklet.addModule("/pages/voice-agent/playback-worklet.js");
  playbackNode = new AudioWorkletNode(playbackContext, "pcm-playback", {
    outputChannelCount: [1],
    processorOptions: { inputSampleRate },
  });
  playbackNode.port.onmessage = ({ data }) => {
    if (
      data.type === "playback.complete" &&
      data.generationId > 0 &&
      socket?.readyState === WebSocket.OPEN
    ) {
      socket.send(JSON.stringify({ type: "playback.complete", generation_id: data.generationId }));
      vadStatus.textContent = "ready";
      playbackStatus.textContent = "waiting";
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
    if (generationId <= cancelledGenerationId) return;
    if (generationId !== audioGenerationId) {
      audioGenerationId = generationId;
      expectedAudioSequence = 0;
    }
    if (sequenceNumber !== expectedAudioSequence) return;
    expectedAudioSequence += 1;
    const pcm = event.data.slice(8);
    playbackNode.port.postMessage({ type: "audio", generationId, pcm }, [pcm]);
    return;
  }
  const message = JSON.parse(event.data);
  logEvent(message);
  if (message.type === "vad.started") vadStatus.textContent = "speaking";
  if (message.type === "vad.stopped") {
    vadStatus.textContent = "thinking";
  }
  if (message.type === "transcript.partial" || message.type === "transcript.final") {
    updateUserDraft(message.text);
  }
  if (message.type === "turn.committed") commitUserTurn(message.text);
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
    assistantTurn(message.generation_id).setState("complete");
    playbackStatus.textContent = "finishing";
  }
  if (message.type === "assistant.cancel") {
    cancelledGenerationId = Math.max(cancelledGenerationId, message.generation_id);
    playbackNode.port.postMessage({ type: "clear", generationId: message.generation_id });
    assistantTurns.get(message.generation_id)?.setState("cancelled");
    vadStatus.textContent = "ready";
    playbackStatus.textContent = "cancelled";
  }
  if (message.type === "error") {
    vadStatus.textContent = "ready";
    setConnection("error", "Server error", message.message);
  }
}

async function stopSession() {
  stopRequested = true;
  setConnection("connected", "Stopping…", "Closing the microphone and server connection.");
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "session.stop" }));
  if (socket) {
    intentionallyClosedSockets.add(socket);
    socket.close();
  }
  await stopMedia();
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
}
function setConnection(state, text, guidance) { connectionStatus.dataset.state = state; connectionStatus.textContent = text; sessionGuidance.dataset.state = state; sessionGuidance.textContent = guidance; }
function logEvent(message) { const item = document.createElement("li"); item.textContent = `${new Date().toLocaleTimeString()} ${message.type}`; eventLog.prepend(item); }

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
