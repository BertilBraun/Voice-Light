const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;
const endpointInput = document.querySelector("#endpoint-url");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const connectionStatus = document.querySelector("#connection-status");
const sessionGuidance = document.querySelector("#session-guidance");
const vadStatus = document.querySelector("#vad-status");
const playbackStatus = document.querySelector("#playback-status");
const userTranscript = document.querySelector("#user-transcript");
const assistantTranscript = document.querySelector("#assistant-transcript");
const eventLog = document.querySelector("#event-log");

let socket;
let microphoneStream;
let captureContext;
let playbackContext;
let playbackNode;
let stopRequested = false;
let displayedGenerationId;
const intentionallyClosedSockets = new WeakSet();

endpointInput.value = localStorage.getItem("voiceAgentEndpoint") ?? "";
startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);

async function startSession() {
  const endpoint = endpointInput.value.trim();
  if (!endpoint.startsWith("wss://") && !endpoint.startsWith("ws://")) {
    setConnection("error", "Invalid endpoint", "Enter a WebSocket URL beginning with ws:// or wss://.");
    return;
  }
  localStorage.setItem("voiceAgentEndpoint", endpoint);
  stopRequested = false;
  startButton.disabled = true;
  startButton.textContent = "Starting…";
  stopButton.disabled = false;
  setConnection("starting", "Server starting…", "Waking the server. This can take about a minute after it has scaled down.");
  try {
    await setupPlayback();
    socket = await openSocket(endpoint);
    setConnection("connected", "Preparing session…", "The server is connected, but the microphone is not ready yet.");
    const sessionReady = waitForSessionReady(socket);
    socket.send(JSON.stringify({ type: "session.start", input_sample_rate: INPUT_SAMPLE_RATE }));
    await sessionReady;
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

async function setupPlayback() {
  playbackContext = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE });
  await playbackContext.audioWorklet.addModule("/pages/voice-agent/playback-worklet.js");
  playbackNode = new AudioWorkletNode(playbackContext, "pcm-playback", { outputChannelCount: [1] });
  playbackNode.connect(playbackContext.destination);
  await playbackContext.resume();
}

function handleMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    if (playbackContext?.state === "suspended") void playbackContext.resume();
    const view = new DataView(event.data);
    const generationId = view.getUint32(0, true);
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
  if (message.type === "transcript.partial" || message.type === "transcript.final") setTranscript(userTranscript, message.text);
  if (message.type === "assistant.text.delta") {
    if (displayedGenerationId !== message.generation_id) {
      displayedGenerationId = message.generation_id;
      setTranscript(assistantTranscript, "");
    }
    appendTranscript(assistantTranscript, message.text);
    playbackStatus.textContent = "generating";
  }
  if (message.type === "assistant.audio.start") {
    if (playbackContext?.state === "suspended") void playbackContext.resume();
    playbackStatus.textContent = "speaking";
  }
  if (message.type === "assistant.audio.end") {
    vadStatus.textContent = "ready";
    playbackStatus.textContent = "waiting";
  }
  if (message.type === "assistant.cancel") {
    playbackNode.port.postMessage({ type: "clear", generationId: message.generation_id });
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

function resetControls() { displayedGenerationId = undefined; startButton.disabled = false; startButton.textContent = "Start microphone"; stopButton.disabled = true; vadStatus.textContent = "waiting"; }
function setConnection(state, text, guidance) { connectionStatus.dataset.state = state; connectionStatus.textContent = text; sessionGuidance.dataset.state = state; sessionGuidance.textContent = guidance; }
function setTranscript(element, text) { element.textContent = text; element.classList.remove("placeholder"); }
function appendTranscript(element, text) { if (element.classList.contains("placeholder")) setTranscript(element, text); else element.textContent += text; }
function logEvent(message) { const item = document.createElement("li"); item.textContent = `${new Date().toLocaleTimeString()} ${message.type}`; eventLog.prepend(item); }
