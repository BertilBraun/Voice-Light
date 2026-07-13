const INPUT_SAMPLE_RATE = 16000;
const OUTPUT_SAMPLE_RATE = 24000;
const endpointInput = document.querySelector("#endpoint-url");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const connectionStatus = document.querySelector("#connection-status");
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

endpointInput.value = localStorage.getItem("voiceAgentEndpoint") ?? "";
startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);

async function startSession() {
  const endpoint = endpointInput.value.trim();
  if (!endpoint.startsWith("wss://") && !endpoint.startsWith("ws://")) {
    setConnection("error", "Enter a WebSocket URL");
    return;
  }
  localStorage.setItem("voiceAgentEndpoint", endpoint);
  startButton.disabled = true;
  try {
    await setupPlayback();
    microphoneStream = await navigator.mediaDevices.getUserMedia({
      audio: { autoGainControl: true, channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    socket = await openSocket(endpoint);
    socket.send(JSON.stringify({ type: "session.start", input_sample_rate: INPUT_SAMPLE_RATE }));
    await setupCapture(microphoneStream);
    stopButton.disabled = false;
  } catch (error) {
    setConnection("error", error.message);
    await stopSession();
  }
}

function openSocket(endpoint) {
  return new Promise((resolve, reject) => {
    const candidate = new WebSocket(endpoint);
    candidate.binaryType = "arraybuffer";
    candidate.addEventListener("open", () => { socket = candidate; setConnection("connected", "Connected"); resolve(candidate); }, { once: true });
    candidate.addEventListener("error", () => reject(new Error("WebSocket connection failed.")), { once: true });
    candidate.addEventListener("message", handleMessage);
    candidate.addEventListener("close", () => { setConnection("idle", "Disconnected"); stopMedia(); });
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
  if (message.type === "vad.stopped") vadStatus.textContent = "thinking";
  if (message.type === "transcript.partial" || message.type === "transcript.final") setTranscript(userTranscript, message.text);
  if (message.type === "assistant.text.delta") appendTranscript(assistantTranscript, message.text);
  if (message.type === "assistant.audio.start") {
    if (playbackContext?.state === "suspended") void playbackContext.resume();
    playbackStatus.textContent = "speaking";
  }
  if (message.type === "assistant.audio.end") playbackStatus.textContent = "waiting";
  if (message.type === "assistant.cancel") {
    playbackNode.port.postMessage({ type: "clear", generationId: message.generation_id });
    playbackStatus.textContent = "cancelled";
  }
  if (message.type === "error") setConnection("error", message.message);
}

async function stopSession() {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "session.stop" }));
  socket?.close();
  await stopMedia();
  startButton.disabled = false;
  stopButton.disabled = true;
}

async function stopMedia() {
  microphoneStream?.getTracks().forEach((track) => track.stop());
  microphoneStream = undefined;
  if (captureContext && captureContext.state !== "closed") await captureContext.close();
  if (playbackContext && playbackContext.state !== "closed") await playbackContext.close();
  captureContext = undefined;
  playbackContext = undefined;
}

function setConnection(state, text) { connectionStatus.dataset.state = state; connectionStatus.textContent = text; }
function setTranscript(element, text) { element.textContent = text; element.classList.remove("placeholder"); }
function appendTranscript(element, text) { if (element.classList.contains("placeholder")) setTranscript(element, text); else element.textContent += text; }
function logEvent(message) { const item = document.createElement("li"); item.textContent = `${new Date().toLocaleTimeString()} ${message.type}`; eventLog.prepend(item); }
