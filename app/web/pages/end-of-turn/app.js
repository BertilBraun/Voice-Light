const sessionSelect = document.querySelector("#session-select");
const loadingIndicator = document.querySelector("#loading-indicator");
const playToggleButton = document.querySelector("#play-toggle-button");
const speaker2ToggleButton = document.querySelector("#speaker2-toggle-button");
const settingsButton = document.querySelector("#settings-button");
const detectorSettingsModal = document.querySelector("#detector-settings-modal");
const closeSettingsButton = document.querySelector("#close-settings-button");
const cancelSettingsButton = document.querySelector("#cancel-settings-button");
const saveSettingsButton = document.querySelector("#save-settings-button");
const selectAllDetectorsButton = document.querySelector("#select-all-detectors-button");
const clearDetectorsButton = document.querySelector("#clear-detectors-button");
const timeSlider = document.querySelector("#time-slider");
const timeReadout = document.querySelector("#time-readout");
const sessionSummary = document.querySelector("#session-summary");
const detectorSettingsSummary = document.querySelector("#detector-settings-summary");
const detectorOptions = document.querySelector("#detector-options");
const transcriptSummary = document.querySelector("#transcript-summary");
const transcriptList = document.querySelector("#transcript-list");
const canvas = document.querySelector("#timeline-canvas");
const context = canvas.getContext("2d");
const speaker1Audio = document.querySelector("#speaker1-audio");
const speaker2Audio = document.querySelector("#speaker2-audio");
const ANALYSIS_CACHE_SIZE = 20;
const CLICK_DRAG_TOLERANCE_PIXELS = 4;
const DETECTOR_SELECTION_STORAGE_KEY = "voice-light-end-of-turn-detectors";
const AUDIO_SOURCE_VERSION = "pcm16-v1";

let sessions = [];
let detectors = [];
let selectedDetectorModes = new Set();
let stagedDetectorModes = new Set();
let currentPayload = null;
let animationFrameIdentifier = null;
let viewportStartSeconds = 0;
let viewportEndSeconds = 0;
let showSpeaker2 = true;
let dragStartSeconds = null;
let dragCurrentSeconds = null;
let dragStartClientX = null;
let dragStartClientY = null;
let isZoomDragActive = false;
let isPlaying = false;
let analysisRequestId = 0;
let activeTranscriptTurnIndex = null;
let desiredPlaybackSeconds = 0;
let pendingPlaybackSeekSeconds = null;
const analysisPayloadCache = new Map();

async function loadInitialOptions() {
  const [sessionsResponse, detectorsResponse] = await Promise.all([
    fetch("/api/sessions"),
    fetch("/api/end-of-turn/detectors"),
  ]);
  const sessionsPayload = await sessionsResponse.json();
  const detectorsPayload = await detectorsResponse.json();
  sessions = sessionsPayload.sessions;
  detectors = detectorsPayload.detectors;
  selectedDetectorModes = loadSelectedDetectorModes();
  stagedDetectorModes = new Set(selectedDetectorModes);
  sessionSelect.replaceChildren(
    ...sessions.map((session) => {
      const option = document.createElement("option");
      option.value = session.identifier;
      option.textContent = `${session.identifier} - ${formatDuration(session.duration_seconds)} - ${session.topic}`;
      return option;
    }),
  );
  updateDetectorSettingsSummary();
  setDetectorSettingsButtonsEnabled(true);
  if (sessions.length > 0) {
    await analyzeSelectedSession();
  }
}

async function analyzeSelectedSession() {
  const identifier = sessionSelect.value;
  if (!identifier) {
    return;
  }

  const requestId = analysisRequestId + 1;
  analysisRequestId = requestId;
  const analysisCacheKey = buildAnalysisCacheKey(identifier);
  const cachedPayload = getCachedAnalysisPayload(analysisCacheKey);
  if (cachedPayload !== null) {
    pauseInSync();
    applyAnalysisPayload(identifier, cachedPayload);
    return;
  }

  setLoading(true);
  pauseInSync();
  try {
    const query = new URLSearchParams({
      id: identifier,
      detectors: selectedDetectorModesQueryValue(),
    });
    const response = await fetch(`/api/end-of-turn/analyze?${query.toString()}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`Analysis failed with HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (requestId !== analysisRequestId) {
      return;
    }

    putCachedAnalysisPayload(analysisCacheKey, payload);
    applyAnalysisPayload(identifier, payload);
  } catch (error) {
    if (requestId === analysisRequestId) {
      sessionSummary.textContent = error.message;
      transcriptSummary.textContent = "Transcript unavailable.";
    }
  } finally {
    if (requestId === analysisRequestId) {
      setLoading(false);
    }
  }
}

function applyAnalysisPayload(identifier, payload) {
  currentPayload = payload;
  setAudioSourceIfChanged(speaker1Audio, currentPayload.speaker1_audio_url);
  setAudioSourceIfChanged(speaker2Audio, currentPayload.speaker2_audio_url);

  const durationSeconds = currentPayload.analysis.speaker1_waveform.duration_seconds;
  viewportStartSeconds = 0;
  viewportEndSeconds = durationSeconds;
  setSliderBounds();
  timeSlider.value = "0";
  desiredPlaybackSeconds = 0;
  timeSlider.disabled = false;
  playToggleButton.disabled = false;
  speaker2ToggleButton.disabled = false;
  sessionSummary.textContent = `${identifier} analyzed with ${currentPayload.analysis.baseline_results.length} selected detectors.`;
  renderTranscript();
  updatePlayToggleLabel();
  drawTimeline();
}

function renderDetectorOptions() {
  detectorOptions.replaceChildren(
    ...detectors.map((detector) => {
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.id = `detector-${detector.mode}`;
      checkbox.value = detector.mode;
      checkbox.checked = stagedDetectorModes.has(detector.mode);
      checkbox.addEventListener("change", updateStagedDetectorModesFromInputs);

      const title = document.createElement("strong");
      title.textContent = detector.label;
      const description = document.createElement("span");
      description.textContent = detector.description;

      const text = document.createElement("div");
      text.append(title, description);

      const option = document.createElement("label");
      option.className = "detector-option";
      option.htmlFor = checkbox.id;
      option.append(checkbox, text);
      return option;
    }),
  );
  updateDetectorSettingsSummary();
  updateSaveSettingsButton();
}

function loadSelectedDetectorModes() {
  const storedValue = window.localStorage.getItem(DETECTOR_SELECTION_STORAGE_KEY);
  if (storedValue === null) {
    return new Set(detectors.map((detector) => detector.mode));
  }

  const storedModes = JSON.parse(storedValue);
  if (!Array.isArray(storedModes)) {
    throw new Error("Stored detector selection must be a list.");
  }

  const availableModes = new Set(detectors.map((detector) => detector.mode));
  return new Set(storedModes.filter((mode) => availableModes.has(mode)));
}

function openSettingsModal() {
  stagedDetectorModes = new Set(selectedDetectorModes);
  renderDetectorOptions();
  detectorSettingsModal.hidden = false;
}

function closeSettingsModal() {
  detectorSettingsModal.hidden = true;
  stagedDetectorModes = new Set(selectedDetectorModes);
  updateDetectorSettingsSummary();
}

function updateStagedDetectorModesFromInputs() {
  stagedDetectorModes = new Set(
    [...detectorOptions.querySelectorAll("input[type='checkbox']")]
      .filter((checkbox) => checkbox.checked)
      .map((checkbox) => checkbox.value),
  );
  updateDetectorSettingsSummary();
  updateSaveSettingsButton();
}

function selectAllDetectors() {
  stagedDetectorModes = new Set(detectors.map((detector) => detector.mode));
  renderDetectorOptions();
}

function clearDetectors() {
  stagedDetectorModes = new Set();
  renderDetectorOptions();
}

function saveDetectorSettings() {
  if (detectorModeSetsEqual(selectedDetectorModes, stagedDetectorModes)) {
    closeSettingsModal();
    return;
  }
  selectedDetectorModes = new Set(stagedDetectorModes);
  persistSelectedDetectorModes();
  updateDetectorSettingsSummary();
  closeSettingsModal();
  void analyzeSelectedSession();
}

function persistSelectedDetectorModes() {
  window.localStorage.setItem(
    DETECTOR_SELECTION_STORAGE_KEY,
    JSON.stringify([...selectedDetectorModes]),
  );
}

function updateDetectorSettingsSummary() {
  const displayedDetectorModes = detectorSettingsModal.hidden
    ? selectedDetectorModes
    : stagedDetectorModes;
  detectorSettingsSummary.textContent = `${displayedDetectorModes.size}/${detectors.length} detectors selected`;
}

function updateSaveSettingsButton() {
  saveSettingsButton.disabled = detectorModeSetsEqual(selectedDetectorModes, stagedDetectorModes);
}

function setDetectorSettingsButtonsEnabled(areEnabled) {
  settingsButton.disabled = !areEnabled;
  selectAllDetectorsButton.disabled = !areEnabled;
  clearDetectorsButton.disabled = !areEnabled;
}

function detectorModeSetsEqual(firstDetectorModes, secondDetectorModes) {
  if (firstDetectorModes.size !== secondDetectorModes.size) {
    return false;
  }
  return [...firstDetectorModes].every((detectorMode) => secondDetectorModes.has(detectorMode));
}

function selectedDetectorModesQueryValue() {
  return detectors
    .filter((detector) => selectedDetectorModes.has(detector.mode))
    .map((detector) => detector.mode)
    .join(",");
}

function setAudioSourceIfChanged(audioElement, sourceUrl) {
  const currentUrl = new URL(audioElement.currentSrc || audioElement.src || "", window.location.href);
  const nextUrl = new URL(sourceUrl, window.location.href);
  nextUrl.searchParams.set("v", AUDIO_SOURCE_VERSION);
  if (currentUrl.href === nextUrl.href) {
    return;
  }
  audioElement.src = nextUrl.href;
  audioElement.load();
}

function buildAnalysisCacheKey(identifier) {
  return `${identifier}:${selectedDetectorModesQueryValue()}`;
}

function getCachedAnalysisPayload(analysisCacheKey) {
  const payload = analysisPayloadCache.get(analysisCacheKey);
  if (payload === undefined) {
    return null;
  }
  analysisPayloadCache.delete(analysisCacheKey);
  analysisPayloadCache.set(analysisCacheKey, payload);
  return payload;
}

function putCachedAnalysisPayload(analysisCacheKey, payload) {
  analysisPayloadCache.delete(analysisCacheKey);
  analysisPayloadCache.set(analysisCacheKey, payload);
  while (analysisPayloadCache.size > ANALYSIS_CACHE_SIZE) {
    const oldestAnalysisCacheKey = analysisPayloadCache.keys().next().value;
    analysisPayloadCache.delete(oldestAnalysisCacheKey);
  }
}

function setLoading(isLoading) {
  sessionSelect.disabled = isLoading;
  loadingIndicator.hidden = !isLoading;
  if (isLoading) {
    sessionSummary.textContent = `Analyzing ${sessionSelect.value}...`;
    transcriptSummary.textContent = "Loading transcript...";
    transcriptList.replaceChildren();
  }
}

function setSliderBounds() {
  timeSlider.min = String(viewportStartSeconds);
  timeSlider.max = String(viewportEndSeconds);
  timeSlider.step = "0.01";
}

function drawTimeline() {
  updateCanvasHeight();
  const layout = resizeCanvas();
  context.clearRect(0, 0, layout.width, layout.height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, layout.width, layout.height);

  if (currentPayload === null) {
    drawEmptyState();
    return;
  }

  const analysis = currentPayload.analysis;
  const durationSeconds = analysis.speaker1_waveform.duration_seconds;
  const rows = [
    { label: "Speaker 1", waveform: analysis.speaker1_waveform, color: "#146b63" },
  ];
  if (showSpeaker2) {
    rows.push({ label: "Speaker 2", waveform: analysis.speaker2_waveform, color: "#8d4d1f" });
  }

  const leftPad = 92;
  const rightPad = 16;
  const trackWidth = layout.width - leftPad - rightPad;
  const waveTop = 44;
  const waveHeight = showSpeaker2 ? 120 : 172;
  const rowGap = 24;

  drawTimeRuler(leftPad, trackWidth, layout.height);
  rows.forEach((row, index) => {
    const top = waveTop + index * (waveHeight + rowGap);
    drawWaveform(row, leftPad, top, trackWidth, waveHeight);
  });

  let baselineTop = waveTop + rows.length * (waveHeight + rowGap) + 16;
  analysis.baseline_results.forEach((baselineResult) => {
    drawBaselineRow(baselineResult, leftPad, baselineTop, trackWidth);
    baselineTop += 74;
  });

  drawZoomDrag(leftPad, trackWidth, layout.height);
  drawPlayhead(leftPad, trackWidth, layout.height, durationSeconds);
  updateTimeReadout();
  updateActiveTranscriptTurn();
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * scale);
  canvas.height = Math.floor(rect.height * scale);
  context.setTransform(scale, 0, 0, scale, 0, 0);
  return { width: rect.width, height: rect.height };
}

function updateCanvasHeight() {
  if (currentPayload === null) {
    canvas.style.height = "560px";
    return;
  }
  const baselineCount = currentPayload.analysis.baseline_results.length;
  const rowCount = showSpeaker2 ? 2 : 1;
  const waveHeight = showSpeaker2 ? 120 : 172;
  const requiredHeight = 44 + rowCount * (waveHeight + 24) + 16 + baselineCount * 74 + 24;
  canvas.style.height = `${Math.max(560, requiredHeight)}px`;
}

function drawEmptyState() {
  context.fillStyle = "#5a666b";
  context.font = "16px sans-serif";
  context.fillText("Choose a session to analyze.", 24, 42);
}

function drawTimeRuler(leftPad, trackWidth, layoutHeight) {
  context.strokeStyle = "#d3dadd";
  context.fillStyle = "#5a666b";
  context.font = "12px sans-serif";
  for (let tick = 0; tick <= 6; tick += 1) {
    const seconds = viewportStartSeconds + (visibleDurationSeconds() * tick) / 6;
    const x = leftPad + (trackWidth * tick) / 6;
    context.beginPath();
    context.moveTo(x, 20);
    context.lineTo(x, layoutHeight - 10);
    context.stroke();
    context.fillText(formatDuration(seconds), x - 18, 16);
  }
}

function drawWaveform(row, leftPad, top, trackWidth, height) {
  context.fillStyle = "#1e2528";
  context.font = "13px sans-serif";
  context.fillText(row.label, 0, top + height / 2 + 4);
  context.strokeStyle = "#edf0f1";
  context.strokeRect(leftPad, top, trackWidth, height);
  context.strokeStyle = row.color;
  context.lineWidth = 1;

  const middle = top + height / 2;
  const minimums = row.waveform.minimums;
  const maximums = row.waveform.maximums;
  const binDurationSeconds = row.waveform.duration_seconds / Math.max(1, maximums.length);
  const columns = Math.max(1, Math.floor(trackWidth));

  for (let column = 0; column < columns; column += 1) {
    const startSeconds = viewportStartSeconds + (column / columns) * visibleDurationSeconds();
    const endSeconds =
      viewportStartSeconds + ((column + 1) / columns) * visibleDurationSeconds();
    const startIndex = clamp(
      Math.floor(startSeconds / binDurationSeconds),
      0,
      maximums.length - 1,
    );
    const endIndex = clamp(Math.ceil(endSeconds / binDurationSeconds), startIndex + 1, maximums.length);
    const columnMinimum = Math.min(...minimums.slice(startIndex, endIndex));
    const columnMaximum = Math.max(...maximums.slice(startIndex, endIndex));
    const x = leftPad + column;
    const yMin = middle + columnMinimum * (height / 2 - 4);
    const yMax = middle + columnMaximum * (height / 2 - 4);
    context.beginPath();
    context.moveTo(x, yMin);
    context.lineTo(x, yMax);
    context.stroke();
  }
}

function drawBaselineRow(baselineResult, leftPad, top, trackWidth) {
  context.fillStyle = "#1e2528";
  context.font = "13px sans-serif";
  context.fillText(baselineResult.name, 0, top + 26);
  context.fillStyle = "#f3f5f6";
  context.fillRect(leftPad, top, trackWidth, 42);

  drawBaselineSpans(
    baselineResult.speech_segments,
    leftPad,
    top + 8,
    trackWidth,
    26,
    "rgba(20, 107, 99, 0.18)",
    "rgba(20, 107, 99, 0.35)",
  );
  drawBaselineSpans(
    baselineResult.pause_spans,
    leftPad,
    top + 11,
    trackWidth,
    20,
    "rgba(224, 173, 42, 0.5)",
    "rgba(156, 110, 0, 0.7)",
  );
  drawBaselineSpans(
    baselineResult.backchannel_spans,
    leftPad,
    top + 6,
    trackWidth,
    30,
    "rgba(126, 87, 194, 0.5)",
    "rgba(90, 54, 153, 0.75)",
  );

  context.strokeStyle = "#c2322d";
  context.fillStyle = "#c2322d";
  baselineResult.end_of_turn_events.forEach((event) => {
    if (event.time_seconds < viewportStartSeconds || event.time_seconds > viewportEndSeconds) {
      return;
    }
    const x = leftPad + secondsToRatio(event.time_seconds) * trackWidth;
    context.beginPath();
    context.moveTo(x, top - 2);
    context.lineTo(x, top + 48);
    context.stroke();
  });
  context.font = "12px sans-serif";
  context.fillText(
    `EOT ${countVisibleEndMarkers(baselineResult)}/${baselineResult.end_of_turn_events.length} | pauses ${baselineResult.pause_spans.length} | backchannels ${baselineResult.backchannel_spans.length}`,
    leftPad,
    top + 62,
  );
}

function drawBaselineSpans(spans, leftPad, top, trackWidth, height, fillStyle, strokeStyle) {
  context.fillStyle = fillStyle;
  context.strokeStyle = strokeStyle;
  spans.forEach((segment) => {
    if (!timeRangesOverlap(segment.start_seconds, segment.end_seconds)) {
      return;
    }
    const startSeconds = Math.max(segment.start_seconds, viewportStartSeconds);
    const endSeconds = Math.min(segment.end_seconds, viewportEndSeconds);
    const x = leftPad + secondsToRatio(startSeconds) * trackWidth;
    const width = ((endSeconds - startSeconds) / visibleDurationSeconds()) * trackWidth;
    const visibleWidth = Math.max(1, width);
    context.fillRect(x, top, visibleWidth, height);
    context.strokeRect(x, top, visibleWidth, height);
  });
}

function drawPlayhead(leftPad, trackWidth, layoutHeight, durationSeconds) {
  const currentTime = desiredPlaybackSeconds;
  if (currentTime < viewportStartSeconds || currentTime > viewportEndSeconds || durationSeconds === 0) {
    return;
  }
  const x = leftPad + secondsToRatio(currentTime) * trackWidth;
  context.strokeStyle = "#11181b";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(x, 20);
  context.lineTo(x, layoutHeight - 10);
  context.stroke();
  context.lineWidth = 1;
}

function drawZoomDrag(leftPad, trackWidth, layoutHeight) {
  if (dragStartSeconds === null || dragCurrentSeconds === null) {
    return;
  }
  const startSeconds = Math.min(dragStartSeconds, dragCurrentSeconds);
  const endSeconds = Math.max(dragStartSeconds, dragCurrentSeconds);
  const x = leftPad + secondsToRatio(startSeconds) * trackWidth;
  const width = ((endSeconds - startSeconds) / visibleDurationSeconds()) * trackWidth;
  context.fillStyle = "rgba(35, 91, 176, 0.18)";
  context.fillRect(x, 22, width, layoutHeight - 34);
  context.strokeStyle = "#235bb0";
  context.strokeRect(x, 22, width, layoutHeight - 34);
}

function togglePlayback() {
  if (isPlaying) {
    pauseInSync();
    return;
  }
  playInSync();
}

function togglePlaybackWithKeyboard(event) {
  if (event.key !== " " || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
    return;
  }
  if (
    event.target instanceof HTMLElement &&
    (event.target.matches("button,select,textarea") ||
      (event.target.matches("input") && event.target !== timeSlider))
  ) {
    return;
  }
  if (currentPayload === null || playToggleButton.disabled) {
    return;
  }
  event.preventDefault();
  togglePlayback();
}

function playInSync() {
  if (currentPayload === null) {
    return;
  }
  desiredPlaybackSeconds = Number(timeSlider.value);
  syncAudioToPlaybackTarget();
  if (
    speaker1Audio.currentTime < viewportStartSeconds ||
    speaker1Audio.currentTime > viewportEndSeconds
  ) {
    speaker1Audio.currentTime = viewportStartSeconds;
    desiredPlaybackSeconds = viewportStartSeconds;
    timeSlider.value = String(desiredPlaybackSeconds);
  }
  const speaker1PlayPromise = speaker1Audio.play();
  void speaker1PlayPromise.then(syncAudioToPlaybackTarget);
  if (showSpeaker2) {
    speaker2Audio.currentTime = speaker1Audio.currentTime;
    void speaker2Audio.play();
  } else {
    speaker2Audio.pause();
  }
  isPlaying = true;
  updatePlayToggleLabel();
  startPlaybackLoop();
}

function pauseInSync() {
  speaker1Audio.pause();
  speaker2Audio.pause();
  isPlaying = false;
  updatePlayToggleLabel();
  stopPlaybackLoop();
  drawTimeline();
}

function startPlaybackLoop() {
  stopPlaybackLoop();
  const drawFrame = () => {
    if (pendingPlaybackSeekSeconds !== null) {
      if (Math.abs(speaker1Audio.currentTime - pendingPlaybackSeekSeconds) > 0.08) {
        speaker1Audio.currentTime = pendingPlaybackSeekSeconds;
        if (showSpeaker2) {
          speaker2Audio.currentTime = pendingPlaybackSeekSeconds;
        }
        desiredPlaybackSeconds = pendingPlaybackSeekSeconds;
        timeSlider.value = String(desiredPlaybackSeconds);
        drawTimeline();
        animationFrameIdentifier = window.requestAnimationFrame(drawFrame);
        return;
      }
      pendingPlaybackSeekSeconds = null;
    }
    if (
      showSpeaker2 &&
      Math.abs(speaker2Audio.currentTime - speaker1Audio.currentTime) > 0.08
    ) {
      speaker2Audio.currentTime = speaker1Audio.currentTime;
    }
    desiredPlaybackSeconds = speaker1Audio.currentTime;
    timeSlider.value = String(desiredPlaybackSeconds);
    drawTimeline();
    animationFrameIdentifier = window.requestAnimationFrame(drawFrame);
  };
  animationFrameIdentifier = window.requestAnimationFrame(drawFrame);
}

function stopPlaybackLoop() {
  if (animationFrameIdentifier !== null) {
    window.cancelAnimationFrame(animationFrameIdentifier);
    animationFrameIdentifier = null;
  }
}

function seekBothAudio() {
  const targetTime = Number(timeSlider.value);
  seekToSeconds(targetTime);
}

function seekToSeconds(targetTime) {
  const durationSeconds = currentPayload.analysis.speaker1_waveform.duration_seconds;
  const boundedTargetTime = clamp(targetTime, 0, durationSeconds);
  desiredPlaybackSeconds = boundedTargetTime;
  timeSlider.value = String(desiredPlaybackSeconds);
  syncAudioToPlaybackTarget();
  drawTimeline();
}

function syncAudioToPlaybackTarget() {
  pendingPlaybackSeekSeconds = desiredPlaybackSeconds;
  speaker1Audio.currentTime = desiredPlaybackSeconds;
  if (showSpeaker2) {
    speaker2Audio.currentTime = desiredPlaybackSeconds;
  }
}

function renderTranscript() {
  activeTranscriptTurnIndex = null;
  const transcriptTurns = transcriptTurnsForCurrentPayload();
  transcriptSummary.textContent = `${transcriptTurns.length} transcript turns`;
  if (transcriptTurns.length === 0) {
    const emptyState = document.createElement("div");
    emptyState.className = "transcript-empty";
    emptyState.textContent = "No transcript turns are available for this session.";
    transcriptList.replaceChildren(emptyState);
    return;
  }

  transcriptList.replaceChildren(
    ...transcriptTurns.map((transcriptTurn, transcriptTurnIndex) => {
      const meta = document.createElement("div");
      meta.className = "transcript-meta";
      meta.textContent = `${transcriptTurn.speaker} ${formatDuration(transcriptTurn.start_seconds)}`;

      const text = document.createElement("div");
      text.className = "transcript-text";
      text.textContent = transcriptTurn.text;

      const turn = document.createElement("div");
      turn.className = `transcript-turn ${transcriptTurnClassName(transcriptTurn.speaker)}`;
      turn.dataset.transcriptTurnIndex = String(transcriptTurnIndex);
      turn.append(meta, text);
      return turn;
    }),
  );
}

function transcriptTurnClassName(speaker) {
  if (speaker === "Speaker1") {
    return "transcript-turn-speaker1";
  }
  if (speaker === "Speaker2") {
    return "transcript-turn-speaker2";
  }
  return "";
}

function updateActiveTranscriptTurn() {
  if (currentPayload === null) {
    return;
  }
  const currentTime = desiredPlaybackSeconds;
  const transcriptTurnIndex = transcriptTurnsForCurrentPayload().findIndex(
    (transcriptTurn) =>
      currentTime >= transcriptTurn.start_seconds && currentTime <= transcriptTurn.end_seconds,
  );
  if (transcriptTurnIndex === activeTranscriptTurnIndex) {
    return;
  }

  const previousTurn = transcriptList.querySelector(".transcript-turn-active");
  if (previousTurn !== null) {
    previousTurn.classList.remove("transcript-turn-active");
  }
  activeTranscriptTurnIndex = transcriptTurnIndex === -1 ? null : transcriptTurnIndex;
  if (activeTranscriptTurnIndex === null) {
    return;
  }

  const activeTurn = transcriptList.querySelector(
    `[data-transcript-turn-index='${activeTranscriptTurnIndex}']`,
  );
  if (activeTurn === null) {
    return;
  }
  activeTurn.classList.add("transcript-turn-active");
  centerTranscriptTurn(activeTurn);
}

function centerTranscriptTurn(transcriptTurnElement) {
  const targetScrollTop =
    transcriptTurnElement.offsetTop -
    transcriptList.clientHeight / 2 +
    transcriptTurnElement.offsetHeight / 2;
  const maximumScrollTop = transcriptList.scrollHeight - transcriptList.clientHeight;
  transcriptList.scrollTop = clamp(targetScrollTop, 0, Math.max(0, maximumScrollTop));
}

function transcriptTurnsForCurrentPayload() {
  if (currentPayload === null || !Array.isArray(currentPayload.analysis.transcript_turns)) {
    return [];
  }
  return currentPayload.analysis.transcript_turns;
}

function toggleSpeaker2() {
  showSpeaker2 = !showSpeaker2;
  speaker2ToggleButton.textContent = showSpeaker2 ? "Hide speaker 2" : "Show speaker 2";
  if (showSpeaker2) {
    speaker2Audio.currentTime = speaker1Audio.currentTime;
    if (isPlaying) {
      void speaker2Audio.play();
    }
  } else {
    speaker2Audio.pause();
  }
  drawTimeline();
}

function beginZoomDrag(event) {
  if (currentPayload === null) {
    return;
  }
  const seconds = eventToSeconds(event);
  if (seconds === null) {
    return;
  }
  event.preventDefault();
  dragStartSeconds = seconds;
  dragCurrentSeconds = null;
  dragStartClientX = event.clientX;
  dragStartClientY = event.clientY;
  isZoomDragActive = false;
  canvas.setPointerCapture(event.pointerId);
}

function updateZoomDrag(event) {
  if (dragStartSeconds === null || dragStartClientX === null || dragStartClientY === null) {
    return;
  }
  const seconds = eventToSeconds(event);
  if (seconds === null) {
    return;
  }
  const pointerDistancePixels = Math.hypot(
    event.clientX - dragStartClientX,
    event.clientY - dragStartClientY,
  );
  if (!isZoomDragActive && pointerDistancePixels < CLICK_DRAG_TOLERANCE_PIXELS) {
    return;
  }
  isZoomDragActive = true;
  dragCurrentSeconds = seconds;
  drawTimeline();
}

function endZoomDrag(event) {
  if (dragStartSeconds === null) {
    return;
  }
  canvas.releasePointerCapture(event.pointerId);
  const targetSeconds = eventToSeconds(event) ?? dragStartSeconds;
  if (!isZoomDragActive || dragCurrentSeconds === null) {
    clearTimelineDrag();
    seekToSeconds(targetSeconds);
    return;
  }
  const startSeconds = Math.min(dragStartSeconds, dragCurrentSeconds);
  const endSeconds = Math.max(dragStartSeconds, dragCurrentSeconds);
  clearTimelineDrag();
  if (endSeconds - startSeconds >= 0.5) {
    setViewport(startSeconds, endSeconds);
  }
  drawTimeline();
}

function cancelZoomDrag(event) {
  if (dragStartSeconds === null) {
    return;
  }
  canvas.releasePointerCapture(event.pointerId);
  clearTimelineDrag();
  drawTimeline();
}

function clearTimelineDrag() {
  dragStartSeconds = null;
  dragCurrentSeconds = null;
  dragStartClientX = null;
  dragStartClientY = null;
  isZoomDragActive = false;
}

function zoomAtPointer(event) {
  if (currentPayload === null) {
    return;
  }
  if (!eventIsOverWaveform(event)) {
    return;
  }
  const anchorSeconds = eventToSeconds(event);
  if (anchorSeconds === null) {
    return;
  }
  event.preventDefault();
  const zoomFactor = event.deltaY < 0 ? 0.8 : 1.25;
  const currentDurationSeconds = visibleDurationSeconds();
  const nextDurationSeconds = clamp(
    currentDurationSeconds * zoomFactor,
    0.5,
    currentPayload.analysis.speaker1_waveform.duration_seconds,
  );
  const anchorRatio = (anchorSeconds - viewportStartSeconds) / currentDurationSeconds;
  const nextStartSeconds = anchorSeconds - nextDurationSeconds * anchorRatio;
  const nextEndSeconds = nextStartSeconds + nextDurationSeconds;
  setViewport(nextStartSeconds, nextEndSeconds);
  drawTimeline();
}

function eventIsOverWaveform(event) {
  const rect = canvas.getBoundingClientRect();
  const leftPad = 92;
  const rightPad = 16;
  const trackWidth = rect.width - leftPad - rightPad;
  const x = event.clientX - rect.left;
  if (x < leftPad || x > leftPad + trackWidth) {
    return false;
  }

  const y = event.clientY - rect.top;
  const waveTop = 44;
  const waveHeight = showSpeaker2 ? 120 : 172;
  const rowGap = 24;
  const rowCount = showSpeaker2 ? 2 : 1;
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const top = waveTop + rowIndex * (waveHeight + rowGap);
    if (y >= top && y <= top + waveHeight) {
      return true;
    }
  }
  return false;
}

function eventToSeconds(event) {
  const rect = canvas.getBoundingClientRect();
  const leftPad = 92;
  const rightPad = 16;
  const trackWidth = rect.width - leftPad - rightPad;
  const x = event.clientX - rect.left;
  if (x < leftPad || x > leftPad + trackWidth) {
    return null;
  }
  const ratio = (x - leftPad) / trackWidth;
  return viewportStartSeconds + ratio * visibleDurationSeconds();
}

function updatePlayToggleLabel() {
  playToggleButton.textContent = isPlaying ? "Pause" : "Play";
}

function updateTimeReadout() {
  timeReadout.textContent = `${formatDuration(desiredPlaybackSeconds)} / ${formatDuration(viewportEndSeconds)}`;
}

function setViewport(startSeconds, endSeconds) {
  const durationSeconds = currentPayload.analysis.speaker1_waveform.duration_seconds;
  const viewportDurationSeconds = Math.min(durationSeconds, Math.max(0.5, endSeconds - startSeconds));
  let nextStartSeconds = startSeconds;
  let nextEndSeconds = nextStartSeconds + viewportDurationSeconds;
  if (nextStartSeconds < 0) {
    nextStartSeconds = 0;
    nextEndSeconds = viewportDurationSeconds;
  }
  if (nextEndSeconds > durationSeconds) {
    nextEndSeconds = durationSeconds;
    nextStartSeconds = durationSeconds - viewportDurationSeconds;
  }
  viewportStartSeconds = nextStartSeconds;
  viewportEndSeconds = nextEndSeconds;
  setSliderBounds();
  desiredPlaybackSeconds = clamp(desiredPlaybackSeconds, 0, durationSeconds);
  timeSlider.value = String(desiredPlaybackSeconds);
}

function visibleDurationSeconds() {
  return Math.max(0.01, viewportEndSeconds - viewportStartSeconds);
}

function secondsToRatio(seconds) {
  return (seconds - viewportStartSeconds) / visibleDurationSeconds();
}

function timeRangesOverlap(startSeconds, endSeconds) {
  return startSeconds <= viewportEndSeconds && endSeconds >= viewportStartSeconds;
}

function countVisibleEndMarkers(baselineResult) {
  return baselineResult.end_of_turn_events.filter(
    (event) => event.time_seconds >= viewportStartSeconds && event.time_seconds <= viewportEndSeconds,
  ).length;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function formatDuration(totalSeconds) {
  const boundedSeconds = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(boundedSeconds / 60);
  const seconds = boundedSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

sessionSelect.addEventListener("change", analyzeSelectedSession);
playToggleButton.addEventListener("click", togglePlayback);
document.addEventListener("keydown", togglePlaybackWithKeyboard);
speaker2ToggleButton.addEventListener("click", toggleSpeaker2);
settingsButton.addEventListener("click", openSettingsModal);
closeSettingsButton.addEventListener("click", closeSettingsModal);
cancelSettingsButton.addEventListener("click", closeSettingsModal);
saveSettingsButton.addEventListener("click", saveDetectorSettings);
selectAllDetectorsButton.addEventListener("click", selectAllDetectors);
clearDetectorsButton.addEventListener("click", clearDetectors);
timeSlider.addEventListener("input", seekBothAudio);
canvas.addEventListener("pointerdown", beginZoomDrag);
canvas.addEventListener("pointermove", updateZoomDrag);
canvas.addEventListener("pointerup", endZoomDrag);
canvas.addEventListener("pointercancel", cancelZoomDrag);
canvas.addEventListener("wheel", zoomAtPointer, { passive: false });
speaker1Audio.addEventListener("ended", pauseInSync);
window.addEventListener("resize", drawTimeline);

await loadInitialOptions();
drawTimeline();
