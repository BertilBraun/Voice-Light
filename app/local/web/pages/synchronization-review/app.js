const MAXIMUM_SHIFT_SECONDS = 12;
const DEFAULT_DURATION_SECONDS = 180;
const WAVEFORM_POINT_COUNT = 1800;

const elements = {
  workspace: document.querySelector("#workspace"),
  errorState: document.querySelector("#error-state"),
  coverage: document.querySelector("#coverage"),
  candidateCount: document.querySelector("#candidate-count"),
  candidateList: document.querySelector("#candidate-list"),
  filter: document.querySelector("#filter"),
  position: document.querySelector("#position"),
  sampleName: document.querySelector("#sample-name"),
  sampleSummary: document.querySelector("#sample-summary"),
  previous: document.querySelector("#previous"),
  next: document.querySelector("#next"),
  usePrediction: document.querySelector("#use-prediction"),
  shift: document.querySelector("#shift"),
  shiftNumber: document.querySelector("#shift-number"),
  play: document.querySelector("#play"),
  seek: document.querySelector("#seek"),
  clock: document.querySelector("#clock"),
  playbackStatus: document.querySelector("#playback-status"),
  gainA: document.querySelector("#gain-a"),
  gainB: document.querySelector("#gain-b"),
  gainAValue: document.querySelector("#gain-a-value"),
  gainBValue: document.querySelector("#gain-b-value"),
  useAutoGain: document.querySelector("#use-auto-gain"),
  gainSummary: document.querySelector("#gain-summary"),
  waveform: document.querySelector("#waveform"),
  timelineTicks: document.querySelector("#timeline-ticks"),
  windowTargets: document.querySelector("#window-targets"),
  evidence: document.querySelector("#evidence"),
  audioA: document.querySelector("#audio-a"),
  audioB: document.querySelector("#audio-b"),
};

const state = {
  candidates: [],
  visibleCandidates: [],
  selectedExternalId: null,
  waveformA: null,
  waveformB: null,
  durationSeconds: DEFAULT_DURATION_SECONDS,
  bShiftSeconds: 0,
  timelineSeconds: 0,
  playing: false,
  startingPlayback: false,
  playbackTimelineStart: 0,
  playbackClockStart: 0,
  animationFrame: null,
  selectionVersion: 0,
  audioContext: null,
  gainNodeA: null,
  gainNodeB: null,
  masterLimiter: null,
};

function selectedCandidate() {
  return state.candidates.find(
    (candidate) => candidate.external_id === state.selectedExternalId,
  );
}

function selectedVisibleIndex() {
  return state.visibleCandidates.findIndex(
    (candidate) => candidate.external_id === state.selectedExternalId,
  );
}

function formatShift(seconds) {
  return `${seconds >= 0 ? "+" : "−"}${Math.abs(seconds).toFixed(1)} s`;
}

function formatTime(seconds) {
  const sign = seconds < 0 ? "−" : "";
  const absoluteSeconds = Math.abs(seconds);
  const minutes = Math.floor(absoluteSeconds / 60);
  const remainingSeconds = absoluteSeconds - minutes * 60;
  return `${sign}${minutes}:${remainingSeconds.toFixed(1).padStart(4, "0")}`;
}

function timelineBounds() {
  return {
    start: Math.min(0, state.bShiftSeconds),
    end: Math.max(
      state.durationSeconds,
      state.durationSeconds + state.bShiftSeconds,
    ),
  };
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function setError(message) {
  elements.workspace.hidden = true;
  elements.errorState.hidden = false;
  elements.errorState.textContent = message;
}

async function loadCandidates() {
  try {
    const response = await fetch("/api/synchronization-review/candidates");
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Request failed with ${response.status}`);
    }
    const payload = await response.json();
    state.candidates = payload.candidates;
    state.visibleCandidates = payload.candidates;
    elements.coverage.textContent =
      `${payload.analyzed_session_count} sessions analyzed · ` +
      `${payload.excluded_session_count} without complete timing evidence`;
    elements.workspace.hidden = false;
    elements.candidateCount.textContent = `${payload.candidates.length} candidates`;
    renderCandidateList();
    if (payload.candidates.length === 0) {
      setError("No synchronization candidates passed the current evidence thresholds.");
      return;
    }
    await selectCandidate(payload.candidates[0].external_id);
  } catch (error) {
    setError(`Could not load synchronization candidates: ${error.message}`);
  }
}

function renderCandidateList() {
  elements.candidateList.replaceChildren();
  for (const candidate of state.visibleCandidates) {
    const button = document.createElement("button");
    button.type = "button";
    button.className =
      candidate.external_id === state.selectedExternalId
        ? "candidate selected"
        : "candidate";
    button.dataset.externalId = candidate.external_id;
    button.innerHTML = `
      <span class="candidate-main">
        <strong>${candidate.external_id.toUpperCase()}</strong>
        <span class="score">${Math.round(candidate.likelihood_score * 100)}%</span>
      </span>
      <span class="candidate-main">
        <span class="badge ${candidate.offset_pattern}">${candidate.offset_pattern}</span>
        <strong>${formatShift(candidate.estimated_b_shift_seconds)}</strong>
      </span>
      <span class="candidate-meta">
        <span>${candidate.overlap_silence_cycle_count} overlap/silence cycles</span>
        <span>${candidate.source_agreement ? "sources agree" : "mixed evidence"}</span>
      </span>
    `;
    button.addEventListener("click", () => {
      void selectCandidate(candidate.external_id);
    });
    elements.candidateList.append(button);
  }
}

async function selectCandidate(externalId) {
  const candidate = state.candidates.find((item) => item.external_id === externalId);
  if (!candidate) {
    return;
  }
  pause();
  state.selectedExternalId = externalId;
  state.bShiftSeconds = candidate.estimated_b_shift_seconds;
  state.timelineSeconds = 0;
  applyAutomaticGains(candidate);
  state.waveformA = null;
  state.waveformB = null;
  state.durationSeconds = DEFAULT_DURATION_SECONDS;
  setPlaybackStatus("", false);
  state.selectionVersion += 1;
  const selectionVersion = state.selectionVersion;
  renderCandidateList();
  document
    .querySelector(`.candidate[data-external-id="${externalId}"]`)
    ?.scrollIntoView({ block: "nearest" });
  renderCandidateDetails();
  configureAudio(candidate);
  drawWaveforms();

  try {
    const [waveformA, waveformB] = await Promise.all([
      fetchWaveform(candidate.sample_id, "speaker1"),
      fetchWaveform(candidate.sample_id, "speaker2"),
    ]);
    if (selectionVersion !== state.selectionVersion) {
      return;
    }
    state.waveformA = waveformA;
    state.waveformB = waveformB;
    state.durationSeconds = Math.min(
      DEFAULT_DURATION_SECONDS,
      Math.max(waveformA.duration_seconds, waveformB.duration_seconds),
    );
    updateTimeline();
  } catch (error) {
    if (selectionVersion === state.selectionVersion) {
      setError(`Could not load waveforms for ${candidate.external_id}: ${error.message}`);
    }
  }
}

function configureAudio(candidate) {
  elements.audioA.src =
    `/api/dataset-dashboard/audio/${candidate.sample_id}/speaker1?trimmed=true`;
  elements.audioB.src =
    `/api/dataset-dashboard/audio/${candidate.sample_id}/speaker2?trimmed=true`;
  elements.audioA.load();
  elements.audioB.load();
}

async function fetchWaveform(sampleId, side) {
  const response = await fetch(
    `/api/dataset-dashboard/waveform/${sampleId}/${side}` +
      `?points=${WAVEFORM_POINT_COUNT}&trimmed=true`,
  );
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Waveform request failed with ${response.status}`);
  }
  return response.json();
}

function renderCandidateDetails() {
  const candidate = selectedCandidate();
  if (!candidate) {
    return;
  }
  const visibleIndex = selectedVisibleIndex();
  elements.position.textContent =
    visibleIndex >= 0
      ? `Candidate ${visibleIndex + 1} of ${state.visibleCandidates.length}`
      : "Candidate hidden by filter";
  elements.sampleName.textContent = candidate.external_id.toUpperCase();
  elements.sampleSummary.innerHTML = `
    <span class="badge ${candidate.offset_pattern}">${candidate.offset_pattern} offset</span>
    <span>${Math.round(candidate.likelihood_score * 100)}% likelihood</span>
    <span>${candidate.source_agreement ? "full-recording sources agree" : "sources disagree"}</span>
  `;
  elements.previous.disabled = visibleIndex <= 0;
  elements.next.disabled =
    visibleIndex < 0 || visibleIndex >= state.visibleCandidates.length - 1;
  renderWindowTargets(candidate);
  renderEvidence(candidate);
  renderGainSummary(candidate);
  updateTimeline();
}

function applyAutomaticGains(candidate) {
  elements.gainA.value = String(candidate.speaker1_gain.default_gain);
  elements.gainB.value = String(candidate.speaker2_gain.default_gain);
  updateGain("a", elements.gainA.value);
  updateGain("b", elements.gainB.value);
  renderGainSummary(candidate);
}

function renderGainSummary(candidate) {
  const speaker1 = candidate.speaker1_gain;
  const speaker2 = candidate.speaker2_gain;
  if (
    speaker1.estimated_active_rms_dbfs === null ||
    speaker2.estimated_active_rms_dbfs === null
  ) {
    elements.gainSummary.textContent =
      "No stored loudness measurement is available; both tracks default to 1.00×.";
    return;
  }
  elements.gainSummary.textContent =
    `Automatic defaults target ${speaker1.target_active_rms_dbfs.toFixed(0)} dBFS ` +
    `active speech: A ${speaker1.estimated_active_rms_dbfs.toFixed(1)} dBFS ` +
    `→ ${speaker1.default_gain.toFixed(2)}×, ` +
    `B ${speaker2.estimated_active_rms_dbfs.toFixed(1)} dBFS ` +
    `→ ${speaker2.default_gain.toFixed(2)}×.`;
}

function renderWindowTargets(candidate) {
  elements.windowTargets.replaceChildren();
  for (const estimate of candidate.window_estimates) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = estimate.meaningful
      ? "window-target"
      : "window-target unreliable";
    button.innerHTML = `
      <span>${formatTime(estimate.start_seconds)}–${formatTime(estimate.end_seconds)}</span>
      <strong>${formatShift(estimate.estimated_b_shift_seconds)}</strong>
      <span>
        ${estimate.meaningful ? "usable local estimate" : "weak local evidence"} ·
        ${Math.round(estimate.bad_state_improvement * 100)}% bad-state reduction
      </span>
    `;
    button.addEventListener("click", () => {
      setShift(estimate.estimated_b_shift_seconds);
      seekTo(Math.max(timelineBounds().start, estimate.start_seconds - 2));
    });
    elements.windowTargets.append(button);
  }
}

function renderEvidence(candidate) {
  const labels = {
    conversation_annotation: "Stored EOT annotation",
    parakeet: "Parakeet timestamps",
    canary: "Canary timestamps",
  };
  elements.evidence.innerHTML = candidate.evidence
    .map(
      (item) => `
        <article class="evidence-card">
          <span>${labels[item.source]}</span>
          <strong>${formatShift(item.estimated_b_shift_seconds)}</strong>
          <span>
            overlap −${Math.round(item.overlap_reduction * 100)} pts ·
            silence −${Math.round(item.silence_reduction * 100)} pts
          </span>
        </article>
      `,
    )
    .join("");
}

function setShift(value) {
  state.bShiftSeconds = clamp(
    Math.round(Number(value) * 10) / 10,
    -MAXIMUM_SHIFT_SECONDS,
    MAXIMUM_SHIFT_SECONDS,
  );
  const bounds = timelineBounds();
  state.timelineSeconds = clamp(state.timelineSeconds, bounds.start, bounds.end);
  if (state.playing) {
    restartPlaybackClock();
  }
  syncPlayers(state.timelineSeconds, state.playing);
  updateTimeline();
}

function updateTimeline() {
  const bounds = timelineBounds();
  elements.shift.value = String(state.bShiftSeconds);
  elements.shiftNumber.value = state.bShiftSeconds.toFixed(1);
  elements.seek.min = String(bounds.start);
  elements.seek.max = String(bounds.end);
  elements.seek.value = String(state.timelineSeconds);
  elements.clock.textContent =
    `${formatTime(state.timelineSeconds)} / ${formatTime(bounds.end)}`;
  renderTimelineTicks(bounds);
  drawWaveforms();
}

function renderTimelineTicks(bounds) {
  const labels = [];
  const tickCount = 6;
  for (let index = 0; index <= tickCount; index += 1) {
    const seconds = bounds.start + ((bounds.end - bounds.start) * index) / tickCount;
    labels.push(`<span>${formatTime(seconds)}</span>`);
  }
  elements.timelineTicks.innerHTML = labels.join("");
}

function drawWaveforms() {
  const canvas = elements.waveform;
  const rectangle = canvas.getBoundingClientRect();
  if (rectangle.width === 0 || rectangle.height === 0) {
    return;
  }
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rectangle.width * pixelRatio);
  canvas.height = Math.round(rectangle.height * pixelRatio);
  const context = canvas.getContext("2d");
  context.scale(pixelRatio, pixelRatio);

  const width = rectangle.width;
  const height = rectangle.height;
  const rowHeight = height / 2;
  const bounds = timelineBounds();
  const timelineDuration = bounds.end - bounds.start;
  const xForTime = (seconds) => ((seconds - bounds.start) / timelineDuration) * width;

  context.fillStyle = "#f4f6f3";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#dce3e0";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, rowHeight);
  context.lineTo(width, rowHeight);
  for (let minute = 0; minute <= state.durationSeconds; minute += 60) {
    const x = xForTime(minute);
    context.moveTo(x, 0);
    context.lineTo(x, height);
  }
  context.stroke();

  drawEnvelope(
    context,
    state.waveformA,
    0,
    0,
    rowHeight,
    "#14786e",
    xForTime,
  );
  drawEnvelope(
    context,
    state.waveformB,
    state.bShiftSeconds,
    rowHeight,
    rowHeight,
    "#c16b32",
    xForTime,
  );

  const playheadX = xForTime(state.timelineSeconds);
  context.strokeStyle = "#1f292c";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(playheadX, 0);
  context.lineTo(playheadX, height);
  context.stroke();
}

function drawEnvelope(
  context,
  waveform,
  timelineOffset,
  rowTop,
  rowHeight,
  color,
  xForTime,
) {
  if (!waveform || waveform.points.length === 0) {
    context.fillStyle = "#87938f";
    context.font = "12px sans-serif";
    context.fillText("Loading waveform…", 38, rowTop + rowHeight / 2);
    return;
  }
  const center = rowTop + rowHeight / 2;
  const amplitude = rowHeight * 0.42;
  const secondsPerPoint = waveform.duration_seconds / waveform.points.length;
  context.strokeStyle = color;
  context.globalAlpha = 0.84;
  context.lineWidth = 1;
  context.beginPath();
  waveform.points.forEach((point, index) => {
    const sourceSeconds = index * secondsPerPoint;
    const x = xForTime(timelineOffset + sourceSeconds);
    context.moveTo(x, center - point.maximum_amplitude * amplitude);
    context.lineTo(x, center - point.minimum_amplitude * amplitude);
  });
  context.stroke();
  context.globalAlpha = 1;
}

function seekTo(seconds) {
  const bounds = timelineBounds();
  state.timelineSeconds = clamp(Number(seconds), bounds.start, bounds.end);
  if (state.playing) {
    restartPlaybackClock();
  }
  syncPlayers(state.timelineSeconds, state.playing);
  updateTimeline();
}

async function setupAudioGraph() {
  if (state.audioContext) {
    if (state.audioContext.state === "suspended") {
      await state.audioContext.resume();
    }
    return;
  }
  state.audioContext = new AudioContext();
  const sourceA = state.audioContext.createMediaElementSource(elements.audioA);
  const sourceB = state.audioContext.createMediaElementSource(elements.audioB);
  state.gainNodeA = state.audioContext.createGain();
  state.gainNodeB = state.audioContext.createGain();
  state.masterLimiter = state.audioContext.createDynamicsCompressor();
  state.masterLimiter.threshold.value = -1;
  state.masterLimiter.knee.value = 0;
  state.masterLimiter.ratio.value = 20;
  state.masterLimiter.attack.value = 0.003;
  state.masterLimiter.release.value = 0.08;
  sourceA.connect(state.gainNodeA).connect(state.masterLimiter);
  sourceB.connect(state.gainNodeB).connect(state.masterLimiter);
  state.masterLimiter.connect(state.audioContext.destination);
  updateGain("a", elements.gainA.value);
  updateGain("b", elements.gainB.value);
}

async function play() {
  if (state.startingPlayback) {
    return;
  }
  if (state.playing) {
    pause();
    return;
  }
  const bounds = timelineBounds();
  if (state.timelineSeconds >= bounds.end - 0.01) {
    state.timelineSeconds = bounds.start;
  }
  state.startingPlayback = true;
  elements.play.disabled = true;
  setPlaybackStatus("Preparing both speaker tracks…", false);
  try {
    await setupAudioGraph();
    await Promise.all([
      ensureMediaReady(elements.audioA),
      ensureMediaReady(elements.audioB),
    ]);
    await preparePlayers(state.timelineSeconds);
    await startActivePlayers(state.timelineSeconds);
    state.playing = true;
    elements.play.textContent = "Pause";
    updateActiveTrackStatus();
    restartPlaybackClock();
    state.animationFrame = requestAnimationFrame(playbackFrame);
  } catch (error) {
    pause();
    setPlaybackStatus(`Playback failed: ${error.message}`, true);
  } finally {
    state.startingPlayback = false;
    elements.play.disabled = false;
  }
}

function pause() {
  if (state.playing) {
    updateTimelineFromClock();
  }
  state.playing = false;
  elements.audioA.pause();
  elements.audioB.pause();
  elements.play.textContent = "Play both";
  if (state.animationFrame !== null) {
    cancelAnimationFrame(state.animationFrame);
    state.animationFrame = null;
  }
  updateTimeline();
}

function setPlaybackStatus(message, isError) {
  elements.playbackStatus.textContent = message;
  elements.playbackStatus.classList.toggle("error", isError);
}

function updateActiveTrackStatus() {
  const speaker1Active =
    state.timelineSeconds >= 0 && state.timelineSeconds < state.durationSeconds;
  const speaker2Seconds = state.timelineSeconds - state.bShiftSeconds;
  const speaker2Active =
    speaker2Seconds >= 0 && speaker2Seconds < state.durationSeconds;
  if (speaker1Active && speaker2Active) {
    setPlaybackStatus("Playing speaker A and speaker B.", false);
  } else if (speaker1Active) {
    setPlaybackStatus("Playing speaker A; speaker B is outside the timeline here.", false);
  } else if (speaker2Active) {
    setPlaybackStatus("Playing speaker B; speaker A is outside the timeline here.", false);
  } else {
    setPlaybackStatus("Neither track has audio at this timeline position.", false);
  }
}

function ensureMediaReady(audio) {
  if (audio.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const ready = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(new Error(`Could not load ${audio === elements.audioA ? "speaker A" : "speaker B"}.`));
    };
    const cleanup = () => {
      audio.removeEventListener("canplay", ready);
      audio.removeEventListener("error", failed);
    };
    audio.addEventListener("canplay", ready, { once: true });
    audio.addEventListener("error", failed, { once: true });
  });
}

async function preparePlayers(timelineSeconds) {
  await Promise.all([
    preparePlayer(elements.audioA, timelineSeconds),
    preparePlayer(elements.audioB, timelineSeconds - state.bShiftSeconds),
  ]);
}

function preparePlayer(audio, sourceSeconds) {
  if (sourceSeconds < 0 || sourceSeconds >= state.durationSeconds) {
    audio.pause();
    return Promise.resolve();
  }
  if (Math.abs(audio.currentTime - sourceSeconds) <= 0.02) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const seeked = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(new Error(`Could not seek ${audio === elements.audioA ? "speaker A" : "speaker B"}.`));
    };
    const cleanup = () => {
      audio.removeEventListener("seeked", seeked);
      audio.removeEventListener("error", failed);
    };
    audio.addEventListener("seeked", seeked, { once: true });
    audio.addEventListener("error", failed, { once: true });
    audio.currentTime = sourceSeconds;
  });
}

async function startActivePlayers(timelineSeconds) {
  const playbackPromises = [];
  if (timelineSeconds >= 0 && timelineSeconds < state.durationSeconds) {
    playbackPromises.push(elements.audioA.play());
  }
  const speaker2Seconds = timelineSeconds - state.bShiftSeconds;
  if (speaker2Seconds >= 0 && speaker2Seconds < state.durationSeconds) {
    playbackPromises.push(elements.audioB.play());
  }
  await Promise.all(playbackPromises);
}

function restartPlaybackClock() {
  state.playbackTimelineStart = state.timelineSeconds;
  state.playbackClockStart = performance.now();
}

function updateTimelineFromClock() {
  if (!state.playing) {
    return;
  }
  const elapsedSeconds = (performance.now() - state.playbackClockStart) / 1000;
  state.timelineSeconds = state.playbackTimelineStart + elapsedSeconds;
}

function playbackFrame() {
  if (!state.playing) {
    return;
  }
  updateTimelineFromClock();
  const bounds = timelineBounds();
  if (state.timelineSeconds >= bounds.end) {
    state.timelineSeconds = bounds.end;
    pause();
    return;
  }
  syncPlayers(state.timelineSeconds, true);
  elements.seek.value = String(state.timelineSeconds);
  elements.clock.textContent =
    `${formatTime(state.timelineSeconds)} / ${formatTime(bounds.end)}`;
  updateActiveTrackStatus();
  drawWaveforms();
  state.animationFrame = requestAnimationFrame(playbackFrame);
}

function syncPlayers(timelineSeconds, shouldPlay) {
  syncPlayer(
    elements.audioA,
    timelineSeconds,
    state.durationSeconds,
    shouldPlay,
  );
  syncPlayer(
    elements.audioB,
    timelineSeconds - state.bShiftSeconds,
    state.durationSeconds,
    shouldPlay,
  );
}

function syncPlayer(audio, sourceSeconds, durationSeconds, shouldPlay) {
  const active = sourceSeconds >= 0 && sourceSeconds < durationSeconds;
  if (!active) {
    audio.pause();
    return;
  }
  if (Math.abs(audio.currentTime - sourceSeconds) > 0.25) {
    audio.currentTime = sourceSeconds;
  }
  if (shouldPlay && audio.paused) {
    audio.play().catch((error) => {
      pause();
      setPlaybackStatus(`Playback failed: ${error.message}`, true);
    });
  } else if (!shouldPlay) {
    audio.pause();
  }
}

function updateGain(side, value) {
  const gain = Number(value);
  if (side === "a") {
    elements.gainAValue.value = `${gain.toFixed(2)}×`;
    if (state.gainNodeA) {
      state.gainNodeA.gain.value = gain;
    }
  } else {
    elements.gainBValue.value = `${gain.toFixed(2)}×`;
    if (state.gainNodeB) {
      state.gainNodeB.gain.value = gain;
    }
  }
}

function moveSelection(delta) {
  const index = selectedVisibleIndex();
  const target = state.visibleCandidates[index + delta];
  if (target) {
    void selectCandidate(target.external_id);
  }
}

elements.filter.addEventListener("input", () => {
  const query = elements.filter.value.trim().toLowerCase();
  state.visibleCandidates = state.candidates.filter((candidate) =>
    candidate.external_id.toLowerCase().includes(query),
  );
  elements.candidateCount.textContent = `${state.visibleCandidates.length} candidates`;
  renderCandidateList();
  renderCandidateDetails();
});
elements.previous.addEventListener("click", () => moveSelection(-1));
elements.next.addEventListener("click", () => moveSelection(1));
elements.usePrediction.addEventListener("click", () => {
  const candidate = selectedCandidate();
  if (candidate) {
    setShift(candidate.estimated_b_shift_seconds);
  }
});
elements.shift.addEventListener("input", () => setShift(elements.shift.value));
elements.shiftNumber.addEventListener("change", () =>
  setShift(elements.shiftNumber.value),
);
elements.play.addEventListener("click", () => {
  void play();
});
elements.seek.addEventListener("input", () => seekTo(elements.seek.value));
elements.gainA.addEventListener("input", () => updateGain("a", elements.gainA.value));
elements.gainB.addEventListener("input", () => updateGain("b", elements.gainB.value));
elements.useAutoGain.addEventListener("click", () => {
  const candidate = selectedCandidate();
  if (candidate) {
    applyAutomaticGains(candidate);
  }
});
elements.waveform.addEventListener("click", (event) => {
  const rectangle = elements.waveform.getBoundingClientRect();
  const fraction = (event.clientX - rectangle.left) / rectangle.width;
  const bounds = timelineBounds();
  seekTo(bounds.start + fraction * (bounds.end - bounds.start));
});
window.addEventListener("resize", drawWaveforms);
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) {
    return;
  }
  if (event.code === "Space") {
    event.preventDefault();
    void play();
  } else if (event.code === "ArrowUp") {
    event.preventDefault();
    moveSelection(-1);
  } else if (event.code === "ArrowDown") {
    event.preventDefault();
    moveSelection(1);
  } else if (event.key === "[") {
    setShift(state.bShiftSeconds - 0.1);
  } else if (event.key === "]") {
    setShift(state.bShiftSeconds + 0.1);
  }
});

void loadCandidates();
