const MAXIMUM_SHIFT_SECONDS = 12;
const DEFAULT_DURATION_SECONDS = 180;
const PLAYBACK_WINDOW_SECONDS = 180;
const PLAYBACK_WINDOW_PREROLL_SECONDS = 30;
const WAVEFORM_POINT_COUNT = 1800;

const elements = {
  workspace: document.querySelector("#workspace"),
  errorState: document.querySelector("#error-state"),
  coverage: document.querySelector("#coverage"),
  candidateCount: document.querySelector("#candidate-count"),
  candidateList: document.querySelector("#candidate-list"),
  filter: document.querySelector("#filter"),
  hideReviewed: document.querySelector("#hide-reviewed"),
  candidateSort: document.querySelector("#candidate-sort"),
  position: document.querySelector("#position"),
  sampleName: document.querySelector("#sample-name"),
  sampleSummary: document.querySelector("#sample-summary"),
  previous: document.querySelector("#previous"),
  next: document.querySelector("#next"),
  usePrediction: document.querySelector("#use-prediction"),
  toggleFullRecording: document.querySelector("#toggle-full-recording"),
  saveReview: document.querySelector("#save-review"),
  reviewSaveStatus: document.querySelector("#review-save-status"),
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
};

const state = {
  candidates: [],
  visibleCandidates: [],
  selectedExternalId: null,
  waveformA: null,
  waveformB: null,
  durationSeconds: DEFAULT_DURATION_SECONDS,
  durationA: DEFAULT_DURATION_SECONDS,
  durationB: DEFAULT_DURATION_SECONDS,
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
  audioBufferA: null,
  audioBufferB: null,
  audioBufferKey: null,
  audioBufferStartA: 0,
  audioBufferStartB: 0,
  sourceNodeA: null,
  sourceNodeB: null,
  gainNormalization: null,
  fullRecordingMode: false,
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
      state.durationA,
      state.durationB + state.bShiftSeconds,
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
    const belowCandidateThreshold =
      payload.analyzed_session_count - payload.offset_candidate_count;
    elements.coverage.textContent =
      `${payload.analyzed_session_count} sessions analyzed · ` +
      `${payload.offset_candidate_count} offset candidates · ` +
      `${belowCandidateThreshold} below the candidate threshold · ` +
      `${payload.excluded_session_count} without complete timing evidence`;
    elements.workspace.hidden = false;
    applyCandidateFilters();
    if (state.visibleCandidates.length === 0) {
      setError("No analyzed sessions match the current review filters.");
      return;
    }
    await selectCandidate(state.visibleCandidates[0].external_id);
  } catch (error) {
    setError(`Could not load synchronization candidates: ${error.message}`);
  }
}

function applyCandidateFilters() {
  const query = elements.filter.value.trim().toLowerCase();
  const hideReviewed = elements.hideReviewed.checked;
  state.visibleCandidates = state.candidates
    .filter(
      (candidate) =>
        candidate.external_id.toLowerCase().includes(query) &&
        (!hideReviewed || candidate.alignment_estimate_origin !== "reviewed"),
    )
    .sort(candidateComparator(elements.candidateSort.value));
  const reviewedCount = state.candidates.filter(
    (candidate) => candidate.alignment_estimate_origin === "reviewed",
  ).length;
  elements.candidateCount.textContent =
    `${state.visibleCandidates.length} shown · ${reviewedCount} reviewed`;
  renderCandidateList();
}

function candidateComparator(sortMode) {
  if (sortMode === "confidence_descending") {
    return (left, right) =>
      right.offset_confidence_score - left.offset_confidence_score ||
      right.likelihood_score - left.likelihood_score;
  }
  if (sortMode === "likelihood_descending") {
    return (left, right) =>
      right.likelihood_score - left.likelihood_score ||
      left.offset_confidence_score - right.offset_confidence_score;
  }
  if (sortMode === "sample_id") {
    return (left, right) => left.external_id.localeCompare(right.external_id);
  }
  return (left, right) =>
    left.offset_confidence_score - right.offset_confidence_score ||
    right.likelihood_score - left.likelihood_score;
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
        <span class="score">likelihood ${Math.round(candidate.likelihood_score * 100)}%</span>
      </span>
      <span class="candidate-main">
        <span class="badge ${candidate.offset_pattern}">${candidate.offset_pattern}</span>
        <strong>${formatShift(candidate.estimated_b_shift_seconds)}</strong>
      </span>
      <span class="candidate-meta">
        <span>
          ${candidate.is_offset_candidate ? "offset candidate" : "below threshold"} ·
          ${candidate.overlap_silence_cycle_count} cycles
        </span>
        <span>
          ${candidate.alignment_estimate_origin === "reviewed"
            ? "manually reviewed"
            : candidate.alignment_estimate_origin === "unresolved"
              ? "alignment unresolved"
            : candidate.source_agreement
              ? "sources agree"
              : "mixed evidence"}
        </span>
      </span>
      <span class="confidence-row">
        <span>offset confidence</span>
        <span>${Math.round(candidate.offset_confidence_score * 100)}%</span>
      </span>
      <span class="confidence-track" title="Heuristic offset confidence">
        <i style="width: ${Math.round(candidate.offset_confidence_score * 100)}%"></i>
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
  state.gainNormalization = candidate;
  applyAutomaticGains(candidate);
  state.waveformA = null;
  state.waveformB = null;
  state.audioBufferA = null;
  state.audioBufferB = null;
  state.audioBufferKey = null;
  state.audioBufferStartA = 0;
  state.audioBufferStartB = 0;
  state.durationSeconds = DEFAULT_DURATION_SECONDS;
  state.durationA = DEFAULT_DURATION_SECONDS;
  state.durationB = DEFAULT_DURATION_SECONDS;
  setPlaybackStatus("", false);
  setReviewSaveStatus("", false);
  state.selectionVersion += 1;
  const selectionVersion = state.selectionVersion;
  renderCandidateList();
  document
    .querySelector(`.candidate[data-external-id="${externalId}"]`)
    ?.scrollIntoView({ block: "nearest" });
  renderCandidateDetails();
  drawWaveforms();
  const gainResultPromise = fetchSpeechGains(candidate.sample_id)
    .then((gainNormalization) => ({ gainNormalization, error: null }))
    .catch((error) => ({ gainNormalization: null, error }));

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
    state.durationA = waveformA.duration_seconds;
    state.durationB = waveformB.duration_seconds;
    state.durationSeconds = Math.min(
      Math.max(waveformA.duration_seconds, waveformB.duration_seconds),
      state.fullRecordingMode ? Number.POSITIVE_INFINITY : DEFAULT_DURATION_SECONDS,
    );
    const gainResult = await gainResultPromise;
    if (selectionVersion !== state.selectionVersion) {
      return;
    }
    if (gainResult.gainNormalization) {
      state.gainNormalization = gainResult.gainNormalization;
      applyAutomaticGains(gainResult.gainNormalization);
    } else {
      elements.gainSummary.textContent +=
        ` Speech-only measurement failed: ${gainResult.error.message}`;
    }
    updateTimeline();
  } catch (error) {
    if (selectionVersion === state.selectionVersion) {
      setError(`Could not load waveforms for ${candidate.external_id}: ${error.message}`);
    }
  }
}

async function fetchSpeechGains(sampleId) {
  const response = await fetch(`/api/synchronization-review/gain/${sampleId}`);
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Gain request failed with ${response.status}`);
  }
  return response.json();
}

async function fetchWaveform(sampleId, side) {
  const response = await fetch(
    `/api/dataset-dashboard/waveform/${sampleId}/${side}` +
      `?points=${WAVEFORM_POINT_COUNT}&trimmed=${!state.fullRecordingMode}`,
  );
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `Waveform request failed with ${response.status}`);
  }
  return response.json();
}

async function toggleFullRecordingMode() {
  const candidate = selectedCandidate();
  if (!candidate) {
    return;
  }
  pause();
  state.fullRecordingMode = !state.fullRecordingMode;
  state.selectionVersion += 1;
  const selectionVersion = state.selectionVersion;
  state.waveformA = null;
  state.waveformB = null;
  state.audioBufferA = null;
  state.audioBufferB = null;
  state.audioBufferKey = null;
  elements.toggleFullRecording.disabled = true;
  renderCandidateDetails();
  drawWaveforms();
  setPlaybackStatus(
    state.fullRecordingMode
      ? "Loading full-recording waveforms…"
      : "Loading first-three-minute waveforms…",
    false,
  );
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
    state.durationA = waveformA.duration_seconds;
    state.durationB = waveformB.duration_seconds;
    state.durationSeconds = Math.max(
      waveformA.duration_seconds,
      waveformB.duration_seconds,
    );
    const bounds = timelineBounds();
    state.timelineSeconds = clamp(state.timelineSeconds, bounds.start, bounds.end);
    setPlaybackStatus(
      state.fullRecordingMode
        ? "Full recording loaded. Seek anywhere, then press Play."
        : "Playback limited to the first three minutes.",
      false,
    );
    updateTimeline();
  } catch (error) {
    setPlaybackStatus(`Could not load recording waveforms: ${error.message}`, true);
  } finally {
    elements.toggleFullRecording.disabled = false;
  }
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
    ${candidate.alignment_estimate_origin === "reviewed" ? '<span class="badge reviewed">reviewed</span>' : ""}
    ${candidate.alignment_estimate_origin === "unresolved" ? '<span class="badge unresolved">unresolved</span>' : ""}
    <span>${Math.round(candidate.likelihood_score * 100)}% likelihood</span>
    <span>
      ${candidate.static_offset_valid ? "static estimate" : "review anchor"}
      ${formatShift(candidate.estimated_b_shift_seconds)}
    </span>
    <span>full recording ${formatShift(candidate.full_recording_estimated_b_shift_seconds)}</span>
    <span>offset confidence ${Math.round(candidate.offset_confidence_score * 100)}%</span>
    <span>${candidate.source_agreement ? "full-recording sources agree" : "sources disagree"}</span>
    ${candidate.duration_mismatch_seconds == null
      ? ""
      : `<span>track duration gap ${candidate.duration_mismatch_seconds.toFixed(2)} s</span>`}
    ${candidate.drift_warning
      ? `<span class="drift-warning">${candidate.drift_warning}</span>`
      : ""}
  `;
  elements.previous.disabled = visibleIndex <= 0;
  elements.next.disabled =
    visibleIndex < 0 || visibleIndex >= state.visibleCandidates.length - 1;
  elements.saveReview.textContent =
    candidate.alignment_estimate_origin === "reviewed"
      ? "Update reviewed offset"
      : "Save current offset as reviewed";
  elements.toggleFullRecording.textContent = state.fullRecordingMode
    ? "Use first 3 minutes"
    : "Scrub full recording";
  renderWindowTargets(candidate);
  renderEvidence(candidate);
  renderGainSummary(state.gainNormalization || candidate);
  updateTimeline();
}

function setReviewSaveStatus(message, isError) {
  elements.reviewSaveStatus.textContent = message;
  elements.reviewSaveStatus.classList.toggle("error", isError);
}

async function saveCurrentOffsetAsReviewed() {
  const candidate = selectedCandidate();
  if (!candidate) {
    return;
  }
  const previousVisibleIndex = selectedVisibleIndex();
  elements.saveReview.disabled = true;
  setReviewSaveStatus(`Saving ${formatShift(state.bShiftSeconds)}…`, false);
  try {
    const response = await fetch(
      `/api/synchronization-review/reviews/${candidate.sample_id}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speaker2_shift_seconds: state.bShiftSeconds,
        }),
      },
    );
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Save failed with ${response.status}`);
    }
    const savedReview = await response.json();
    candidate.estimated_b_shift_seconds = savedReview.speaker2_shift_seconds;
    candidate.alignment_estimate_origin = "reviewed";
    candidate.offset_confidence_score = 1;
    state.bShiftSeconds = savedReview.speaker2_shift_seconds;
    applyCandidateFilters();
    const savedMessage =
      `Saved ${candidate.external_id.toUpperCase()} at ` +
      `${formatShift(savedReview.speaker2_shift_seconds)}.`;
    if (
      !state.visibleCandidates.some(
        (visibleCandidate) => visibleCandidate.external_id === candidate.external_id,
      ) &&
      state.visibleCandidates.length > 0
    ) {
      const nextCandidate =
        state.visibleCandidates[
          Math.max(
            0,
            Math.min(previousVisibleIndex, state.visibleCandidates.length - 1),
          )
        ];
      await selectCandidate(nextCandidate.external_id);
    } else {
      renderCandidateDetails();
    }
    setReviewSaveStatus(
      savedMessage,
      false,
    );
  } catch (error) {
    setReviewSaveStatus(`Could not save review: ${error.message}`, true);
  } finally {
    elements.saveReview.disabled = false;
  }
}

function applyAutomaticGains(gainNormalization) {
  elements.gainA.value = String(gainNormalization.speaker1_gain.default_gain);
  elements.gainB.value = String(gainNormalization.speaker2_gain.default_gain);
  updateGain("a", elements.gainA.value);
  updateGain("b", elements.gainB.value);
  renderGainSummary(gainNormalization);
}

function renderGainSummary(gainNormalization) {
  const speaker1 = gainNormalization.speaker1_gain;
  const speaker2 = gainNormalization.speaker2_gain;
  if (
    speaker1.estimated_active_rms_dbfs === null ||
    speaker2.estimated_active_rms_dbfs === null
  ) {
    elements.gainSummary.textContent =
      "No stored loudness measurement is available; both tracks default to 1.00×.";
    return;
  }
  const measurementDescription =
    speaker1.measurement_basis === "annotated_speech"
      ? `Speech-only RMS across A ${speaker1.measured_speech_duration_seconds.toFixed(1)} s ` +
        `and B ${speaker2.measured_speech_duration_seconds.toFixed(1)} s. `
      : "Temporary whole-track estimate while speech-only RMS loads. ";
  elements.gainSummary.textContent =
    measurementDescription +
    `Defaults target ${speaker1.target_active_rms_dbfs.toFixed(0)} dBFS ` +
    `active speech: A ${speaker1.estimated_active_rms_dbfs.toFixed(1)} dBFS ` +
    `→ ${speaker1.default_gain.toFixed(2)}×, ` +
    `B ${speaker2.estimated_active_rms_dbfs.toFixed(1)} dBFS ` +
    `→ ${speaker2.default_gain.toFixed(2)}×.`;
}

function renderWindowTargets(candidate) {
  const sourceLabels = {
    audio_activity: "Audio activity",
    conversation_annotation: "EOT annotation",
    parakeet: "Parakeet",
    canary: "Canary",
  };
  elements.windowTargets.replaceChildren();
  for (const estimate of candidate.window_estimates) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = estimate.meaningful
      ? "window-target"
      : "window-target unreliable";
    button.innerHTML = `
      <span>
        ${sourceLabels[estimate.source]} ·
        ${formatTime(estimate.start_seconds)}–${formatTime(estimate.end_seconds)}
      </span>
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
    restartBufferPlayback();
  }
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
    restartBufferPlayback();
  }
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
  state.gainNodeA = state.audioContext.createGain();
  state.gainNodeB = state.audioContext.createGain();
  state.masterLimiter = state.audioContext.createDynamicsCompressor();
  state.masterLimiter.threshold.value = -1;
  state.masterLimiter.knee.value = 0;
  state.masterLimiter.ratio.value = 20;
  state.masterLimiter.attack.value = 0.003;
  state.masterLimiter.release.value = 0.08;
  state.gainNodeA.connect(state.masterLimiter);
  state.gainNodeB.connect(state.masterLimiter);
  state.masterLimiter.connect(state.audioContext.destination);
  updateGain("a", elements.gainA.value);
  updateGain("b", elements.gainB.value);
}

async function loadAudioBuffers(selectionVersion) {
  const candidate = selectedCandidate();
  if (!candidate) {
    throw new Error("No synchronization candidate is selected.");
  }
  const sourceSecondsA = state.timelineSeconds;
  const sourceSecondsB = state.timelineSeconds - state.bShiftSeconds;
  const windowStartA = playbackWindowStart(sourceSecondsA, state.durationA);
  const windowStartB = playbackWindowStart(sourceSecondsB, state.durationB);
  const bufferKey =
    `${candidate.external_id}:${windowStartA.toFixed(1)}:${windowStartB.toFixed(1)}`;
  if (
    state.audioBufferKey === bufferKey &&
    state.audioBufferA &&
    state.audioBufferB
  ) {
    return true;
  }
  const [responseA, responseB] = await Promise.all([
    fetch(audioWindowUrl(candidate.sample_id, "speaker1", windowStartA)),
    fetch(audioWindowUrl(candidate.sample_id, "speaker2", windowStartB)),
  ]);
  if (!responseA.ok || !responseB.ok) {
    throw new Error("Could not download both speaker tracks.");
  }
  const [bytesA, bytesB] = await Promise.all([
    responseA.arrayBuffer(),
    responseB.arrayBuffer(),
  ]);
  if (selectionVersion !== state.selectionVersion) {
    return false;
  }
  const [audioBufferA, audioBufferB] = await Promise.all([
    state.audioContext.decodeAudioData(bytesA),
    state.audioContext.decodeAudioData(bytesB),
  ]);
  if (selectionVersion !== state.selectionVersion) {
    return false;
  }
  state.audioBufferA = audioBufferA;
  state.audioBufferB = audioBufferB;
  state.audioBufferKey = bufferKey;
  state.audioBufferStartA = windowStartA;
  state.audioBufferStartB = windowStartB;
  return true;
}

function playbackWindowStart(sourceSeconds, trackDuration) {
  if (!state.fullRecordingMode) {
    return 0;
  }
  if (sourceSeconds >= trackDuration) {
    return Math.max(0, Math.floor(trackDuration - 1));
  }
  return Math.floor(
    Math.max(0, sourceSeconds - PLAYBACK_WINDOW_PREROLL_SECONDS),
  );
}

function audioWindowUrl(sampleId, side, startSeconds) {
  return (
    `/api/synchronization-review/audio-window/${sampleId}/${side}` +
    `?start_seconds=${startSeconds}&duration_seconds=${PLAYBACK_WINDOW_SECONDS}`
  );
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
  setPlaybackStatus("Decoding a three-minute playback window…", false);
  const selectionVersion = state.selectionVersion;
  try {
    await setupAudioGraph();
    const loaded = await loadAudioBuffers(selectionVersion);
    if (!loaded) {
      return;
    }
    if (state.audioContext.state === "suspended") {
      await state.audioContext.resume();
    }
    state.playing = true;
    elements.play.textContent = "Pause";
    startBufferPlayback();
    updateActiveTrackStatus();
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
  stopPlaybackSources();
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
    state.timelineSeconds >= 0 && state.timelineSeconds < state.durationA;
  const speaker2Seconds = state.timelineSeconds - state.bShiftSeconds;
  const speaker2Active =
    speaker2Seconds >= 0 && speaker2Seconds < state.durationB;
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

function startBufferPlayback() {
  stopPlaybackSources();
  const startTime = state.audioContext.currentTime + 0.02;
  state.playbackTimelineStart = state.timelineSeconds;
  state.playbackClockStart = startTime;
  state.sourceNodeA = scheduleTrack(
    state.audioBufferA,
    0,
    state.audioBufferStartA,
    state.gainNodeA,
    startTime,
  );
  state.sourceNodeB = scheduleTrack(
    state.audioBufferB,
    state.bShiftSeconds,
    state.audioBufferStartB,
    state.gainNodeB,
    startTime,
  );
}

function scheduleTrack(
  audioBuffer,
  trackTimelineStart,
  bufferSourceStart,
  gainNode,
  startTime,
) {
  if (!audioBuffer || !gainNode) {
    return null;
  }
  const sourceSeconds =
    state.timelineSeconds - trackTimelineStart - bufferSourceStart;
  if (sourceSeconds >= audioBuffer.duration) {
    return null;
  }
  const sourceNode = state.audioContext.createBufferSource();
  sourceNode.buffer = audioBuffer;
  sourceNode.connect(gainNode);
  if (sourceSeconds >= 0) {
    sourceNode.start(startTime, sourceSeconds);
  } else {
    sourceNode.start(startTime - sourceSeconds, 0);
  }
  return sourceNode;
}

function stopPlaybackSources() {
  stopSourceNode(state.sourceNodeA);
  stopSourceNode(state.sourceNodeB);
  state.sourceNodeA = null;
  state.sourceNodeB = null;
}

function stopSourceNode(sourceNode) {
  if (!sourceNode) {
    return;
  }
  try {
    sourceNode.stop();
  } catch (error) {
    if (error.name !== "InvalidStateError") {
      throw error;
    }
  }
  sourceNode.disconnect();
}

function restartBufferPlayback() {
  if (!state.playing || !state.audioBufferA || !state.audioBufferB) {
    return;
  }
  if (buffersCoverTimeline()) {
    startBufferPlayback();
    return;
  }
  stopPlaybackSources();
  state.playing = false;
  state.audioBufferKey = null;
  elements.play.textContent = "Play both";
  if (state.animationFrame !== null) {
    cancelAnimationFrame(state.animationFrame);
    state.animationFrame = null;
  }
  setPlaybackStatus("Press Play to load audio around this position.", false);
}

function buffersCoverTimeline() {
  return (
    bufferCoversSource(
      state.timelineSeconds,
      state.durationA,
      state.audioBufferA,
      state.audioBufferStartA,
    ) &&
    bufferCoversSource(
      state.timelineSeconds - state.bShiftSeconds,
      state.durationB,
      state.audioBufferB,
      state.audioBufferStartB,
    )
  );
}

function bufferCoversSource(
  sourceSeconds,
  trackDuration,
  audioBuffer,
  bufferSourceStart,
) {
  if (sourceSeconds < 0 || sourceSeconds >= trackDuration) {
    return true;
  }
  return (
    audioBuffer &&
    sourceSeconds >= bufferSourceStart &&
    sourceSeconds < bufferSourceStart + audioBuffer.duration
  );
}

function updateTimelineFromClock() {
  if (!state.playing || !state.audioContext) {
    return;
  }
  const elapsedSeconds = Math.max(
    0,
    state.audioContext.currentTime - state.playbackClockStart,
  );
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
  if (
    state.fullRecordingMode &&
    state.timelineSeconds >= currentBufferTimelineEnd() - 0.02
  ) {
    pause();
    state.audioBufferKey = null;
    setPlaybackStatus(
      "Playback window ended. Press Play to load the next three-minute window.",
      false,
    );
    return;
  }
  elements.seek.value = String(state.timelineSeconds);
  elements.clock.textContent =
    `${formatTime(state.timelineSeconds)} / ${formatTime(bounds.end)}`;
  updateActiveTrackStatus();
  drawWaveforms();
  state.animationFrame = requestAnimationFrame(playbackFrame);
}

function currentBufferTimelineEnd() {
  const speaker1End =
    state.audioBufferStartA + (state.audioBufferA?.duration || 0);
  const speaker2End =
    state.bShiftSeconds +
    state.audioBufferStartB +
    (state.audioBufferB?.duration || 0);
  return Math.max(speaker1End, speaker2End);
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

function refreshCandidateFilterSelection() {
  applyCandidateFilters();
  const selectedStillVisible = state.visibleCandidates.some(
    (candidate) => candidate.external_id === state.selectedExternalId,
  );
  if (selectedStillVisible) {
    renderCandidateDetails();
  } else if (state.visibleCandidates.length > 0) {
    void selectCandidate(state.visibleCandidates[0].external_id);
  }
}

elements.filter.addEventListener("input", refreshCandidateFilterSelection);
elements.hideReviewed.addEventListener("change", refreshCandidateFilterSelection);
elements.candidateSort.addEventListener("change", refreshCandidateFilterSelection);
elements.previous.addEventListener("click", () => moveSelection(-1));
elements.next.addEventListener("click", () => moveSelection(1));
elements.usePrediction.addEventListener("click", () => {
  const candidate = selectedCandidate();
  if (candidate) {
    setShift(candidate.estimated_b_shift_seconds);
  }
});
elements.toggleFullRecording.addEventListener("click", () => {
  void toggleFullRecordingMode();
});
elements.saveReview.addEventListener("click", () => {
  void saveCurrentOffsetAsReviewed();
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
  if (state.gainNormalization) {
    applyAutomaticGains(state.gainNormalization);
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
