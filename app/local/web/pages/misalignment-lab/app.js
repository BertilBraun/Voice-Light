import { drawAnnotationTimelineRow } from "/pages/shared/annotation-timeline.js";

const QUEUE_SIZE = 50;
const QUEUE_SEED = "misalignment-lab-v1";
const state = {
  queue: [],
  index: 0,
  preview: null,
  progress: null,
  audioContext: null,
  buffers: [],
  sources: [],
  playbackStartedAt: null,
  playbackOffsetSeconds: 0,
  animationFrame: null,
  judging: false,
};

const elements = {
  queuePosition: document.querySelector("#queue-position"),
  reviewedCount: document.querySelector("#reviewed-count"),
  alignedCount: document.querySelector("#aligned-count"),
  quarantinedCount: document.querySelector("#quarantined-count"),
  unsureCount: document.querySelector("#unsure-count"),
  emptyState: document.querySelector("#empty-state"),
  review: document.querySelector("#review"),
  sampleName: document.querySelector("#sample-name"),
  candidateTime: document.querySelector("#candidate-time"),
  candidateReason: document.querySelector("#candidate-reason"),
  candidateDetails: document.querySelector("#candidate-details"),
  play: document.querySelector("#play"),
  seek: document.querySelector("#seek"),
  clock: document.querySelector("#clock"),
  playbackStatus: document.querySelector("#playback-status"),
  speaker1Timeline: document.querySelector("#speaker1-timeline"),
  speaker2Timeline: document.querySelector("#speaker2-timeline"),
  speaker1Events: document.querySelector("#speaker1-events"),
  speaker2Events: document.querySelector("#speaker2-events"),
  aligned: document.querySelector("#aligned"),
  misaligned: document.querySelector("#misaligned"),
  unsure: document.querySelector("#unsure"),
};

elements.play.addEventListener("click", () => void startPlayback(true));
elements.seek.addEventListener("input", () => {
  state.playbackOffsetSeconds = (Number(elements.seek.value) / 1000) * clipDuration();
  stopPlayback();
  updatePlaybackDisplay();
});
elements.seek.addEventListener("change", () => void startPlayback(true));
elements.aligned.addEventListener("click", () => void submitJudgment("plausibly_aligned"));
elements.misaligned.addEventListener("click", () => void submitJudgment("likely_misaligned"));
elements.unsure.addEventListener("click", () => void submitJudgment("unsure"));
for (const canvas of [elements.speaker1Timeline, elements.speaker2Timeline]) {
  canvas.addEventListener("click", (event) => {
    const bounds = canvas.getBoundingClientRect();
    state.playbackOffsetSeconds =
      Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)) * clipDuration();
    void startPlayback(true);
  });
}
document.addEventListener("keydown", (event) => {
  if (event.target !== document.body || state.judging) {
    return;
  }
  if (event.key.toLowerCase() === "a") {
    void submitJudgment("plausibly_aligned");
  } else if (event.key.toLowerCase() === "m") {
    void submitJudgment("likely_misaligned");
  } else if (event.key.toLowerCase() === "u") {
    void submitJudgment("unsure");
  } else if (event.key === " ") {
    event.preventDefault();
    void startPlayback(true);
  }
});
window.addEventListener("resize", drawTimelines);

void loadQueue();

async function loadQueue() {
  try {
    const query = new URLSearchParams({ seed: QUEUE_SEED, limit: String(QUEUE_SIZE) });
    const response = await fetch(`/api/misalignment-lab/queue?${query.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Queue request failed (${response.status})`);
    }
    state.queue = payload.candidates;
    state.progress = payload.progress;
    renderProgress();
    if (state.queue.length === 0) {
      showEmpty("No unreviewed, non-quarantined annotated sessions remain.");
      return;
    }
    await loadCandidate(0, true);
  } catch (error) {
    showEmpty(error instanceof Error ? error.message : "Could not load the review queue.");
  }
}

async function loadCandidate(index, autoplay) {
  if (index < 0 || index >= state.queue.length) {
    showEmpty(`Queue complete. Reviewed ${state.queue.length} high-value clips in this pass.`);
    return;
  }
  state.index = index;
  state.preview = null;
  state.playbackOffsetSeconds = 0;
  stopPlayback();
  setDecisionDisabled(true);
  const candidate = currentCandidate();
  elements.queuePosition.textContent = `${index + 1} / ${state.queue.length}`;
  elements.sampleName.textContent = candidate.external_id;
  elements.candidateTime.textContent =
    `${formatAbsoluteTime(candidate.window_start_seconds)}-${formatAbsoluteTime(candidate.window_end_seconds)}`;
  elements.candidateReason.textContent = "Loading raw WAV windows and stored full-duration annotations...";
  elements.review.hidden = false;
  elements.emptyState.hidden = true;
  try {
    const previewRequest = fetch(
      `/api/misalignment-lab/preview/${candidate.sample_id}/${candidate.candidate_id}`,
    );
    const audioRequests = ["speaker1", "speaker2"].map((side) => {
      const query = new URLSearchParams({
        start_seconds: String(candidate.window_start_seconds),
        duration_seconds: String(clipDuration(candidate)),
      });
      return fetch(
        `/api/synchronization-review/audio-window/${candidate.sample_id}/${side}?${query.toString()}`,
      );
    });
    const [previewResponse, ...audioResponses] = await Promise.all([
      previewRequest,
      ...audioRequests,
    ]);
    const previewPayload = await previewResponse.json();
    if (!previewResponse.ok) {
      throw new Error(previewPayload.detail || `Preview request failed (${previewResponse.status})`);
    }
    for (const response of audioResponses) {
      if (!response.ok) {
        throw new Error(`Audio window request failed (${response.status})`);
      }
    }
    state.preview = previewPayload;
    await decodeAudioWindows(audioResponses);
    renderCandidate();
    setDecisionDisabled(false);
    if (autoplay) {
      await startPlayback(false);
    }
  } catch (error) {
    elements.candidateReason.textContent =
      error instanceof Error ? error.message : "Could not load this candidate.";
    setDecisionDisabled(false);
  }
}

async function decodeAudioWindows(responses) {
  const audioContext = await ensureAudioContext(false);
  const waveBuffers = await Promise.all(responses.map((response) => response.arrayBuffer()));
  state.buffers = await Promise.all(
    waveBuffers.map((waveBuffer) => audioContext.decodeAudioData(waveBuffer)),
  );
}

async function ensureAudioContext(resume) {
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }
  if (resume && state.audioContext.state !== "running") {
    await state.audioContext.resume();
  }
  return state.audioContext;
}

async function startPlayback(fromUserGesture) {
  if (state.buffers.length !== 2) {
    return;
  }
  const audioContext = await ensureAudioContext(fromUserGesture);
  if (audioContext.state !== "running") {
    elements.playbackStatus.textContent =
      "Browser autoplay is paused. Press Start / replay both once; later clips will auto-play.";
    elements.playbackStatus.classList.add("warning");
    return;
  }
  stopPlayback();
  if (state.playbackOffsetSeconds >= clipDuration() - 0.02) {
    state.playbackOffsetSeconds = 0;
  }
  const scheduledTime = audioContext.currentTime + 0.05;
  state.sources = state.buffers.map((buffer) => {
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start(scheduledTime, state.playbackOffsetSeconds);
    return source;
  });
  state.playbackStartedAt = scheduledTime - state.playbackOffsetSeconds;
  elements.playbackStatus.textContent =
    "Playing both raw tracks on one AudioContext clock with exactly zero applied shift.";
  elements.playbackStatus.classList.remove("warning");
  schedulePlaybackUpdate();
}

function stopPlayback() {
  for (const source of state.sources) {
    try {
      source.stop();
    } catch {
      // A source can already have ended between animation frames.
    }
  }
  state.sources = [];
  state.playbackStartedAt = null;
  if (state.animationFrame !== null) {
    cancelAnimationFrame(state.animationFrame);
    state.animationFrame = null;
  }
}

function schedulePlaybackUpdate() {
  updatePlaybackDisplay();
  if (state.playbackStartedAt === null || !state.audioContext) {
    return;
  }
  const elapsed = state.audioContext.currentTime - state.playbackStartedAt;
  if (elapsed >= clipDuration()) {
    state.playbackOffsetSeconds = clipDuration();
    stopPlayback();
    updatePlaybackDisplay();
    return;
  }
  state.animationFrame = requestAnimationFrame(schedulePlaybackUpdate);
}

function playbackSeconds() {
  if (state.playbackStartedAt !== null && state.audioContext) {
    return Math.min(
      clipDuration(),
      Math.max(0, state.audioContext.currentTime - state.playbackStartedAt),
    );
  }
  return state.playbackOffsetSeconds;
}

function updatePlaybackDisplay() {
  const seconds = playbackSeconds();
  elements.seek.value = String(Math.round((seconds / clipDuration()) * 1000));
  elements.clock.textContent = `${formatClipTime(seconds)} / ${formatClipTime(clipDuration())}`;
  drawTimelines();
}

async function submitJudgment(judgment) {
  const candidate = currentCandidate();
  if (!candidate || state.judging) {
    return;
  }
  state.judging = true;
  stopPlayback();
  setDecisionDisabled(true);
  try {
    await ensureAudioContext(true);
    const response = await fetch("/api/misalignment-lab/judgments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate_id: candidate.candidate_id,
        sample_id: candidate.sample_id,
        window_start_seconds: candidate.window_start_seconds,
        window_end_seconds: candidate.window_end_seconds,
        judgment,
        queue_seed: QUEUE_SEED,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Judgment request failed (${response.status})`);
    }
    state.progress = payload.progress;
    renderProgress();
    if (payload.session_quarantined) {
      elements.playbackStatus.textContent =
        `${candidate.external_id} quarantined; no future queue will sample it.`;
    }
    await loadCandidate(state.index + 1, true);
  } catch (error) {
    elements.playbackStatus.textContent =
      error instanceof Error ? error.message : "Could not save the judgment.";
    elements.playbackStatus.classList.add("warning");
    setDecisionDisabled(false);
  } finally {
    state.judging = false;
  }
}

function renderCandidate() {
  const candidate = currentCandidate();
  const preview = state.preview;
  if (!candidate || !preview) {
    return;
  }
  const interaction = candidate.interaction;
  elements.candidateReason.textContent =
    `${interaction.alternating_speaker_boundaries} alternating boundaries, ` +
    `${interaction.backchannel_count} backchannels, ${interaction.interruption_count} interruptions; ` +
    `${formatDuration(candidate.seconds_from_recording_end)} before recording end.`;
  elements.speaker1Events.textContent = annotationSummary(preview.speaker1);
  elements.speaker2Events.textContent = annotationSummary(preview.speaker2);
  elements.candidateDetails.replaceChildren(
    ...detailRows([
      ["Interaction score", interaction.interaction_score.toFixed(1)],
      ["Alternating / rapid boundaries", `${interaction.alternating_speaker_boundaries} / ${interaction.rapid_speaker_boundaries}`],
      ["Turn / backchannel / interruption events", `${interaction.turn_count} / ${interaction.backchannel_count} / ${interaction.interruption_count}`],
      ["Annotated simultaneous speech", `${interaction.both_speakers_active_seconds.toFixed(2)} s`],
      ["Synchronization suspicion", formatPercent(candidate.suspicion_score)],
      ["Audit anomaly hint", nullablePercent(candidate.audit_anomaly_score)],
      ["Late audit shift hint", nullableSeconds(candidate.audit_late_shift_seconds)],
      ["Absolute WAV duration gap", nullableAbsoluteSeconds(candidate.duration_mismatch_seconds)],
      ["Sampling weight", candidate.sampling_weight.toFixed(2)],
      ["Annotation version", preview.annotation_version],
    ]),
  );
  drawTimelines();
}

function drawTimelines() {
  if (!state.preview) {
    return;
  }
  drawTrackTimeline(
    elements.speaker1Timeline,
    state.preview.speaker1_waveform,
    state.preview.speaker1,
  );
  drawTrackTimeline(
    elements.speaker2Timeline,
    state.preview.speaker2_waveform,
    state.preview.speaker2,
  );
}

function drawTrackTimeline(canvas, waveform, annotation) {
  const width = Math.max(360, Math.floor(canvas.getBoundingClientRect().width));
  if (canvas.width !== width) {
    canvas.width = width;
  }
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f5f7f8";
  context.fillRect(0, 4, width, 65);
  context.strokeStyle = "#385c72";
  context.lineWidth = 1;
  const centerY = 36;
  const pointWidth = width / Math.max(1, waveform.length);
  waveform.forEach((point, index) => {
    const x = index * pointWidth;
    context.beginPath();
    context.moveTo(x, centerY - point.maximum_amplitude * 29);
    context.lineTo(x, centerY - point.minimum_amplitude * 29);
    context.stroke();
  });
  const candidate = currentCandidate();
  drawAnnotationTimelineRow({
    context,
    annotation,
    left: 0,
    top: 79,
    width,
    viewportStartSeconds: candidate.window_start_seconds,
    viewportEndSeconds: candidate.window_end_seconds,
  });
  const progressX = (playbackSeconds() / clipDuration()) * width;
  context.strokeStyle = "#111827";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(progressX, 1);
  context.lineTo(progressX, 130);
  context.stroke();
}

function setDecisionDisabled(disabled) {
  elements.aligned.disabled = disabled;
  elements.misaligned.disabled = disabled;
  elements.unsure.disabled = disabled;
}

function renderProgress() {
  if (!state.progress) {
    return;
  }
  elements.reviewedCount.textContent = String(state.progress.reviewed_snippet_count);
  elements.alignedCount.textContent = String(state.progress.plausibly_aligned_count);
  elements.quarantinedCount.textContent = String(state.progress.quarantined_session_count);
  elements.unsureCount.textContent = String(state.progress.unsure_count);
}

function showEmpty(message) {
  stopPlayback();
  elements.review.hidden = true;
  elements.emptyState.hidden = false;
  elements.emptyState.textContent = message;
  elements.queuePosition.textContent = "Queue complete";
}

function currentCandidate() {
  return state.queue[state.index];
}

function clipDuration(candidate = currentCandidate()) {
  return candidate ? candidate.window_end_seconds - candidate.window_start_seconds : 20;
}

function annotationSummary(annotation) {
  return `${annotation.segment_targets.length} segments | ${annotation.turns.length} EOT | ${annotation.backchannels.length} backchannels`;
}

function detailRows(rows) {
  return rows.flatMap(([term, description]) => {
    const termElement = document.createElement("dt");
    termElement.textContent = term;
    const descriptionElement = document.createElement("dd");
    descriptionElement.textContent = description;
    return [termElement, descriptionElement];
  });
}

function formatAbsoluteTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  return `${minutes}:${remaining.toFixed(1).padStart(4, "0")}`;
}

function formatClipTime(seconds) {
  return `0:${seconds.toFixed(1).padStart(4, "0")}`;
}

function formatDuration(seconds) {
  return seconds < 60 ? `${seconds.toFixed(0)} s` : `${(seconds / 60).toFixed(1)} min`;
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function nullablePercent(value) {
  return value === null ? "No audit report" : formatPercent(value);
}

function nullableSeconds(value) {
  return value === null ? "No clear hint" : `${value >= 0 ? "+" : ""}${value.toFixed(2)} s`;
}

function nullableAbsoluteSeconds(value) {
  return value === null ? "Unavailable" : `${value.toFixed(2)} s`;
}
