import { drawAnnotationTimelineRow } from "/pages/shared/annotation-timeline.js";

const QUEUE_SIZE = 50;
const QUEUE_SEED = "misalignment-lab-v1";
const initialMode =
  new URLSearchParams(window.location.search).get("mode") === "repairs" ? "repairs" : "triage";
const state = {
  mode: initialMode,
  queue: [],
  index: 0,
  preview: null,
  progress: null,
  audioContext: null,
  buffers: {
    speaker1: null,
    speaker2Original: null,
    speaker2Predicted: null,
  },
  sources: [],
  usePredictedShift: false,
  playbackStartedAt: null,
  playbackOffsetSeconds: 0,
  animationFrame: null,
  judging: false,
  transition: {
    preview: null,
    markerSeconds: null,
    buffers: {
      speaker1: null,
      speaker2Raw: null,
      speaker2First: null,
      speaker2Second: null,
    },
    sources: [],
    alignmentMode: "piecewise",
    playbackStartedAt: null,
    playbackOffsetSeconds: 0,
    animationFrame: null,
  },
};

const elements = {
  queuePosition: document.querySelector("#queue-position"),
  reviewedCount: document.querySelector("#reviewed-count"),
  alignedCount: document.querySelector("#aligned-count"),
  quarantinedCount: document.querySelector("#quarantined-count"),
  unsureCount: document.querySelector("#unsure-count"),
  triageMode: document.querySelector("#triage-mode"),
  repairMode: document.querySelector("#repair-mode"),
  emptyState: document.querySelector("#empty-state"),
  review: document.querySelector("#review"),
  sampleName: document.querySelector("#sample-name"),
  candidateTime: document.querySelector("#candidate-time"),
  candidateReason: document.querySelector("#candidate-reason"),
  reviewCategory: document.querySelector("#review-category"),
  reviewCategorySummary: document.querySelector("#review-category-summary"),
  shiftBadge: document.querySelector("#shift-badge"),
  candidateDetails: document.querySelector("#candidate-details"),
  comparisonControls: document.querySelector("#comparison-controls"),
  repairEvidence: document.querySelector("#repair-evidence"),
  originalShift: document.querySelector("#original-shift"),
  predictedShift: document.querySelector("#predicted-shift"),
  predictedShiftLabel: document.querySelector("#predicted-shift-label"),
  recommendedFixes: document.querySelector("#recommended-fixes"),
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
  decisionTitle: document.querySelector("#decision-title"),
  decisionDescription: document.querySelector("#decision-description"),
  triageDecisions: document.querySelector("#triage-decisions"),
  repairDecisions: document.querySelector("#repair-decisions"),
  repairPlausible: document.querySelector("#repair-plausible"),
  repairRejected: document.querySelector("#repair-rejected"),
  repairUnsure: document.querySelector("#repair-unsure"),
  transitionReview: document.querySelector("#transition-review"),
  transitionSummary: document.querySelector("#transition-summary"),
  transitionMarkerTime: document.querySelector("#transition-marker-time"),
  transitionMarker: document.querySelector("#transition-marker"),
  transitionRangeStart: document.querySelector("#transition-range-start"),
  transitionRangeEnd: document.querySelector("#transition-range-end"),
  transitionEarlierWindow: document.querySelector("#transition-earlier-window"),
  transitionLaterWindow: document.querySelector("#transition-later-window"),
  transitionWindowLabel: document.querySelector("#transition-window-label"),
  transitionRaw: document.querySelector("#transition-raw"),
  transitionEarly: document.querySelector("#transition-early"),
  transitionLate: document.querySelector("#transition-late"),
  transitionRepaired: document.querySelector("#transition-repaired"),
  transitionPlay: document.querySelector("#transition-play"),
  transitionSeek: document.querySelector("#transition-seek"),
  transitionClock: document.querySelector("#transition-clock"),
  transitionStatus: document.querySelector("#transition-status"),
  transitionSpeaker1Timeline: document.querySelector("#transition-speaker1-timeline"),
  transitionSpeaker2Timeline: document.querySelector("#transition-speaker2-timeline"),
  transitionConfirm: document.querySelector("#transition-confirm"),
  transitionReject: document.querySelector("#transition-reject"),
  transitionCancel: document.querySelector("#transition-cancel"),
  decisionPanel: document.querySelector(".decision-panel"),
};

elements.triageMode.addEventListener("click", () => void switchMode("triage"));
elements.repairMode.addEventListener("click", () => void switchMode("repairs"));
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
elements.originalShift.addEventListener("click", () => setComparisonShift(false));
elements.predictedShift.addEventListener("click", () => setComparisonShift(true));
elements.recommendedFixes.addEventListener("click", () => void submitRecommendedFix());
elements.repairPlausible.addEventListener("click", () => void submitRepairJudgment("plausible"));
elements.repairRejected.addEventListener("click", () =>
  void submitRepairJudgment("not_plausible"),
);
elements.repairUnsure.addEventListener("click", () => void submitRepairJudgment("unsure"));
elements.transitionRaw.addEventListener("click", () => setTransitionAlignmentMode("raw"));
elements.transitionEarly.addEventListener("click", () => setTransitionAlignmentMode("early"));
elements.transitionLate.addEventListener("click", () => setTransitionAlignmentMode("late"));
elements.transitionRepaired.addEventListener("click", () =>
  setTransitionAlignmentMode("piecewise"),
);
elements.transitionEarlierWindow.addEventListener("click", () =>
  void shiftTransitionWindow(-60),
);
elements.transitionLaterWindow.addEventListener("click", () =>
  void shiftTransitionWindow(60),
);
elements.transitionPlay.addEventListener("click", () => void startTransitionPlayback(true));
elements.transitionSeek.addEventListener("input", () => {
  state.transition.playbackOffsetSeconds =
    (Number(elements.transitionSeek.value) / 1000) * transitionDuration();
  stopTransitionPlayback();
  updateTransitionPlaybackDisplay();
});
elements.transitionSeek.addEventListener("change", () => void startTransitionPlayback(true));
elements.transitionMarker.addEventListener("input", () => {
  state.transition.markerSeconds = Number(elements.transitionMarker.value);
  stopTransitionPlayback();
  renderTransitionMarker();
  drawTransitionTimelines();
});
elements.transitionMarker.addEventListener("change", () => void recenterTransitionPreview());
elements.transitionConfirm.addEventListener("click", () =>
  void submitPiecewiseTransition("plausible"),
);
elements.transitionReject.addEventListener("click", () =>
  void submitPiecewiseTransition("not_plausible"),
);
elements.transitionCancel.addEventListener("click", closeTransitionReview);
for (const canvas of [elements.speaker1Timeline, elements.speaker2Timeline]) {
  canvas.addEventListener("click", (event) => {
    const bounds = canvas.getBoundingClientRect();
    state.playbackOffsetSeconds =
      Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)) * clipDuration();
    void startPlayback(true);
  });
}
for (const canvas of [
  elements.transitionSpeaker1Timeline,
  elements.transitionSpeaker2Timeline,
]) {
  canvas.addEventListener("click", (event) => {
    const bounds = canvas.getBoundingClientRect();
    state.transition.playbackOffsetSeconds =
      Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)) *
      transitionDuration();
    void startTransitionPlayback(true);
  });
}
document.addEventListener("keydown", (event) => {
  if (event.target !== document.body || state.judging) {
    return;
  }
  if (!elements.transitionReview.hidden) {
    if (event.key === " ") {
      event.preventDefault();
      void startTransitionPlayback(true);
    }
    return;
  }
  if (event.key.toLowerCase() === "a") {
    if (state.mode === "triage") {
      void submitJudgment("plausibly_aligned");
    }
  } else if (event.key.toLowerCase() === "m") {
    if (state.mode === "triage") {
      void submitJudgment("likely_misaligned");
    }
  } else if (event.key.toLowerCase() === "u") {
    if (state.mode === "triage") {
      void submitJudgment("unsure");
    } else {
      void submitRepairJudgment("unsure");
    }
  } else if (event.key.toLowerCase() === "f") {
    if (state.mode === "repairs") {
      void submitRepairJudgment("plausible");
    } else if (currentOffsetRecommendation()) {
      void submitRecommendedFix();
    }
  } else if (event.key.toLowerCase() === "n" && state.mode === "repairs") {
    void submitRepairJudgment("not_plausible");
  } else if (event.key.toLowerCase() === "o" && currentOffsetRecommendation()) {
    setComparisonShift(false);
  } else if (event.key.toLowerCase() === "p" && currentOffsetRecommendation()) {
    setComparisonShift(true);
  } else if (event.key === " ") {
    event.preventDefault();
    void startPlayback(true);
  }
});
window.addEventListener("resize", () => {
  drawTimelines();
  drawTransitionTimelines();
});

void loadQueue();

async function loadQueue() {
  try {
    renderMode();
    showLoading(
      state.mode === "repairs"
        ? "Loading conservative repair candidates..."
        : "Loading high-value alignment checks...",
    );
    const queueUrl =
      state.mode === "repairs"
        ? "/api/misalignment-lab/repair-queue"
        : `/api/misalignment-lab/queue?${new URLSearchParams({
            seed: QUEUE_SEED,
            limit: String(QUEUE_SIZE),
          }).toString()}`;
    const response = await fetch(queueUrl);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Queue request failed (${response.status})`);
    }
    state.queue = payload.candidates;
    state.progress = payload.progress;
    renderProgress();
    if (state.queue.length === 0) {
      showEmpty(
        state.mode === "repairs"
          ? "No quarantined sessions have a conservative piecewise repair estimate."
          : "No unreviewed, non-quarantined annotated sessions remain.",
      );
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
  state.usePredictedShift = false;
  state.buffers = {
    speaker1: null,
    speaker2Original: null,
    speaker2Predicted: null,
  };
  stopPlayback();
  resetTransitionReview();
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
    const originalAudioRequests = ["speaker1", "speaker2"].map((side) => {
      const query = new URLSearchParams({
        start_seconds: String(candidate.window_start_seconds),
        duration_seconds: String(clipDuration(candidate)),
      });
      return fetch(
        `/api/synchronization-review/audio-window/${candidate.sample_id}/${side}?${query.toString()}`,
      );
    });
    const offsetRecommendation = currentOffsetRecommendation();
    const predictedAudioRequest =
      offsetRecommendation
        ? fetch(
            `/api/synchronization-review/audio-window/${candidate.sample_id}/speaker2?${new URLSearchParams(
              {
                start_seconds: String(
                  candidate.window_start_seconds -
                    offsetRecommendation.shift_seconds,
                ),
                duration_seconds: String(clipDuration(candidate)),
              },
            ).toString()}`,
          )
        : null;
    const [previewResponse, ...audioResponses] = await Promise.all([
      previewRequest,
      ...originalAudioRequests,
      ...(predictedAudioRequest ? [predictedAudioRequest] : []),
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
  const decoded = await Promise.all(
    waveBuffers.map((waveBuffer) => audioContext.decodeAudioData(waveBuffer)),
  );
  state.buffers = {
    speaker1: decoded[0] ?? null,
    speaker2Original: decoded[1] ?? null,
    speaker2Predicted: decoded[2] ?? null,
  };
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
  const speaker2Buffer =
    state.usePredictedShift
      ? state.buffers.speaker2Predicted
      : state.buffers.speaker2Original;
  if (!state.buffers.speaker1 || !speaker2Buffer) {
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
  state.sources = [state.buffers.speaker1, speaker2Buffer].map((buffer) => {
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start(scheduledTime, state.playbackOffsetSeconds);
    return source;
  });
  state.playbackStartedAt = scheduledTime - state.playbackOffsetSeconds;
  elements.playbackStatus.textContent =
    state.usePredictedShift
      ? `Playing speaker 2 with the predicted ${formatSignedSeconds(
          currentOffsetRecommendation().shift_seconds,
        )} shift. No files are modified.`
      : "Playing both raw tracks on one AudioContext clock with exactly zero applied shift.";
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

async function submitRepairJudgment(judgment) {
  const candidate = currentCandidate();
  const repairEstimate = currentRepairEstimate();
  if (!candidate || !repairEstimate || state.judging) {
    return;
  }
  if (judgment === "plausible") {
    await openTransitionReview();
    return;
  }
  state.judging = true;
  stopPlayback();
  setDecisionDisabled(true);
  try {
    const response = await fetch("/api/misalignment-lab/repair-judgments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_id: candidate.sample_id,
        candidate_id: candidate.candidate_id,
        predicted_shift_seconds: repairEstimate.predicted_second_part_shift_seconds,
        estimator_version: repairEstimate.estimator_version,
        repair_scope: "after_change_point",
        first_part_shift_seconds: repairEstimate.first_part_shift_seconds,
        change_point_seconds: null,
        change_interval_start_seconds: repairEstimate.conservative_first_part_end_seconds,
        change_interval_end_seconds: repairEstimate.conservative_second_part_start_seconds,
        transition_confirmed: false,
        judgment,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Repair judgment failed (${response.status})`);
    }
    currentQueueItem().stored_judgment = payload.stored;
    state.progress = payload.progress;
    renderProgress();
    await loadCandidate(state.index + 1, true);
  } catch (error) {
    elements.playbackStatus.textContent =
      error instanceof Error ? error.message : "Could not save the repair judgment.";
    elements.playbackStatus.classList.add("warning");
    setDecisionDisabled(false);
  } finally {
    state.judging = false;
  }
}

async function submitRecommendedFix() {
  const candidate = currentCandidate();
  const recommendation = currentOffsetRecommendation();
  if (!candidate || !recommendation || state.mode !== "triage" || state.judging) {
    return;
  }
  if (recommendation.repair_scope === "after_change_point") {
    await openTransitionReview();
    return;
  }
  state.judging = true;
  stopPlayback();
  setDecisionDisabled(true);
  try {
    const judgmentResponse = await fetch("/api/misalignment-lab/judgments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate_id: candidate.candidate_id,
        sample_id: candidate.sample_id,
        window_start_seconds: candidate.window_start_seconds,
        window_end_seconds: candidate.window_end_seconds,
        judgment: "likely_misaligned",
        queue_seed: QUEUE_SEED,
      }),
    });
    const judgmentPayload = await judgmentResponse.json();
    if (!judgmentResponse.ok) {
      throw new Error(
        judgmentPayload.detail || `Judgment request failed (${judgmentResponse.status})`,
      );
    }
    const repairResponse = await fetch("/api/misalignment-lab/repair-judgments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_id: candidate.sample_id,
        candidate_id: candidate.candidate_id,
        predicted_shift_seconds: recommendation.shift_seconds,
        estimator_version: recommendation.estimator_version,
        repair_scope: "global_offset",
        first_part_shift_seconds: null,
        change_point_seconds: null,
        change_interval_start_seconds: null,
        change_interval_end_seconds: null,
        transition_confirmed: true,
        judgment: "plausible",
      }),
    });
    const repairPayload = await repairResponse.json();
    if (!repairResponse.ok) {
      throw new Error(
        repairPayload.detail || `Offset assessment failed (${repairResponse.status})`,
      );
    }
    state.progress = judgmentPayload.progress;
    renderProgress();
    await loadCandidate(state.index + 1, true);
  } catch (error) {
    elements.playbackStatus.textContent =
      error instanceof Error ? error.message : "Could not save the offset assessment.";
    elements.playbackStatus.classList.add("warning");
    setDecisionDisabled(false);
  } finally {
    state.judging = false;
  }
}

async function openTransitionReview() {
  const repairEstimate = currentRepairEstimate();
  if (!repairEstimate || state.judging) {
    return;
  }
  state.judging = true;
  stopPlayback();
  setDecisionDisabled(true);
  elements.transitionReview.hidden = false;
  elements.decisionPanel.hidden = true;
  elements.transitionStatus.textContent = "Loading the two-minute transition window...";
  elements.transitionReview.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    await ensureAudioContext(true);
    await loadTransitionPreview(null, true);
  } catch (error) {
    elements.transitionStatus.textContent =
      error instanceof Error ? error.message : "Could not load the transition review.";
    elements.transitionStatus.classList.add("warning");
  } finally {
    state.judging = false;
    setDecisionDisabled(false);
  }
}

async function loadTransitionPreview(centerSeconds, autoplay) {
  const candidate = currentCandidate();
  if (!candidate) {
    return;
  }
  stopTransitionPlayback();
  const query = new URLSearchParams();
  if (centerSeconds !== null) {
    query.set("center_seconds", String(centerSeconds));
  }
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  const previewResponse = await fetch(
    `/api/misalignment-lab/transition-preview/${candidate.sample_id}/${candidate.candidate_id}${suffix}`,
  );
  const preview = await previewResponse.json();
  if (!previewResponse.ok) {
    throw new Error(
      preview.detail || `Transition preview request failed (${previewResponse.status})`,
    );
  }
  const durationSeconds = preview.window_end_seconds - preview.window_start_seconds;
  const audioStarts = [
    preview.window_start_seconds,
    preview.window_start_seconds,
    preview.window_start_seconds - preview.first_part_shift_seconds,
    preview.window_start_seconds - preview.second_part_shift_seconds,
  ];
  const audioSides = ["speaker1", "speaker2", "speaker2", "speaker2"];
  const audioResponses = await Promise.all(
    audioStarts.map((startSeconds, index) =>
      fetch(
        `/api/synchronization-review/audio-window/${candidate.sample_id}/${audioSides[index]}?${new URLSearchParams(
          {
            start_seconds: String(startSeconds),
            duration_seconds: String(durationSeconds),
            sample_rate: "16000",
          },
        ).toString()}`,
      ),
    ),
  );
  for (const response of audioResponses) {
    if (!response.ok) {
      throw new Error(`Transition audio request failed (${response.status})`);
    }
  }
  const audioContext = await ensureAudioContext(false);
  const encodedAudio = await Promise.all(audioResponses.map((response) => response.arrayBuffer()));
  const decodedAudio = await Promise.all(
    encodedAudio.map((audio) => audioContext.decodeAudioData(audio)),
  );
  state.transition.preview = preview;
  state.transition.markerSeconds ??=
    currentQueueItem()?.stored_judgment?.change_point_seconds ??
    preview.estimated_change_point_seconds;
  state.transition.buffers = {
    speaker1: decodedAudio[0] ?? null,
    speaker2Raw: decodedAudio[1] ?? null,
    speaker2First: decodedAudio[2] ?? null,
    speaker2Second: decodedAudio[3] ?? null,
  };
  state.transition.playbackOffsetSeconds = 0;
  renderTransitionReview();
  if (autoplay) {
    await startTransitionPlayback(false);
  }
}

async function recenterTransitionPreview() {
  const preview = state.transition.preview;
  const markerSeconds = state.transition.markerSeconds;
  if (!preview || markerSeconds === null) {
    return;
  }
  if (
    markerSeconds >= preview.window_start_seconds + 10 &&
    markerSeconds <= preview.window_end_seconds - 10
  ) {
    return;
  }
  elements.transitionStatus.textContent = "Centering the replay on the selected timestamp...";
  try {
    await loadTransitionPreview(markerSeconds, false);
  } catch (error) {
    elements.transitionStatus.textContent =
      error instanceof Error ? error.message : "Could not recenter the transition replay.";
    elements.transitionStatus.classList.add("warning");
  }
}

async function shiftTransitionWindow(deltaSeconds) {
  const preview = state.transition.preview;
  if (!preview) {
    return;
  }
  const currentCenter = (preview.window_start_seconds + preview.window_end_seconds) / 2;
  const targetCenter = Math.min(
    preview.search_end_seconds,
    Math.max(preview.search_start_seconds, currentCenter + deltaSeconds),
  );
  if (Math.abs(targetCenter - currentCenter) < 0.01) {
    return;
  }
  elements.transitionStatus.textContent = "Loading the adjacent transition window...";
  try {
    await loadTransitionPreview(targetCenter, false);
  } catch (error) {
    elements.transitionStatus.textContent =
      error instanceof Error ? error.message : "Could not load the adjacent transition window.";
    elements.transitionStatus.classList.add("warning");
  }
}

async function submitPiecewiseTransition(judgment) {
  const candidate = currentCandidate();
  const repairEstimate = currentRepairEstimate();
  const markerSeconds = state.transition.markerSeconds;
  if (!candidate || !repairEstimate || markerSeconds === null || state.judging) {
    return;
  }
  state.judging = true;
  stopTransitionPlayback();
  setDecisionDisabled(true);
  try {
    let triageProgress = null;
    if (state.mode === "triage") {
      const judgmentResponse = await fetch("/api/misalignment-lab/judgments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          candidate_id: candidate.candidate_id,
          sample_id: candidate.sample_id,
          window_start_seconds: candidate.window_start_seconds,
          window_end_seconds: candidate.window_end_seconds,
          judgment: "likely_misaligned",
          queue_seed: QUEUE_SEED,
        }),
      });
      const judgmentPayload = await judgmentResponse.json();
      if (!judgmentResponse.ok) {
        throw new Error(
          judgmentPayload.detail || `Judgment request failed (${judgmentResponse.status})`,
        );
      }
      triageProgress = judgmentPayload.progress;
    }
    const transitionConfirmed = judgment === "plausible";
    const repairResponse = await fetch("/api/misalignment-lab/repair-judgments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_id: candidate.sample_id,
        candidate_id: candidate.candidate_id,
        predicted_shift_seconds: repairEstimate.predicted_second_part_shift_seconds,
        estimator_version: repairEstimate.estimator_version,
        repair_scope: "after_change_point",
        first_part_shift_seconds: repairEstimate.first_part_shift_seconds,
        change_point_seconds: transitionConfirmed ? markerSeconds : null,
        change_interval_start_seconds: repairEstimate.conservative_first_part_end_seconds,
        change_interval_end_seconds: repairEstimate.conservative_second_part_start_seconds,
        transition_confirmed: transitionConfirmed,
        judgment,
      }),
    });
    const repairPayload = await repairResponse.json();
    if (!repairResponse.ok) {
      throw new Error(
        repairPayload.detail || `Transition assessment failed (${repairResponse.status})`,
      );
    }
    state.progress = triageProgress ?? repairPayload.progress;
    renderProgress();
    closeTransitionReview();
    await loadCandidate(state.index + 1, true);
  } catch (error) {
    elements.transitionStatus.textContent =
      error instanceof Error ? error.message : "Could not save the transition assessment.";
    elements.transitionStatus.classList.add("warning");
    setDecisionDisabled(false);
  } finally {
    state.judging = false;
  }
}

function renderTransitionReview() {
  const preview = state.transition.preview;
  const markerSeconds = state.transition.markerSeconds;
  if (!preview || markerSeconds === null) {
    return;
  }
  elements.transitionSummary.textContent =
    `The beginning supports ${formatSignedSeconds(preview.first_part_shift_seconds)} and the end ` +
    `supports ${formatSignedSeconds(preview.second_part_shift_seconds)}. The orange band ` +
    `${formatAbsoluteTime(preview.change_interval_start_seconds)}-${formatAbsoluteTime(
      preview.change_interval_end_seconds,
    )} is an overlapping-window midpoint hint, not a hard boundary. Search the full ` +
    `${formatAbsoluteTime(preview.search_start_seconds)}-${formatAbsoluteTime(
      preview.search_end_seconds,
    )} range.`;
  elements.transitionMarker.min = String(preview.search_start_seconds);
  elements.transitionMarker.max = String(preview.search_end_seconds);
  elements.transitionMarker.step =
    preview.search_end_seconds - preview.search_start_seconds < 2 ? "0.1" : "1";
  elements.transitionMarker.value = String(markerSeconds);
  elements.transitionRangeStart.textContent =
    `Conservative search start ${formatAbsoluteTime(
      preview.search_start_seconds,
    )}`;
  elements.transitionRangeEnd.textContent =
    `Conservative search end ${formatAbsoluteTime(preview.search_end_seconds)}`;
  elements.transitionWindowLabel.textContent =
    `Viewing ${formatAbsoluteTime(preview.window_start_seconds)}-${formatAbsoluteTime(
      preview.window_end_seconds,
    )}`;
  const currentCenter = (preview.window_start_seconds + preview.window_end_seconds) / 2;
  elements.transitionEarlierWindow.disabled = currentCenter <= preview.search_start_seconds + 0.01;
  elements.transitionLaterWindow.disabled = currentCenter >= preview.search_end_seconds - 0.01;
  elements.transitionRaw.setAttribute(
    "aria-pressed",
    String(state.transition.alignmentMode === "raw"),
  );
  elements.transitionEarly.setAttribute(
    "aria-pressed",
    String(state.transition.alignmentMode === "early"),
  );
  elements.transitionLate.setAttribute(
    "aria-pressed",
    String(state.transition.alignmentMode === "late"),
  );
  elements.transitionRepaired.setAttribute(
    "aria-pressed",
    String(state.transition.alignmentMode === "piecewise"),
  );
  elements.transitionStatus.textContent = transitionReadyStatus();
  elements.transitionStatus.classList.remove("warning");
  renderTransitionMarker();
  updateTransitionPlaybackDisplay();
}

function renderTransitionMarker() {
  if (state.transition.markerSeconds === null) {
    return;
  }
  elements.transitionMarkerTime.textContent =
    `Selected ${formatAbsoluteTime(state.transition.markerSeconds)}`;
}

function setTransitionAlignmentMode(alignmentMode) {
  stopTransitionPlayback();
  state.transition.alignmentMode = alignmentMode;
  renderTransitionReview();
}

function transitionReadyStatus() {
  switch (state.transition.alignmentMode) {
    case "raw":
      return "Ready: both tracks use their original timestamps throughout the window.";
    case "early":
      return "Ready: the early alignment shift is applied throughout the window.";
    case "late":
      return "Ready: the recommended late shift is applied throughout the window.";
    case "piecewise":
      return "Ready: early alignment before the marker, recommended late shift after it.";
  }
}

async function startTransitionPlayback(fromUserGesture) {
  const preview = state.transition.preview;
  const markerSeconds = state.transition.markerSeconds;
  const buffers = state.transition.buffers;
  const selectedSpeaker2Buffer = transitionSpeaker2Buffer();
  if (
    !preview ||
    markerSeconds === null ||
    !buffers.speaker1 ||
    (state.transition.alignmentMode !== "piecewise" && !selectedSpeaker2Buffer) ||
    (state.transition.alignmentMode === "piecewise" &&
      (!buffers.speaker2First || !buffers.speaker2Second))
  ) {
    return;
  }
  const audioContext = await ensureAudioContext(fromUserGesture);
  if (audioContext.state !== "running") {
    elements.transitionStatus.textContent = "Press Replay transition once to enable audio.";
    elements.transitionStatus.classList.add("warning");
    return;
  }
  stopPlayback();
  stopTransitionPlayback();
  if (state.transition.playbackOffsetSeconds >= transitionDuration() - 0.02) {
    state.transition.playbackOffsetSeconds = 0;
  }
  const scheduledTime = audioContext.currentTime + 0.05;
  state.transition.sources.push(
    scheduleAudioBuffer(
      buffers.speaker1,
      scheduledTime,
      state.transition.playbackOffsetSeconds,
      null,
    ),
  );
  if (state.transition.alignmentMode !== "piecewise") {
    state.transition.sources.push(
      scheduleAudioBuffer(
        selectedSpeaker2Buffer,
        scheduledTime,
        state.transition.playbackOffsetSeconds,
        null,
      ),
    );
  } else {
    const changeOffsetSeconds = markerSeconds - preview.window_start_seconds;
    if (state.transition.playbackOffsetSeconds < changeOffsetSeconds) {
      const firstDurationSeconds = changeOffsetSeconds - state.transition.playbackOffsetSeconds;
      state.transition.sources.push(
        scheduleAudioBuffer(
          buffers.speaker2First,
          scheduledTime,
          state.transition.playbackOffsetSeconds,
          firstDurationSeconds,
        ),
        scheduleAudioBuffer(
          buffers.speaker2Second,
          scheduledTime + firstDurationSeconds,
          changeOffsetSeconds,
          null,
        ),
      );
    } else {
      state.transition.sources.push(
        scheduleAudioBuffer(
          buffers.speaker2Second,
          scheduledTime,
          state.transition.playbackOffsetSeconds,
          null,
        ),
      );
    }
  }
  state.transition.playbackStartedAt =
    scheduledTime - state.transition.playbackOffsetSeconds;
  elements.transitionStatus.textContent = transitionPlaybackStatus(markerSeconds);
  elements.transitionStatus.classList.remove("warning");
  scheduleTransitionPlaybackUpdate();
}

function transitionSpeaker2Buffer() {
  switch (state.transition.alignmentMode) {
    case "raw":
      return state.transition.buffers.speaker2Raw;
    case "early":
      return state.transition.buffers.speaker2First;
    case "late":
      return state.transition.buffers.speaker2Second;
    case "piecewise":
      return null;
  }
}

function transitionPlaybackStatus(markerSeconds) {
  switch (state.transition.alignmentMode) {
    case "raw":
      return "Playing both original tracks with no reconstruction.";
    case "early":
      return "Playing the early alignment shift throughout the window.";
    case "late":
      return "Playing the recommended late shift throughout the window.";
    case "piecewise":
      return `Playing the piecewise timeline with the boundary at ${formatAbsoluteTime(markerSeconds)}.`;
  }
}

function scheduleAudioBuffer(buffer, startTime, offsetSeconds, durationSeconds) {
  const source = state.audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(state.audioContext.destination);
  if (durationSeconds === null) {
    source.start(startTime, offsetSeconds);
  } else {
    source.start(startTime, offsetSeconds, durationSeconds);
  }
  return source;
}

function stopTransitionPlayback() {
  for (const source of state.transition.sources) {
    try {
      source.stop();
    } catch {
      // A source can already have ended between animation frames.
    }
  }
  state.transition.sources = [];
  state.transition.playbackStartedAt = null;
  if (state.transition.animationFrame !== null) {
    cancelAnimationFrame(state.transition.animationFrame);
    state.transition.animationFrame = null;
  }
}

function scheduleTransitionPlaybackUpdate() {
  updateTransitionPlaybackDisplay();
  if (state.transition.playbackStartedAt === null || !state.audioContext) {
    return;
  }
  if (transitionPlaybackSeconds() >= transitionDuration()) {
    state.transition.playbackOffsetSeconds = transitionDuration();
    stopTransitionPlayback();
    updateTransitionPlaybackDisplay();
    return;
  }
  state.transition.animationFrame = requestAnimationFrame(scheduleTransitionPlaybackUpdate);
}

function transitionPlaybackSeconds() {
  if (state.transition.playbackStartedAt !== null && state.audioContext) {
    return Math.min(
      transitionDuration(),
      Math.max(0, state.audioContext.currentTime - state.transition.playbackStartedAt),
    );
  }
  return state.transition.playbackOffsetSeconds;
}

function updateTransitionPlaybackDisplay() {
  const preview = state.transition.preview;
  if (!preview) {
    return;
  }
  const seconds = transitionPlaybackSeconds();
  elements.transitionSeek.value = String(Math.round((seconds / transitionDuration()) * 1000));
  elements.transitionClock.textContent =
    `${formatAbsoluteTime(preview.window_start_seconds + seconds)} / ` +
    `${formatAbsoluteTime(preview.window_end_seconds)}`;
  drawTransitionTimelines();
}

function drawTransitionTimelines() {
  const preview = state.transition.preview;
  if (!preview) {
    return;
  }
  drawTransitionTrack(
    elements.transitionSpeaker1Timeline,
    preview.speaker1_waveform,
    preview.speaker1,
  );
  const speaker2Waveform = transitionSpeaker2Waveform(preview);
  const speaker2Annotation = transitionSpeaker2Annotation(preview);
  drawTransitionTrack(
    elements.transitionSpeaker2Timeline,
    speaker2Waveform,
    speaker2Annotation,
  );
}

function transitionSpeaker2Waveform(preview) {
  switch (state.transition.alignmentMode) {
    case "raw":
      return preview.speaker2_raw_waveform;
    case "early":
      return preview.speaker2_first_alignment_waveform;
    case "late":
      return preview.speaker2_second_alignment_waveform;
    case "piecewise":
      return piecewiseWaveform(
        preview.speaker2_first_alignment_waveform,
        preview.speaker2_second_alignment_waveform,
      );
  }
}

function transitionSpeaker2Annotation(preview) {
  switch (state.transition.alignmentMode) {
    case "raw":
      return preview.speaker2_raw;
    case "early":
      return preview.speaker2_first_alignment;
    case "late":
      return preview.speaker2_second_alignment;
    case "piecewise":
      return piecewiseAnnotation(
        preview.speaker2_first_alignment,
        preview.speaker2_second_alignment,
      );
  }
}

function drawTransitionTrack(canvas, waveform, annotation) {
  const preview = state.transition.preview;
  const markerSeconds = state.transition.markerSeconds;
  if (!preview || markerSeconds === null) {
    return;
  }
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
  const intervalStart = Math.max(
    preview.window_start_seconds,
    preview.change_interval_start_seconds,
  );
  const intervalEnd = Math.min(
    preview.window_end_seconds,
    preview.change_interval_end_seconds,
  );
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
  drawAnnotationTimelineRow({
    context,
    annotation,
    left: 0,
    top: 79,
    width,
    viewportStartSeconds: preview.window_start_seconds,
    viewportEndSeconds: preview.window_end_seconds,
  });
  if (intervalEnd > intervalStart) {
    const intervalX = transitionTimelineX(intervalStart, width);
    const intervalWidth = transitionTimelineX(intervalEnd, width) - intervalX;
    context.fillStyle = "rgba(224, 151, 25, 0.12)";
    context.fillRect(intervalX, 0, intervalWidth, 132);
  }
  if (
    markerSeconds >= preview.window_start_seconds &&
    markerSeconds <= preview.window_end_seconds
  ) {
    const selectedBoxStart = Math.max(preview.window_start_seconds, markerSeconds - 5);
    const selectedBoxEnd = Math.min(preview.window_end_seconds, markerSeconds + 5);
    const selectedBoxX = transitionTimelineX(selectedBoxStart, width);
    const selectedBoxWidth = transitionTimelineX(selectedBoxEnd, width) - selectedBoxX;
    context.fillStyle = "rgba(224, 151, 25, 0.24)";
    context.strokeStyle = "#d08700";
    context.lineWidth = 1;
    context.fillRect(selectedBoxX, 0, selectedBoxWidth, 132);
    context.strokeRect(selectedBoxX, 0, selectedBoxWidth, 132);
    context.strokeStyle = "#b85f00";
    context.lineWidth = 3;
    const markerX = transitionTimelineX(markerSeconds, width);
    context.beginPath();
    context.moveTo(markerX, 0);
    context.lineTo(markerX, 132);
    context.stroke();
  }
  const progressX = (transitionPlaybackSeconds() / transitionDuration()) * width;
  context.strokeStyle = "#111827";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(progressX, 1);
  context.lineTo(progressX, 130);
  context.stroke();
}

function transitionTimelineX(timeSeconds, width) {
  const preview = state.transition.preview;
  return (
    ((timeSeconds - preview.window_start_seconds) /
      (preview.window_end_seconds - preview.window_start_seconds)) *
    width
  );
}

function piecewiseWaveform(firstAlignment, secondAlignment) {
  const preview = state.transition.preview;
  const markerSeconds = state.transition.markerSeconds;
  const pointCount = Math.min(firstAlignment.length, secondAlignment.length);
  const markerRatio =
    (markerSeconds - preview.window_start_seconds) /
    (preview.window_end_seconds - preview.window_start_seconds);
  const markerIndex = Math.round(markerRatio * pointCount);
  return firstAlignment
    .slice(0, markerIndex)
    .concat(secondAlignment.slice(markerIndex, pointCount));
}

function piecewiseAnnotation(firstAlignment, secondAlignment) {
  const markerSeconds = state.transition.markerSeconds;
  return {
    side: firstAlignment.side,
    speech_segments: mergeSpans(
      firstAlignment.speech_segments,
      secondAlignment.speech_segments,
      markerSeconds,
    ),
    pauses: mergeSpans(firstAlignment.pauses, secondAlignment.pauses, markerSeconds),
    backchannels: mergeSpans(
      firstAlignment.backchannels,
      secondAlignment.backchannels,
      markerSeconds,
    ),
    turns: mergePoints(firstAlignment.turns, secondAlignment.turns, markerSeconds),
    interruptions: mergePoints(
      firstAlignment.interruptions,
      secondAlignment.interruptions,
      markerSeconds,
    ),
    segment_targets: mergeSpans(
      firstAlignment.segment_targets,
      secondAlignment.segment_targets,
      markerSeconds,
    ),
    connection_targets: [
      ...firstAlignment.connection_targets.filter(
        (target) => target.later_start_seconds <= markerSeconds,
      ),
      ...secondAlignment.connection_targets.filter(
        (target) => target.earlier_end_seconds >= markerSeconds,
      ),
    ],
  };
}

function mergeSpans(first, second, markerSeconds) {
  return [
    ...first
      .filter((span) => span.start_seconds < markerSeconds)
      .map((span) => ({ ...span, end_seconds: Math.min(span.end_seconds, markerSeconds) })),
    ...second
      .filter((span) => span.end_seconds >= markerSeconds)
      .map((span) => ({ ...span, start_seconds: Math.max(span.start_seconds, markerSeconds) })),
  ];
}

function mergePoints(first, second, markerSeconds) {
  return [
    ...first.filter((point) => point.time_seconds < markerSeconds),
    ...second.filter((point) => point.time_seconds >= markerSeconds),
  ];
}

function transitionDuration() {
  const preview = state.transition.preview;
  return preview ? preview.window_end_seconds - preview.window_start_seconds : 120;
}

function closeTransitionReview() {
  resetTransitionReview();
  elements.decisionPanel.hidden = false;
  elements.decisionPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function resetTransitionReview() {
  stopTransitionPlayback();
  elements.transitionReview.hidden = true;
  elements.decisionPanel.hidden = false;
  state.transition.preview = null;
  state.transition.markerSeconds = null;
  state.transition.buffers = {
    speaker1: null,
    speaker2Raw: null,
    speaker2First: null,
    speaker2Second: null,
  };
  state.transition.alignmentMode = "piecewise";
  state.transition.playbackOffsetSeconds = 0;
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
  elements.speaker2Events.textContent = annotationSummary(activeSpeaker2Annotation());
  renderReviewCategory();
  renderOffsetComparison();
  const repairEstimate = currentRepairEstimate();
  elements.repairPlausible.firstChild.textContent = repairEstimate
    ? "Shift sounds right - locate transition "
    : "Prediction sounds plausible ";
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
      ...(repairEstimate
        ? [
            [
              "First / second stable shift",
              `${formatSignedSeconds(repairEstimate.first_part_shift_seconds)} / ${formatSignedSeconds(
                repairEstimate.predicted_second_part_shift_seconds,
              )}`,
            ],
            [
              "Stable second-part evidence",
              `${formatDuration(repairEstimate.stable_second_part_duration_seconds)} across ${repairEstimate.supporting_window_count} windows`,
            ],
            [
              "Second-part shift spread",
              `${repairEstimate.shift_spread_seconds.toFixed(2)} s`,
            ],
            [
              "Estimated change interval",
              `${formatAbsoluteTime(repairEstimate.change_interval_start_seconds)}-${formatAbsoluteTime(
                repairEstimate.change_interval_end_seconds,
              )}`,
            ],
            [
              "Conservative retained ranges",
              `first part through ${formatAbsoluteTime(
                repairEstimate.conservative_first_part_end_seconds,
              )}; second part from ${formatAbsoluteTime(
                repairEstimate.conservative_second_part_start_seconds,
              )}`,
            ],
            ["Repair confidence", formatPercent(repairEstimate.confidence_score)],
          ]
        : []),
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
    activeSpeaker2Waveform(),
    activeSpeaker2Annotation(),
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
  elements.repairPlausible.disabled = disabled;
  elements.repairRejected.disabled = disabled;
  elements.repairUnsure.disabled = disabled;
  elements.recommendedFixes.disabled = disabled;
  elements.transitionConfirm.disabled = disabled;
  elements.transitionReject.disabled = disabled;
  elements.transitionCancel.disabled = disabled;
  elements.transitionRaw.disabled = disabled;
  elements.transitionEarly.disabled = disabled;
  elements.transitionLate.disabled = disabled;
  elements.transitionRepaired.disabled = disabled;
  const transitionPreview = state.transition.preview;
  const transitionCenter = transitionPreview
    ? (transitionPreview.window_start_seconds + transitionPreview.window_end_seconds) / 2
    : null;
  elements.transitionEarlierWindow.disabled =
    disabled ||
    transitionPreview === null ||
    transitionCenter <= transitionPreview.search_start_seconds + 0.01;
  elements.transitionLaterWindow.disabled =
    disabled ||
    transitionPreview === null ||
    transitionCenter >= transitionPreview.search_end_seconds - 0.01;
}

function renderProgress() {
  if (!state.progress) {
    return;
  }
  if (state.mode === "repairs") {
    elements.reviewedCount.textContent = String(state.progress.reviewed_repair_count);
    elements.alignedCount.textContent = String(state.progress.plausible_repair_count);
    elements.quarantinedCount.textContent =
      `${state.progress.repair_candidate_count} / ${state.progress.quarantined_session_count}`;
    elements.unsureCount.textContent = String(state.progress.unsure_repair_count);
    return;
  }
  elements.reviewedCount.textContent = String(state.progress.reviewed_snippet_count);
  elements.alignedCount.textContent = String(state.progress.plausibly_aligned_count);
  elements.quarantinedCount.textContent = String(state.progress.quarantined_session_count);
  elements.unsureCount.textContent = String(state.progress.unsure_count);
}

async function switchMode(mode) {
  if (state.mode === mode) {
    return;
  }
  state.mode = mode;
  state.queue = [];
  state.index = 0;
  state.preview = null;
  state.progress = null;
  stopPlayback();
  resetTransitionReview();
  const url = new URL(window.location.href);
  if (mode === "repairs") {
    url.searchParams.set("mode", "repairs");
  } else {
    url.searchParams.delete("mode");
  }
  window.history.replaceState(null, "", url);
  await loadQueue();
}

function renderMode() {
  const repairMode = state.mode === "repairs";
  elements.triageMode.setAttribute("aria-pressed", String(!repairMode));
  elements.repairMode.setAttribute("aria-pressed", String(repairMode));
  elements.comparisonControls.hidden = !repairMode;
  elements.triageDecisions.hidden = repairMode;
  elements.repairDecisions.hidden = !repairMode;
  elements.decisionTitle.textContent = repairMode
    ? "Does the predicted second-part shift plausibly repair this clip?"
    : "Do the two raw tracks plausibly share one timeline?";
  elements.decisionDescription.textContent = repairMode
    ? "Toggle repeatedly between the raw and predicted timelines. This only records your assessment; no audio is rewritten."
    : "Plausibly aligned accepts the recording; likely misaligned quarantines it. Unsure asks for a different exchange later. No audio is rewritten.";
}

function renderReviewCategory() {
  const candidate = currentCandidate();
  const labels = {
    likely_aligned: "1 - Very likely aligned",
    likely_constant_offset: "2 - Likely one-offset repair",
    non_constant_or_uncertain: "3 - Non-constant / no safe shift",
  };
  elements.reviewCategory.textContent = labels[candidate.review_category];
  elements.reviewCategory.className = candidate.review_category.replaceAll("_", "-");
  elements.reviewCategorySummary.textContent = candidate.review_category_summary;
}

function renderOffsetComparison() {
  const recommendation = currentOffsetRecommendation();
  if (!recommendation) {
    elements.comparisonControls.hidden = true;
    elements.shiftBadge.textContent = "Raw tracks - shift 0.0 s";
    return;
  }
  elements.comparisonControls.hidden = false;
  elements.recommendedFixes.hidden = state.mode === "repairs";
  elements.recommendedFixes.firstChild.textContent =
    recommendation.repair_scope === "after_change_point"
      ? "Shift sounds right - locate transition "
      : "Recommended shift fixes it ";
  elements.predictedShiftLabel.textContent = formatSignedSeconds(recommendation.shift_seconds);
  elements.repairEvidence.textContent = recommendation.summary;
  elements.originalShift.setAttribute("aria-pressed", String(!state.usePredictedShift));
  elements.predictedShift.setAttribute("aria-pressed", String(state.usePredictedShift));
  elements.shiftBadge.textContent = state.usePredictedShift
    ? `Recommended speaker 2 shift ${formatSignedSeconds(recommendation.shift_seconds)}`
    : "Original tracks - shift 0.0 s";
  const storedJudgment = currentQueueItem().stored_judgment;
  if (storedJudgment) {
    elements.decisionDescription.textContent =
      `Previously marked ${storedJudgment.judgment.replaceAll("_", " ")}. ` +
      "You can listen again and replace that assessment.";
  }
}

function setComparisonShift(usePredictedShift) {
  if (!currentOffsetRecommendation()) {
    return;
  }
  const wasPlaying = state.playbackStartedAt !== null;
  state.playbackOffsetSeconds = playbackSeconds();
  stopPlayback();
  state.usePredictedShift = usePredictedShift;
  renderCandidate();
  if (wasPlaying) {
    void startPlayback(true);
  } else {
    elements.playbackStatus.textContent = usePredictedShift
      ? `Ready to play speaker 2 with the recommended ${formatSignedSeconds(
          currentOffsetRecommendation().shift_seconds,
        )} shift.`
      : "Ready to play both original tracks with exactly zero applied shift.";
  }
}

function activeSpeaker2Waveform() {
  if (
    state.usePredictedShift &&
    state.preview.predicted_speaker2_waveform
  ) {
    return state.preview.predicted_speaker2_waveform;
  }
  return state.preview.speaker2_waveform;
}

function activeSpeaker2Annotation() {
  if (
    state.usePredictedShift &&
    state.preview.predicted_speaker2
  ) {
    return state.preview.predicted_speaker2;
  }
  return state.preview.speaker2;
}

function showEmpty(message) {
  stopPlayback();
  elements.review.hidden = true;
  elements.emptyState.hidden = false;
  elements.emptyState.textContent = message;
  elements.queuePosition.textContent = "Queue complete";
}

function showLoading(message) {
  stopPlayback();
  elements.review.hidden = true;
  elements.emptyState.hidden = false;
  elements.emptyState.textContent = message;
  elements.queuePosition.textContent = "Loading";
}

function currentCandidate() {
  const item = currentQueueItem();
  return item ? item.candidate ?? item : null;
}

function currentQueueItem() {
  return state.queue[state.index] ?? null;
}

function currentRepairEstimate() {
  const item = currentQueueItem();
  return item?.repair_estimate ?? state.preview?.repair_estimate ?? null;
}

function currentOffsetRecommendation() {
  const repairEstimate = currentRepairEstimate();
  if (repairEstimate) {
    return {
      shift_seconds: repairEstimate.predicted_second_part_shift_seconds,
      estimator_version: repairEstimate.estimator_version,
      repair_scope: "after_change_point",
      summary:
        `${repairEstimate.supporting_window_count} stable windows over ` +
        `${formatDuration(repairEstimate.stable_second_part_duration_seconds)}; ` +
        `${repairEstimate.shift_spread_seconds.toFixed(2)} s spread`,
    };
  }
  return currentCandidate()?.offset_recommendation ?? null;
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

function formatSignedSeconds(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)} s`;
}
