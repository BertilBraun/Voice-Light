import { drawAnnotationTimelineRow } from "/pages/shared/annotation-timeline.js";

const state = {
  candidates: [],
  index: 0,
  players: [],
  playing: false,
  seeking: false,
  nextOffset: 0,
  loadingMore: false,
};
const CANDIDATE_PAGE_SIZE = 50;

const elements = {
  position: document.querySelector("#position"),
  emptyState: document.querySelector("#empty-state"),
  review: document.querySelector("#review"),
  sampleName: document.querySelector("#sample-name"),
  candidateTitle: document.querySelector("#candidate-title"),
  previous: document.querySelector("#previous"),
  next: document.querySelector("#next"),
  play: document.querySelector("#play"),
  seek: document.querySelector("#seek"),
  clock: document.querySelector("#clock"),
  tracks: document.querySelector("#tracks"),
  timelineInspector: document.querySelector("#timeline-inspector"),
  beforeCard: document.querySelector("#before-card"),
  responseCard: document.querySelector("#response-card"),
  afterCard: document.querySelector("#after-card"),
  scores: document.querySelector("#scores"),
};

elements.previous.addEventListener("click", () => showCandidate(state.index - 1));
elements.next.addEventListener("click", () => void showNextCandidate(true));
elements.play.addEventListener("click", () => {
  if (state.playing) {
    pause();
  } else {
    void play();
  }
});
elements.seek.addEventListener("input", () => {
  state.seeking = true;
  seekToRatio(Number(elements.seek.value) / 1000);
});
elements.seek.addEventListener("change", () => {
  state.seeking = false;
  seekToRatio(Number(elements.seek.value) / 1000);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight") {
    void showNextCandidate(false);
  } else if (event.key === "ArrowLeft") {
    showCandidate(state.index - 1);
  } else if (event.key === " " && event.target === document.body) {
    event.preventDefault();
    elements.play.click();
  }
});
window.addEventListener("resize", drawTimelines);

void loadCandidates();

async function loadCandidates() {
  try {
    await loadNextCandidatePage();
    if (state.candidates.length === 0) {
      showEmpty("No merged connections with an intervening transcribed speaker were found.");
      return;
    }
    showCandidate(0);
  } catch (error) {
    showEmpty(error instanceof Error ? error.message : "Could not load candidates.");
  }
}

async function loadNextCandidatePage() {
  if (state.nextOffset === null || state.loadingMore) {
    return;
  }
  state.loadingMore = true;
  try {
    const query = new URLSearchParams({
      offset: String(state.nextOffset),
      limit: String(CANDIDATE_PAGE_SIZE),
    });
    const response = await fetch(`/api/backchannel-review/candidates?${query.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `Candidate request failed (${response.status})`);
    }
    state.candidates.push(...payload.candidates);
    state.nextOffset = payload.next_offset;
  } finally {
    state.loadingMore = false;
  }
}

async function showNextCandidate(autoplay) {
  const nextIndex = state.index + 1;
  if (nextIndex >= state.candidates.length && state.nextOffset !== null) {
    await loadNextCandidatePage();
  }
  showCandidate(nextIndex, autoplay);
}

function showEmpty(message) {
  destroyPlayers();
  elements.position.textContent = "0 candidates";
  elements.emptyState.textContent = message;
  elements.emptyState.hidden = false;
  elements.review.hidden = true;
}

function showCandidate(index, autoplay = false) {
  if (index < 0 || index >= state.candidates.length) {
    return;
  }
  state.index = index;
  const candidate = currentCandidate();
  pause();
  elements.emptyState.hidden = true;
  elements.review.hidden = false;
  const candidateCountSuffix = state.nextOffset === null ? "" : "+";
  elements.position.textContent =
    `${index + 1} / ${state.candidates.length}${candidateCountSuffix}`;
  elements.sampleName.textContent = candidate.external_id;
  elements.candidateTitle.textContent = `${speakerLabel(candidate.possible_backchannel_side)} inside ${speakerLabel(candidate.floor_holder_side)}'s pause`;
  elements.previous.disabled = index === 0;
  elements.next.disabled = index === state.candidates.length - 1;
  renderSequence(candidate);
  renderScores(candidate);
  createPlayers(candidate);
  renderSegmentInspection(
    candidate.possible_backchannel_side,
    candidate.possible_backchannel,
    "Highlighted candidate",
  );
  if (autoplay) {
    void play();
  }
}

function renderSequence(candidate) {
  renderTranscriptCard(
    elements.beforeCard,
    `${speakerLabel(candidate.floor_holder_side)} before`,
    candidate.floor_holder_before,
  );
  renderTranscriptCard(
    elements.responseCard,
    `${speakerLabel(candidate.possible_backchannel_side)} possible backchannel`,
    candidate.possible_backchannel,
  );
  renderTranscriptCard(
    elements.afterCard,
    `${speakerLabel(candidate.floor_holder_side)} after`,
    candidate.floor_holder_after,
  );
}

function renderTranscriptCard(element, label, segment) {
  const labelElement = document.createElement("span");
  labelElement.className = "card-label";
  labelElement.textContent = label;
  const utterance = document.createElement("div");
  utterance.className = "utterance";
  utterance.textContent = segment.text;
  const timestamp = document.createElement("span");
  timestamp.className = "timestamp";
  timestamp.textContent = `${formatTime(segment.start_seconds)} - ${formatTime(segment.end_seconds)} | ${segment.evidence_source.replace("_", " ")}`;
  element.replaceChildren(labelElement, utterance, timestamp);
}

function renderScores(candidate) {
  const connection = candidate.floor_holder_connection;
  const response = candidate.possible_backchannel;
  elements.scores.replaceChildren(
    interpretationCard(
      `${speakerLabel(candidate.floor_holder_side)} connection`,
      "Do the two surrounding transcript parts belong to the same turn?",
      [
        ["Same turn", connection.merge_confidence, "same-turn"],
        ["New turn", 1 - connection.merge_confidence, "new-turn"],
      ],
      ["Pause evidence", connection.pause_confidence, "pause"],
    ),
    interpretationCard(
      `${speakerLabel(candidate.possible_backchannel_side)} utterance`,
      "How does the segment scorer interpret the highlighted response?",
      [
        ["Backchannel", response.keep_playing_confidence, "backchannel"],
        ["Turn", response.turn_confidence, "turn"],
      ],
      ["Interruption evidence", response.interruption_confidence, "interruption"],
    ),
  );
}

function interpretationCard(title, description, choices, evidence) {
  const card = document.createElement("article");
  card.className = "interpretation";
  const heading = document.createElement("strong");
  heading.textContent = title;
  const explanation = document.createElement("p");
  explanation.textContent = description;
  card.append(heading, explanation, choiceBar(choices), evidenceMeter(...evidence));
  return card;
}

function choiceBar(choices) {
  const wrapper = document.createElement("div");
  wrapper.className = "choice";
  const labels = document.createElement("div");
  labels.className = "choice-labels";
  const track = document.createElement("div");
  track.className = "choice-track";
  for (const [label, value, tone] of choices) {
    const labelElement = document.createElement("span");
    labelElement.textContent = `${label} ${formatProbability(value)}`;
    labels.appendChild(labelElement);
    const fill = document.createElement("div");
    fill.className = `choice-fill ${tone}`;
    fill.style.width = `${boundedPercentage(value)}%`;
    fill.title = `${label}: ${value.toFixed(3)}`;
    track.appendChild(fill);
  }
  wrapper.append(labels, track);
  return wrapper;
}

function evidenceMeter(label, value, tone) {
  const wrapper = document.createElement("div");
  wrapper.className = "evidence";
  const labelElement = document.createElement("span");
  labelElement.textContent = `${label} ${formatProbability(value)}`;
  const track = document.createElement("div");
  track.className = "evidence-track";
  const fill = document.createElement("div");
  fill.className = `evidence-fill ${tone}`;
  fill.style.width = `${boundedPercentage(value)}%`;
  track.appendChild(fill);
  wrapper.append(labelElement, track);
  return wrapper;
}

function createPlayers(candidate) {
  destroyPlayers();
  const tracks = [
    ["speaker1", candidate.speaker1],
    ["speaker2", candidate.speaker2],
  ];
  for (const [side, annotation] of tracks) {
    const track = document.createElement("div");
    track.className = "track";
    const heading = document.createElement("div");
    heading.className = "track-heading";
    const title = document.createElement("strong");
    title.textContent = speakerLabel(side);
    const metadata = document.createElement("span");
    metadata.textContent = `${annotation.segment_targets.length} segments | ${annotation.turns.length} EOT | ${annotation.connection_targets.length} connections`;
    heading.append(title, metadata);
    const canvas = document.createElement("canvas");
    canvas.className = "timeline";
    canvas.height = 68;
    canvas.setAttribute("aria-label", `${speakerLabel(side)} ASR annotations`);
    canvas.addEventListener("click", (event) => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = (event.clientX - bounds.left) / bounds.width;
      inspectTimelineAt(side, annotation, ratio);
      seekToRatio(ratio);
    });
    canvas.addEventListener("pointermove", (event) => {
      const bounds = canvas.getBoundingClientRect();
      inspectTimelineAt(side, annotation, (event.clientX - bounds.left) / bounds.width);
    });
    canvas.addEventListener("pointerleave", () => {
      renderSegmentInspection(
        candidate.possible_backchannel_side,
        candidate.possible_backchannel,
        "Highlighted candidate",
      );
    });
    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.src = `/api/dataset-dashboard/audio/${candidate.sample_id}/${side}`;
    audio.addEventListener("loadedmetadata", () => {
      audio.currentTime = candidate.window_start_seconds;
      updatePlayback();
    });
    audio.addEventListener("error", () => {
      metadata.textContent = "Audio unavailable";
    });
    track.append(heading, canvas, audio);
    elements.tracks.appendChild(track);
    state.players.push({ audio, canvas, annotation });
  }
  seekToRatio(0);
  drawTimelines();
  state.players[0]?.audio.addEventListener("timeupdate", updatePlayback);
}

function inspectTimelineAt(side, annotation, rawRatio) {
  const candidate = currentCandidate();
  const ratio = Math.min(1, Math.max(0, rawRatio));
  const seconds = candidate.window_start_seconds + ratio * clipDuration(candidate);
  const segment = annotation.segment_targets.find(
    (target) => target.start_seconds <= seconds && seconds <= target.end_seconds,
  );
  if (segment) {
    renderSegmentInspection(side, segment, "Timeline segment");
    return;
  }
  const connection = annotation.connection_targets.find(
    (target) =>
      target.earlier_end_seconds <= seconds && seconds <= target.later_start_seconds,
  );
  if (connection) {
    renderConnectionInspection(side, connection);
    return;
  }
  const pointTolerance = clipDuration(candidate) * 0.008;
  const endOfTurn = annotation.turns.find(
    (turn) => Math.abs(turn.time_seconds - seconds) <= pointTolerance,
  );
  if (endOfTurn) {
    renderPointInspection(side, "End-of-turn marker", endOfTurn.time_seconds);
    return;
  }
  elements.timelineInspector.replaceChildren(
    inspectorHeading(`${speakerLabel(side)} at ${formatTime(seconds)}`),
    inspectorText("No scored transcript segment or connection at this position."),
  );
}

function renderSegmentInspection(side, segment, contextLabel) {
  const primaryLabel =
    segment.evidence_source === "transcript" ? "Backchannel" : "Keep playing";
  elements.timelineInspector.replaceChildren(
    inspectorHeading(`${contextLabel} - ${speakerLabel(side)}`),
    inspectorText(
      `${formatTime(segment.start_seconds)} - ${formatTime(segment.end_seconds)} | ${segment.text}`,
    ),
    choiceBar([
      [primaryLabel, segment.keep_playing_confidence, "backchannel"],
      ["Turn", segment.turn_confidence, "turn"],
    ]),
    evidenceMeter(
      "Interruption evidence",
      segment.interruption_confidence,
      "interruption",
    ),
  );
}

function renderConnectionInspection(side, connection) {
  elements.timelineInspector.replaceChildren(
    inspectorHeading(`Timeline connection - ${speakerLabel(side)}`),
    inspectorText(
      `${formatTime(connection.earlier_end_seconds)} - ${formatTime(connection.later_start_seconds)} | ${connection.gap_seconds.toFixed(2)} s gap`,
    ),
    choiceBar([
      ["Same turn", connection.merge_confidence, "same-turn"],
      ["New turn", 1 - connection.merge_confidence, "new-turn"],
    ]),
    evidenceMeter("Pause evidence", connection.pause_confidence, "pause"),
  );
}

function renderPointInspection(side, label, timeSeconds) {
  elements.timelineInspector.replaceChildren(
    inspectorHeading(`${label} - ${speakerLabel(side)}`),
    inspectorText(
      `${formatTime(timeSeconds)} | This emitted marker does not store a separate confidence value.`,
    ),
  );
}

function inspectorHeading(text) {
  const heading = document.createElement("strong");
  heading.textContent = text;
  return heading;
}

function inspectorText(text) {
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  return paragraph;
}

async function play() {
  const candidate = currentCandidate();
  const master = state.players[0]?.audio;
  if (!master) {
    return;
  }
  if (master.currentTime >= candidate.window_end_seconds - 0.02) {
    seekToRatio(0);
  }
  const startSeconds = master.currentTime;
  for (const player of state.players) {
    player.audio.currentTime = startSeconds;
  }
  const results = await Promise.allSettled(state.players.map((player) => player.audio.play()));
  if (results.some((result) => result.status === "rejected")) {
    pause();
    return;
  }
  state.playing = true;
  updatePlayback();
}

function pause() {
  for (const player of state.players) {
    player.audio.pause();
  }
  state.playing = false;
  elements.play.textContent = "Play both";
}

function seekToRatio(rawRatio) {
  const candidate = currentCandidate();
  if (!candidate) {
    return;
  }
  const ratio = Math.min(1, Math.max(0, rawRatio));
  const seconds = candidate.window_start_seconds + ratio * clipDuration(candidate);
  for (const player of state.players) {
    player.audio.currentTime = seconds;
  }
  updatePlayback();
}

function updatePlayback() {
  const candidate = currentCandidate();
  const master = state.players[0]?.audio;
  if (!candidate || !master) {
    return;
  }
  if (master.currentTime >= candidate.window_end_seconds) {
    for (const player of state.players) {
      player.audio.currentTime = candidate.window_end_seconds;
    }
    pause();
  }
  if (state.playing) {
    for (const player of state.players.slice(1)) {
      if (Math.abs(player.audio.currentTime - master.currentTime) > 0.06) {
        player.audio.currentTime = master.currentTime;
      }
    }
  }
  const elapsed = Math.min(
    clipDuration(candidate),
    Math.max(0, master.currentTime - candidate.window_start_seconds),
  );
  const ratio = clipDuration(candidate) > 0 ? elapsed / clipDuration(candidate) : 0;
  if (!state.seeking) {
    elements.seek.value = String(Math.round(ratio * 1000));
  }
  elements.clock.textContent = `${formatClipTime(elapsed)} / ${formatClipTime(clipDuration(candidate))}`;
  elements.play.textContent = state.playing ? "Pause both" : "Play both";
  drawTimelines();
}

function drawTimelines() {
  const candidate = currentCandidate();
  if (!candidate) {
    return;
  }
  for (const player of state.players) {
    const width = Math.max(320, Math.floor(player.canvas.getBoundingClientRect().width));
    if (player.canvas.width !== width) {
      player.canvas.width = width;
    }
    const context = player.canvas.getContext("2d");
    if (!context) {
      continue;
    }
    context.clearRect(0, 0, player.canvas.width, player.canvas.height);
    drawAnnotationTimelineRow({
      context,
      annotation: player.annotation,
      left: 0,
      top: 10,
      width: player.canvas.width,
      viewportStartSeconds: candidate.window_start_seconds,
      viewportEndSeconds: candidate.window_end_seconds,
    });
    drawFocus(context, candidate, player.canvas.width);
    const master = state.players[0]?.audio;
    if (master) {
      const progressRatio = Math.min(
        1,
        Math.max(
          0,
          (master.currentTime - candidate.window_start_seconds) / clipDuration(candidate),
        ),
      );
      context.strokeStyle = "#111827";
      context.lineWidth = 1.5;
      context.beginPath();
      context.moveTo(progressRatio * player.canvas.width, 2);
      context.lineTo(progressRatio * player.canvas.width, 66);
      context.stroke();
    }
  }
}

function drawFocus(context, candidate, width) {
  const startRatio =
    (candidate.possible_backchannel.start_seconds - candidate.window_start_seconds) /
    clipDuration(candidate);
  const endRatio =
    (candidate.possible_backchannel.end_seconds - candidate.window_start_seconds) /
    clipDuration(candidate);
  context.strokeStyle = "#7c3aed";
  context.lineWidth = 2;
  context.strokeRect(startRatio * width, 1, Math.max(2, (endRatio - startRatio) * width), 65);
}

function destroyPlayers() {
  pause();
  for (const player of state.players) {
    player.audio.removeAttribute("src");
    player.audio.load();
  }
  state.players = [];
  elements.tracks.replaceChildren();
}

function currentCandidate() {
  return state.candidates[state.index];
}

function clipDuration(candidate) {
  return candidate.window_end_seconds - candidate.window_start_seconds;
}

function speakerLabel(side) {
  return side === "speaker1" ? "Speaker 1" : "Speaker 2";
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
}

function formatClipTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function boundedPercentage(value) {
  return Math.max(0, Math.min(100, value * 100));
}

function formatProbability(value) {
  return `${Math.round(boundedPercentage(value))}%`;
}
