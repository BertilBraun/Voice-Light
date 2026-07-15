const sampleSelect = document.querySelector("#sample-select");
const userSideSelect = document.querySelector("#user-side-select");
const startInput = document.querySelector("#start-input");
const loadButton = document.querySelector("#load-button");
const randomButton = document.querySelector("#random-button");
const positionSlider = document.querySelector("#position-slider");
const positionLabel = document.querySelector("#position-label");
const roleLabel = document.querySelector("#role-label");
const playButton = document.querySelector("#play-button");
const timeReadout = document.querySelector("#time-readout");
const userAudio = document.querySelector("#user-audio");
const timeline = document.querySelector("#timeline");
const status = document.querySelector("#status");
const summary = document.querySelector("#summary");
const frameTime = document.querySelector("#frame-time");
const frameDetails = document.querySelector("#frame-details");

const rowDefinitions = [
  ["User waveform", "waveform"],
  ["Candidate decision mask", "candidate"],
  ["Primary p(YIELD)", "yield_probability"],
  ["Primary p(HOLD)", "hold_probability"],
  ["Event: turn completion", "event_distribution.turn_completion"],
  ["Event: continuation pause", "event_distribution.continuation_pause"],
  ["Event: backchannel", "event_distribution.backchannel"],
  ["Event: interruption", "event_distribution.interruption"],
  ["Event: other", "event_distribution.other"],
  ["Future activity 0–200 ms", "future_activity.0"],
  ["Future activity 200–500 ms", "future_activity.1"],
  ["Future activity 500–1000 ms", "future_activity.2"],
  ["Future activity 1000–1500 ms", "future_activity.3"],
];

let preview = null;
let selectedFrameIndex = null;
let playbackSeconds = 0;

async function loadSamples() {
  try {
    const response = await fetch("/api/dataset-dashboard/samples?limit=200");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(errorMessage(payload, response.status));
    }
    const annotatedSamples = payload.samples.filter(
      (sample) => sample.latest_quality?.payload?.conversation_annotation != null,
    );
    sampleSelect.replaceChildren(
      ...annotatedSamples.map((sample) => {
        const option = document.createElement("option");
        option.value = sample.sample.id;
        const annotation = sample.latest_quality.payload.conversation_annotation;
        option.textContent = `${sample.sample.external_id} · ${formatDuration(sample.sample.duration_seconds ?? 0)} · ${annotation.usable_event_count} events`;
        return option;
      }),
    );
    if (annotatedSamples.length === 0) {
      throw new Error("No annotated dataset samples are available.");
    }
    setStatus(`${annotatedSamples.length} annotated samples available`, false);
    await loadPreview(true);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
  }
}

async function loadPreview(randomLocation) {
  const sampleId = sampleSelect.value;
  if (!sampleId) {
    return;
  }
  pausePlayback();
  setStatus("Building frame preview…", false);
  const parameters = new URLSearchParams({
    sample_id: sampleId,
    user_side: userSideSelect.value,
  });
  if (!randomLocation) {
    parameters.set("start_seconds", String(Number(startInput.value)));
  }
  try {
    const response = await fetch(`/api/training-samples/preview?${parameters.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(errorMessage(payload, response.status));
    }
    preview = payload;
    selectedFrameIndex = null;
    playbackSeconds = preview.start_seconds;
    configureControls();
    configureAudio();
    renderSummary();
    renderSelectedFrame(null);
    drawTimeline();
    const coverageMessage =
      preview.annotated_duration_seconds < preview.represented_duration_seconds
        ? `Preview ready · ${formatDuration(preview.annotated_duration_seconds)} annotated of ${formatDuration(preview.represented_duration_seconds)}`
        : "Preview ready";
    setStatus(coverageMessage, false);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
  }
}

function configureControls() {
  const maximumStart = Math.max(
    0,
    preview.eligible_duration_seconds - preview.input_duration_seconds,
  );
  startInput.max = String(maximumStart);
  startInput.value = preview.start_seconds.toFixed(2);
  positionSlider.max = String(maximumStart);
  positionSlider.value = String(preview.start_seconds);
  positionLabel.textContent = `${formatDuration(preview.start_seconds)}–${formatDuration(preview.end_seconds)} of ${formatDuration(preview.eligible_duration_seconds)} annotated`;
  roleLabel.textContent = `${prettySide(preview.user_side)} = user input · ${prettySide(preview.assistant_side)} = reference assistant`;
  playButton.disabled = false;
  updateTimeReadout();
}

function configureAudio() {
  userAudio.src = `/api/dataset-dashboard/audio/${preview.sample_id}/${preview.user_side}`;
  userAudio.currentTime = preview.start_seconds;
}

function renderSummary() {
  const supervisedFrames = preview.frames.filter((frame) => frame.supervised);
  const candidateFrames = supervisedFrames.filter((frame) => frame.candidate);
  const validPrimaryFrames = candidateFrames.filter((frame) => frame.primary_valid);
  const ambiguousFrames = validPrimaryFrames.filter(
    (frame) => frame.yield_probability >= 0.25 && frame.yield_probability <= 0.75,
  );
  const meanYield =
    validPrimaryFrames.length === 0
      ? null
      : validPrimaryFrames.reduce((total, frame) => total + frame.yield_probability, 0) /
        validPrimaryFrames.length;
  const missingReliability = validPrimaryFrames.filter(
    (frame) => frame.primary_reliability === null,
  ).length;
  summary.replaceChildren(
    ...definitionRows([
      ["Input / supervised", `${preview.input_duration_seconds.toFixed(1)} s / ${preview.supervised_duration_seconds.toFixed(1)} s`],
      ["Recording duration", formatDuration(preview.represented_duration_seconds)],
      ["Annotated duration", formatDuration(preview.annotated_duration_seconds)],
      ["Annotation coverage", percentage(preview.annotated_duration_seconds, preview.represented_duration_seconds)],
      ["Frame interval", `${Math.round(preview.frame_seconds * 1000)} ms`],
      ["Supervised frames", String(supervisedFrames.length)],
      ["Candidate frames", String(candidateFrames.length)],
      ["Valid primary targets", String(validPrimaryFrames.length)],
      ["Mean p(YIELD)", meanYield === null ? "—" : meanYield.toFixed(3)],
      ["Ambiguous primary frames", String(ambiguousFrames.length)],
      ["Reliability unmeasured", String(missingReliability)],
    ]),
  );
}

function drawTimeline() {
  const devicePixelRatio = window.devicePixelRatio || 1;
  const displayWidth = Math.max(980, timeline.clientWidth);
  const displayHeight = 610;
  timeline.width = Math.round(displayWidth * devicePixelRatio);
  timeline.height = Math.round(displayHeight * devicePixelRatio);
  const context = timeline.getContext("2d");
  context.scale(devicePixelRatio, devicePixelRatio);
  context.clearRect(0, 0, displayWidth, displayHeight);
  if (preview === null) {
    return;
  }

  const left = 190;
  const right = 16;
  const top = 36;
  const rowHeight = 36;
  const rowGap = 6;
  const plotWidth = displayWidth - left - right;
  const burnRatio = (preview.burn_in_end_seconds - preview.start_seconds) / preview.input_duration_seconds;
  context.fillStyle = "#eef1f2";
  context.fillRect(left, top - 16, plotWidth * burnRatio, displayHeight - top + 2);
  context.fillStyle = "#68767b";
  context.font = "11px sans-serif";
  context.fillText("4 s burn-in · loss masked", left + 8, top - 5);

  rowDefinitions.forEach(([label, field], rowIndex) => {
    const rowTop = top + rowIndex * (rowHeight + rowGap);
    context.fillStyle = rowIndex % 2 === 0 ? "#f7f9f8" : "#f1f4f3";
    context.fillRect(left, rowTop, plotWidth, rowHeight);
    context.fillStyle = "#48575c";
    context.font = field === "yield_probability" ? "bold 12px sans-serif" : "12px sans-serif";
    context.fillText(label, 8, rowTop + 23);
    if (field === "waveform") {
      drawWaveform(context, left, rowTop, plotWidth, rowHeight);
    } else {
      drawTargetFrames(context, field, left, rowTop, plotWidth, rowHeight);
    }
  });

  drawTimeAxis(context, left, plotWidth, displayHeight);
  drawCursor(context, left, plotWidth, top, displayHeight);
}

function drawWaveform(context, left, top, width, height) {
  const points = preview.user_waveform;
  const middle = top + height / 2;
  context.strokeStyle = "#267d74";
  context.lineWidth = 1;
  points.forEach((point, index) => {
    const x = left + (index / Math.max(1, points.length - 1)) * width;
    context.beginPath();
    context.moveTo(x, middle - point.maximum_amplitude * height * 0.45);
    context.lineTo(x, middle - point.minimum_amplitude * height * 0.45);
    context.stroke();
  });
}

function drawTargetFrames(context, field, left, top, width, height) {
  const frameWidth = width / preview.frames.length;
  preview.frames.forEach((frame, index) => {
    const value = targetValue(frame, field);
    if (value === null) {
      return;
    }
    if (typeof value === "boolean") {
      context.fillStyle = value ? "#7ebdb4" : "rgba(0, 0, 0, 0)";
      if (!value) {
        return;
      }
    } else {
      context.fillStyle = `rgba(20, 107, 99, ${0.08 + 0.92 * value})`;
    }
    if (value === true || typeof value === "number") {
      context.fillRect(left + index * frameWidth, top + 3, Math.max(1, frameWidth + 0.2), height - 6);
    }
  });
}

function targetValue(frame, field) {
  if (field === "candidate") {
    return frame.candidate;
  }
  if (field.startsWith("event_distribution.")) {
    return frame.event_distribution?.[field.split(".")[1]] ?? null;
  }
  if (field.startsWith("future_activity.")) {
    const target = frame.future_activity[Number(field.split(".")[1])];
    return target?.valid === true ? target.active : null;
  }
  return frame[field] ?? null;
}

function drawTimeAxis(context, left, width, height) {
  context.strokeStyle = "#cbd3d1";
  context.fillStyle = "#657378";
  context.font = "11px sans-serif";
  for (let seconds = 0; seconds <= preview.input_duration_seconds; seconds += 2) {
    const ratio = seconds / preview.input_duration_seconds;
    const x = left + ratio * width;
    context.beginPath();
    context.moveTo(x, 24);
    context.lineTo(x, height - 18);
    context.stroke();
    context.fillText(`${seconds}s`, x + 3, height - 4);
  }
}

function drawCursor(context, left, width, top, height) {
  if (playbackSeconds < preview.start_seconds || playbackSeconds > preview.end_seconds) {
    return;
  }
  const ratio = (playbackSeconds - preview.start_seconds) / preview.input_duration_seconds;
  const x = left + ratio * width;
  context.strokeStyle = "#172024";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(x, top - 17);
  context.lineTo(x, height - 18);
  context.stroke();
}

function renderSelectedFrame(frame) {
  if (frame === null) {
    frameTime.textContent = "Select a frame";
    frameDetails.replaceChildren();
    return;
  }
  frameTime.textContent = `Frame ${frame.frame_index} · +${frame.relative_time_seconds.toFixed(2)} s · absolute ${formatDuration(frame.time_seconds)}`;
  frameDetails.replaceChildren(
    ...definitionRows([
      ["Contributes loss", booleanLabel(frame.supervised)],
      ["Candidate", booleanLabel(frame.candidate)],
      ["Candidate source", frame.candidate_source ?? "—"],
      ["Since user speech offset", optionalSeconds(frame.seconds_since_speech_offset)],
      ["Primary target p(YIELD)", optionalProbability(frame.yield_probability)],
      ["Primary target p(HOLD)", optionalProbability(frame.hold_probability)],
      ["Primary target valid", booleanLabel(frame.primary_valid)],
      ["Primary reliability", optionalProbability(frame.primary_reliability)],
      ["Reliability source", frame.primary_reliability_source ?? "—"],
      ["Event: completion", eventProbability(frame, "turn_completion")],
      ["Event: continuation pause", eventProbability(frame, "continuation_pause")],
      ["Event: backchannel", eventProbability(frame, "backchannel")],
      ["Event: interruption", eventProbability(frame, "interruption")],
      ["Event: other", eventProbability(frame, "other")],
      ...frame.future_activity.map((target) => [
        `Future ${target.start_milliseconds}–${target.end_milliseconds} ms`,
        target.valid ? booleanLabel(target.active) : "MASKED",
      ]),
    ]),
  );
}

function definitionRows(entries) {
  return entries.flatMap(([label, value]) => {
    const term = document.createElement("dt");
    term.textContent = label;
    const definition = document.createElement("dd");
    definition.textContent = value;
    if (value === "YES" || value === "SPEAK") {
      definition.className = "value-positive";
    } else if (value === "NO" || value === "LISTEN") {
      definition.className = "value-negative";
    }
    return [term, definition];
  });
}

function selectFrameAtEvent(event) {
  if (preview === null) {
    return;
  }
  const rect = timeline.getBoundingClientRect();
  const left = 190;
  const right = 16;
  const plotWidth = rect.width - left - right;
  const x = event.clientX - rect.left;
  if (x < left || x > left + plotWidth) {
    return;
  }
  const ratio = (x - left) / plotWidth;
  selectedFrameIndex = Math.min(
    preview.frames.length - 1,
    Math.max(0, Math.floor(ratio * preview.frames.length)),
  );
  const frame = preview.frames[selectedFrameIndex];
  playbackSeconds = frame.time_seconds;
  userAudio.currentTime = playbackSeconds;
  renderSelectedFrame(frame);
  updateTimeReadout();
  drawTimeline();
}

async function togglePlayback() {
  if (preview === null) {
    return;
  }
  if (!userAudio.paused) {
    pausePlayback();
    return;
  }
  if (userAudio.currentTime < preview.start_seconds || userAudio.currentTime >= preview.end_seconds) {
    userAudio.currentTime = preview.start_seconds;
  }
  await userAudio.play();
  playButton.textContent = "Pause";
}

function pausePlayback() {
  userAudio.pause();
  playButton.textContent = "Play user input";
}

function trackPlayback() {
  if (preview === null) {
    return;
  }
  playbackSeconds = userAudio.currentTime;
  if (playbackSeconds >= preview.end_seconds) {
    userAudio.currentTime = preview.end_seconds;
    playbackSeconds = preview.end_seconds;
    pausePlayback();
  }
  updateTimeReadout();
  drawTimeline();
}

function updateTimeReadout() {
  if (preview === null) {
    return;
  }
  timeReadout.textContent = `${Math.max(0, playbackSeconds - preview.start_seconds).toFixed(2)} s / ${preview.input_duration_seconds.toFixed(2)} s`;
}

function setStatus(message, isError) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function errorMessage(payload, statusCode) {
  return typeof payload.detail === "string" ? payload.detail : `Request failed (${statusCode})`;
}

function prettySide(side) {
  return side === "speaker1" ? "Speaker 1" : "Speaker 2";
}

function booleanLabel(value) {
  return value ? "YES" : "NO";
}

function optionalProbability(value) {
  return value === null ? "UNMEASURED" : value.toFixed(3);
}

function optionalSeconds(value) {
  return value === null ? "—" : `${value.toFixed(2)} s`;
}

function eventProbability(frame, field) {
  return frame.event_distribution === null ? "—" : frame.event_distribution[field].toFixed(3);
}

function percentage(count, total) {
  return total === 0 ? "0.0%" : `${((100 * count) / total).toFixed(1)}%`;
}

function formatDuration(totalSeconds) {
  const bounded = Math.max(0, totalSeconds);
  const minutes = Math.floor(bounded / 60);
  const seconds = bounded - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}

sampleSelect.addEventListener("change", () => loadPreview(true));
userSideSelect.addEventListener("change", () => loadPreview(true));
loadButton.addEventListener("click", () => loadPreview(false));
randomButton.addEventListener("click", () => loadPreview(true));
positionSlider.addEventListener("input", () => {
  startInput.value = Number(positionSlider.value).toFixed(2);
});
positionSlider.addEventListener("change", () => loadPreview(false));
playButton.addEventListener("click", togglePlayback);
userAudio.addEventListener("timeupdate", trackPlayback);
userAudio.addEventListener("ended", pausePlayback);
timeline.addEventListener("click", selectFrameAtEvent);
timeline.addEventListener("pointermove", (event) => {
  if (event.buttons === 0) {
    selectFrameAtEvent(event);
  }
});
window.addEventListener("resize", drawTimeline);

await loadSamples();
