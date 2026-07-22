import { drawAnnotationTimelineRow } from "/pages/shared/annotation-timeline.js";
import { createMediaElementGainController } from "/pages/shared/audio-gain.js";
import { commonWaveformDisplayScale } from "/pages/shared/waveform-rendering.js";
import {
  createConversationContextOverview,
  drawUnusableRegionOverlay,
} from "/pages/training-samples/context-overview.js";

const datasetSelect = document.querySelector("#dataset-select");
const sampleSelect = document.querySelector("#sample-select");
const userSideSelect = document.querySelector("#user-side-select");
const startInput = document.querySelector("#start-input");
const minimumQualityInput = document.querySelector("#minimum-quality-input");
const samplingModeSelect = document.querySelector("#sampling-mode-select");
const randomizeInitialInput = document.querySelector("#randomize-initial-input");
const loadButton = document.querySelector("#load-button");
const randomButton = document.querySelector("#random-button");
const nextRandomButton = document.querySelector("#next-random-button");
const positionSlider = document.querySelector("#position-slider");
const positionLabel = document.querySelector("#position-label");
const roleLabel = document.querySelector("#role-label");
const playButton = document.querySelector("#play-button");
const playBothInput = document.querySelector("#play-both-input");
const timeReadout = document.querySelector("#time-readout");
const userAudio = document.querySelector("#user-audio");
const assistantAudio = document.querySelector("#assistant-audio");
const timeline = document.querySelector("#timeline");
const annotationTimeline = document.querySelector("#annotation-timeline");
const annotationSource = document.querySelector("#annotation-source");
const contextOverview = document.querySelector("#context-overview");
const contextLabel = document.querySelector("#context-label");
const status = document.querySelector("#status");
const summary = document.querySelector("#summary");
const frameTime = document.querySelector("#frame-time");
const frameDetails = document.querySelector("#frame-details");
const NO_AUXILIARY_ANNOTATION_REASON = "No auxiliary annotation applies at this frame";
const OUTSIDE_USER_YIELD_CONTEXT_REASON = "Outside the user-yield decision window";
const BURN_IN_REASON = "Burn-in recurrent-state warm-up";
const PREPARED_PREVIEW_TARGET = 2;

const rowDefinitions = [
  { label: "User waveform", field: "waveform" },
  { label: "INPUT · p_assistant_has_floor", field: "assistant_has_floor_input" },
  {
    label: "RUNTIME · p_user_has_floor",
    field: "user_has_floor_target",
    validField: "user_has_floor_valid",
  },
  {
    label: "RUNTIME · p_user_yield ≤ 500 ms",
    field: "user_yield_target",
    validField: "user_yield_valid",
  },
  {
    label: "RUNTIME · p_assistant_backchannel ≤ 200 ms",
    field: "assistant_backchannel_target",
    validField: "assistant_backchannel_valid",
  },
  {
    label: "AUX point · turn completion",
    field: "interaction_auxiliary.turn_completion",
  },
  {
    label: "AUX span · continuation pause",
    field: "interaction_auxiliary.continuation_pause",
  },
  {
    label: "AUX span · non-floor feedback",
    field: "interaction_auxiliary.non_floor_feedback",
  },
  {
    label: "AUX point · floor take / interruption",
    field: "interaction_auxiliary.floor_take",
  },
  { label: "AUX future user audio · 0–200 ms", field: "future_activity.0" },
  { label: "AUX future user audio · 200–500 ms", field: "future_activity.1" },
  { label: "AUX future user audio · 500–1000 ms", field: "future_activity.2" },
  { label: "AUX future user audio · 1000–1500 ms", field: "future_activity.3" },
];

let preview = null;
let selectedFrameIndex = null;
let playbackSeconds = 0;
let preparedPreviewQueue = [];
let nextPreviewPreparation = null;
let reviewQueueGeneration = 0;
const playbackGainController = createMediaElementGainController([
  { id: "user", element: userAudio },
  { id: "assistant", element: assistantAudio },
]);
const contextOverviewController = createConversationContextOverview({
  canvas: contextOverview,
  label: contextLabel,
  getPreview: () => preview,
  getSelectedStartSeconds: () => Number(startInput.value),
  setSelectedStartSeconds: (startSeconds) => {
    startInput.value = startSeconds.toFixed(2);
    positionSlider.value = String(startSeconds);
  },
  commitSelection: () => {
    void loadPreview(false);
  },
  reportError: (error) => {
    setStatus(error instanceof Error ? error.message : String(error), true);
  },
});

async function loadDatasets() {
  const response = await fetch("/api/dataset-dashboard/datasets", {
    cache: "no-store",
  });
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(errorMessage(payload, response.status));
  }
  datasetSelect.replaceChildren(
    ...payload.datasets.map((dataset) => {
      const option = document.createElement("option");
      option.value = dataset.id;
      option.textContent = dataset.name;
      option.dataset.name = dataset.name;
      return option;
    }),
  );
  if (payload.datasets.length === 0) {
    throw new Error("No datasets are available.");
  }
  const meetingsOption = Array.from(datasetSelect.options).find(
    (option) => option.dataset.name === "meetings-s3",
  );
  datasetSelect.value = meetingsOption?.value ?? datasetSelect.options[0].value;
}

async function loadSamples() {
  try {
    const parameters = new URLSearchParams({
      dataset_id: datasetSelect.value,
      limit: "100",
    });
    const minimumQuality = selectedMinimumQuality();
    if (minimumQuality !== null) {
      parameters.set("minimum_quality", String(minimumQuality));
    }
    const response = await fetch(`/api/training-samples/options?${parameters}`, {
      cache: "no-store",
    });
    const samples = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(errorMessage(samples, response.status));
    }
    sampleSelect.replaceChildren(
      ...samples.map((sample) => {
        const option = document.createElement("option");
        option.value = sample.sample_id;
        option.textContent = sampleOptionLabel(sample);
        return option;
      }),
    );
    if (samples.length === 0) {
      throw new Error("No annotated dataset samples are available.");
    }
    if (randomizeInitialInput.checked) {
      const randomIndex = Math.floor(Math.random() * samples.length);
      sampleSelect.value = samples[randomIndex].sample_id;
    }
    setStatus(`${samples.length} recent annotated samples loaded`, false);
    await loadPreview(true);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
  }
}

async function loadPreview(randomLocation, autoplay = false) {
  const sampleId = sampleSelect.value;
  if (!sampleId) {
    return;
  }
  resetReviewQueue();
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
    const response = await fetch(`/api/training-samples/preview?${parameters.toString()}`, {
      cache: "no-store",
    });
    const payload = await readJsonResponse(response);
    if (!response.ok) {
      throw new Error(errorMessage(payload, response.status));
    }
    await applyPreview(payload, autoplay);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
  }
}

function resetReviewQueue() {
  reviewQueueGeneration += 1;
  preparedPreviewQueue = [];
  nextPreviewPreparation = null;
  updateNextSampleButton();
}

async function prepareNextReviewSample() {
  if (
    preview === null ||
    nextPreviewPreparation !== null ||
    preparedPreviewQueue.length >= PREPARED_PREVIEW_TARGET
  ) {
    return;
  }
  const generation = reviewQueueGeneration;
  updateNextSampleButton();
  nextPreviewPreparation = fillPreparedPreviewQueue(generation);
  try {
    await nextPreviewPreparation;
  } catch (error) {
    if (generation !== reviewQueueGeneration) {
      return;
    }
    setStatus(error instanceof Error ? error.message : String(error), true);
  } finally {
    if (generation === reviewQueueGeneration) {
      nextPreviewPreparation = null;
      updateNextSampleButton();
    }
  }
}

async function fillPreparedPreviewQueue(generation) {
  let sourcePreview = preparedPreviewQueue.at(-1) ?? preview;
  while (
    generation === reviewQueueGeneration &&
    preparedPreviewQueue.length < PREPARED_PREVIEW_TARGET
  ) {
    const payload = await buildNextReviewPreview(sourcePreview, generation);
    if (payload === null || generation !== reviewQueueGeneration) {
      return;
    }
    preparedPreviewQueue.push(payload);
    sourcePreview = payload;
    updateNextSampleButton();
  }
}

function updateNextSampleButton() {
  const preparedCount = preparedPreviewQueue.length;
  nextRandomButton.disabled = preparedCount === 0;
  nextRandomButton.textContent =
    preparedCount === 0 ? "Preparing next sample…" : `Next sample (${preparedCount} ready)`;
}

async function buildNextReviewPreview(sourcePreview, generation) {
  const nextConversation = await fetchNextConversationPreview(sourcePreview);
  return generation === reviewQueueGeneration ? nextConversation : null;
}

async function fetchNextConversationPreview(sourcePreview) {
  const parameters = new URLSearchParams({
    dataset_id: sourcePreview.dataset_id,
    current_sample_id: sourcePreview.sample_id,
    sampling_mode: samplingModeSelect.value,
  });
  const minimumQuality = selectedMinimumQuality();
  if (minimumQuality !== null) {
    parameters.set("minimum_quality", String(minimumQuality));
  }
  const response = await fetch(
    `/api/training-samples/random-preview?${parameters}`,
    { cache: "no-store" },
  );
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(errorMessage(payload, response.status));
  }
  return payload;
}

async function applyPreview(payload, autoplay) {
  preview = payload;
  contextOverviewController.reset();
  selectedFrameIndex = null;
  playbackSeconds = preview.start_seconds;
  ensureSampleOption();
  userSideSelect.value = preview.user_side;
  configureControls();
  configureAudio();
  configurePlaybackGains();
  renderSummary();
  renderAnnotationSource();
  renderSelectedFrame(null);
  drawTimeline();
  drawSourceAnnotationTimeline();
  void contextOverviewController
    .load(preview.start_seconds)
    .catch((error) =>
      setStatus(error instanceof Error ? error.message : String(error), true),
    );
  const coverageMessage =
    preview.annotated_duration_seconds < preview.represented_duration_seconds
      ? `Preview ready · ${formatDuration(preview.annotated_duration_seconds)} annotated of ${formatDuration(preview.represented_duration_seconds)}`
      : "Preview ready";
  const playbackStarted = autoplay ? await startPlayback() : false;
  let statusMessage = coverageMessage;
  if (playbackStarted) {
    statusMessage = `${coverageMessage} - playing`;
  } else if (autoplay) {
    statusMessage = `${coverageMessage} - press Play if autoplay was blocked`;
  }
  setStatus(statusMessage, false);
  void prepareNextReviewSample();
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
  roleLabel.textContent = `${prettySide(preview.user_side)} = user audio · ${prettySide(preview.assistant_side)} = assistant-state input`;
  playButton.disabled = false;
  updateTimeReadout();
}

function configureAudio() {
  configureAudioElement(
    userAudio,
    preview.user_side,
    preview.user_audio_sha256,
  );
  configureAudioElement(
    assistantAudio,
    preview.assistant_side,
    preview.assistant_audio_sha256,
  );
  setCanonicalPlaybackTime(preview.start_seconds);
}

function configureAudioElement(audioElement, side, audioSha256) {
  const sourceUrl =
    `/api/dataset-dashboard/audio/${preview.sample_id}/${side}` +
    `?v=${encodeURIComponent(audioSha256)}` +
    `&timeline=${encodeURIComponent(preview.timeline_fingerprint ?? "raw")}`;
  if (audioElement.dataset.sourceUrl === sourceUrl) {
    return;
  }
  audioElement.dataset.sourceUrl = sourceUrl;
  audioElement.src = sourceUrl;
  audioElement.load();
}

function configurePlaybackGains() {
  playbackGainController.setGain("user", preview.user_gain.default_gain);
  playbackGainController.setGain("assistant", preview.assistant_gain.default_gain);
}

async function startPlayback() {
  synchronizeAudioTracks();
  if (!playBothInput.checked) {
    assistantAudio.pause();
  }
  try {
    await playbackGainController.ensureConnected();
    const playRequests = [userAudio.play()];
    if (playBothInput.checked) {
      playRequests.push(assistantAudio.play());
    }
    await Promise.all(playRequests);
    playButton.textContent = "Pause";
    return true;
  } catch {
    pausePlayback();
    return false;
  }
}

async function loadNextPreparedSample() {
  if (preparedPreviewQueue.length === 0) {
    await prepareNextReviewSample();
    if (nextPreviewPreparation !== null) {
      await nextPreviewPreparation;
    }
  }
  if (preparedPreviewQueue.length === 0) {
    return;
  }
  const payload = preparedPreviewQueue.shift();
  updateNextSampleButton();
  pausePlayback();
  await applyPreview(payload, true);
}

function ensureSampleOption() {
  let option = Array.from(sampleSelect.options).find(
    (candidate) => candidate.value === String(preview.sample_id),
  );
  if (option === undefined) {
    option = document.createElement("option");
    option.value = preview.sample_id;
    option.textContent = `${preview.external_id} · quality ${optionalScore(preview.quality.total_score)} · ${formatDuration(preview.represented_duration_seconds)}`;
    sampleSelect.append(option);
  }
  sampleSelect.value = preview.sample_id;
}

function renderSummary() {
  const supervisedFrames = preview.frames.filter((frame) => frame.supervised);
  const assistantFloorFrames = preview.frames.filter(
    (frame) => frame.assistant_has_floor_input >= 0.5,
  );
  const validHasFloorFrames = supervisedFrames.filter(
    (frame) => frame.user_has_floor_valid,
  );
  const validYieldFrames = supervisedFrames.filter((frame) => frame.user_yield_valid);
  const validAssistantBackchannelFrames = supervisedFrames.filter(
    (frame) => frame.assistant_backchannel_valid,
  );
  const hasFloorPositives = validHasFloorFrames.filter(
    (frame) => frame.user_has_floor_target >= 0.5,
  ).length;
  const yieldSoonPositives = validYieldFrames.filter(
    (frame) => frame.user_yield_target >= 0.5,
  ).length;
  const assistantBackchannelPositives = validAssistantBackchannelFrames.filter(
    (frame) => frame.assistant_backchannel_target >= 0.5,
  ).length;
  const auxiliaryCounts = interactionAuxiliaryCounts(preview.frames);
  summary.replaceChildren(
    ...definitionRows([
      ["Overall quality", optionalScore(preview.quality.total_score)],
      ["Conversation quality", optionalScore(preview.quality.conversation_quality_score)],
      ["Audio quality", optionalScore(preview.quality.audio_quality_score)],
      ["User automatic gain", formatGain(preview.user_gain)],
      ["Assistant automatic gain", formatGain(preview.assistant_gain)],
      ["Timing reliability", optionalScore(preview.quality.timing_reliability_score)],
      ["Interaction density", optionalScore(preview.quality.interaction_density_score)],
      ["Usable events", optionalInteger(preview.quality.usable_event_count)],
      ["Events per hour", optionalDecimal(preview.quality.events_per_hour)],
      [
        "Permissive usable duration",
        preview.conversation_regions === null
          ? "Not analyzed"
          : formatDuration(preview.conversation_regions.usable_duration_seconds),
      ],
      [
        "Permissive usable ratio",
        preview.conversation_regions === null
          ? "Not analyzed"
          : `${(preview.conversation_regions.usable_ratio * 100).toFixed(1)}%`,
      ],
      ["Quality flags", preview.quality.flags.length === 0 ? "None" : preview.quality.flags.join(", ")],
      ["Annotation version", preview.annotation_version],
      ["Annotation generated", formatDateTime(preview.annotation_generated_at)],
      ["Quality metric", preview.quality_metric_version],
      ["Input / supervised", `${preview.input_duration_seconds.toFixed(1)} s / ${preview.supervised_duration_seconds.toFixed(1)} s`],
      ["Recording duration", formatDuration(preview.represented_duration_seconds)],
      ["Annotated duration", formatDuration(preview.annotated_duration_seconds)],
      ["Annotation coverage", percentage(preview.annotated_duration_seconds, preview.represented_duration_seconds)],
      ["Frame interval", `${Math.round(preview.frame_seconds * 1000)} ms`],
      ["Supervised frames", String(supervisedFrames.length)],
      ["Assistant-floor input ≥ 0.5", percentage(assistantFloorFrames.length, preview.frames.length)],
      ["Runtime floor supervision", supervisionCoverage(validHasFloorFrames, supervisedFrames)],
      ["Floor target ≥ 0.5", String(hasFloorPositives)],
      ["Runtime yield supervision", supervisionCoverage(validYieldFrames, supervisedFrames)],
      ["Floor available in 500 ms ≥ 0.5", String(yieldSoonPositives)],
      ["Assistant backchannel ≤ 200 ms ≥ 0.5", String(assistantBackchannelPositives)],
      ["Completion / floor-take point frames", `${auxiliaryCounts.turnCompletion} / ${auxiliaryCounts.floorTake}`],
      ["Pause / feedback span frames", `${auxiliaryCounts.continuationPause} / ${auxiliaryCounts.nonFloorFeedback}`],
    ]),
  );
}

function renderAnnotationSource() {
  const timelineIdentity =
    preview.timeline_fingerprint === null
      ? "raw timeline"
      : `repaired timeline ${preview.timeline_fingerprint.slice(0, 12)}…`;
  annotationSource.textContent =
    `${preview.annotation_version} · ${timelineIdentity} · annotation generated ` +
    `${formatDateTime(preview.annotation_generated_at)} · ` +
    `user audio ${preview.user_audio_sha256.slice(0, 12)}… · ` +
    `assistant audio ${preview.assistant_audio_sha256.slice(0, 12)}…`;
}

function interactionAuxiliaryCounts(frames) {
  const counts = {
    turnCompletion: 0,
    continuationPause: 0,
    nonFloorFeedback: 0,
    floorTake: 0,
  };
  frames.forEach((frame) => {
    if (frame.interaction_auxiliary.turn_completion.valid) {
      counts.turnCompletion += 1;
    }
    if (frame.interaction_auxiliary.continuation_pause.valid) {
      counts.continuationPause += 1;
    }
    if (frame.interaction_auxiliary.non_floor_feedback.valid) {
      counts.nonFloorFeedback += 1;
    }
    if (frame.interaction_auxiliary.floor_take.valid) {
      counts.floorTake += 1;
    }
  });
  return counts;
}

function drawTimeline() {
  const devicePixelRatio = window.devicePixelRatio || 1;
  const displayWidth = Math.max(980, timeline.clientWidth);
  const rowHeight = 36;
  const rowGap = 6;
  const displayHeight = 58 + rowDefinitions.length * (rowHeight + rowGap);
  timeline.width = Math.round(displayWidth * devicePixelRatio);
  timeline.height = Math.round(displayHeight * devicePixelRatio);
  const context = timeline.getContext("2d");
  context.scale(devicePixelRatio, devicePixelRatio);
  context.clearRect(0, 0, displayWidth, displayHeight);
  if (preview === null) {
    return;
  }

  const left = 245;
  const right = 16;
  const top = 36;
  const plotWidth = displayWidth - left - right;
  const burnRatio = (preview.burn_in_end_seconds - preview.start_seconds) / preview.input_duration_seconds;

  rowDefinitions.forEach(({ label, field, validField }, rowIndex) => {
    const rowTop = top + rowIndex * (rowHeight + rowGap);
    context.fillStyle = rowIndex % 2 === 0 ? "#f7f9f8" : "#f1f4f3";
    context.fillRect(left, rowTop, plotWidth, rowHeight);
    context.fillStyle = "#48575c";
    context.font =
      field === "user_yield_target" ||
      field === "user_has_floor_target" ||
      field === "assistant_has_floor_input" ||
      field === "assistant_backchannel_target"
        ? "bold 12px sans-serif"
        : "12px sans-serif";
    context.fillText(label, 8, rowTop + 23);
    if (field === "waveform") {
      drawWaveform(context, left, rowTop, plotWidth, rowHeight);
    } else {
      drawTargetFrames(context, field, validField, left, rowTop, plotWidth, rowHeight);
    }
  });

  drawBurnInOverlay(context, left, top, plotWidth, displayHeight, burnRatio);
  drawUnusableRegionOverlay(
    context,
    preview.conversation_regions,
    left,
    top - 16,
    plotWidth,
    displayHeight - top + 2,
    preview.start_seconds,
    preview.end_seconds,
  );
  drawTimeAxis(context, left, plotWidth, displayHeight);
  drawCursor(context, left, plotWidth, top, displayHeight);
}

function drawSourceAnnotationTimeline() {
  const devicePixelRatio = window.devicePixelRatio || 1;
  const displayWidth = Math.max(980, annotationTimeline.clientWidth);
  const displayHeight = 270;
  annotationTimeline.width = Math.round(displayWidth * devicePixelRatio);
  annotationTimeline.height = Math.round(displayHeight * devicePixelRatio);
  const context = annotationTimeline.getContext("2d");
  context.scale(devicePixelRatio, devicePixelRatio);
  context.clearRect(0, 0, displayWidth, displayHeight);
  if (preview === null) {
    return;
  }

  const left = 245;
  const right = 16;
  const plotWidth = displayWidth - left - right;
  const waveformRows = [
    {
      label: `USER AUDIO · ${prettySide(preview.user_side)}`,
      points: preview.user_waveform,
      color: "#0057d9",
      gain: preview.user_gain.default_gain,
    },
    {
      label: `ASSISTANT AUDIO · ${prettySide(preview.assistant_side)}`,
      points: preview.assistant_waveform,
      color: "#7a1fa2",
      gain: preview.assistant_gain.default_gain,
    },
  ];
  const displayScale = commonWaveformDisplayScale(waveformRows);
  waveformRows.forEach((row, rowIndex) => {
    const top = 32 + rowIndex * 42;
    context.fillStyle = "#48575c";
    context.font = "bold 12px sans-serif";
    context.fillText(row.label, 8, top + 22);
    drawWaveformPoints(
      context,
      row.points,
      left,
      top,
      plotWidth,
      32,
      row.color,
      row.gain * displayScale,
    );
  });
  const rows = [
    {
      label: `USER · ${prettySide(preview.user_side)}`,
      annotation: sourceAnnotation("user"),
    },
    {
      label: `ASSISTANT · ${prettySide(preview.assistant_side)}`,
      annotation: sourceAnnotation("assistant"),
    },
  ];
  rows.forEach((row, rowIndex) => {
    const top = 124 + rowIndex * 58;
    context.fillStyle = "#48575c";
    context.font = "bold 12px sans-serif";
    context.fillText(row.label, 8, top + 26);
    drawAnnotationTimelineRow({
      context,
      annotation: row.annotation,
      left,
      top,
      width: plotWidth,
      viewportStartSeconds: preview.start_seconds,
      viewportEndSeconds: preview.end_seconds,
    });
  });
  drawUnusableRegionOverlay(
    context,
    preview.conversation_regions,
    left,
    30,
    plotWidth,
    displayHeight - 48,
    preview.start_seconds,
    preview.end_seconds,
  );
  drawAnnotationTimeAxis(context, left, plotWidth, displayHeight);
  drawCursor(context, left, plotWidth, 30, displayHeight);
}

function sourceAnnotation(role) {
  const spans = preview[`${role}_spans`];
  const points = preview[`${role}_points`];
  return {
    speech_segments: spans.filter((span) => span.event_type === `${role}_speech`),
    pauses: spans.filter((span) => span.event_type === `${role}_pause`),
    backchannels: spans.filter((span) => span.event_type === `${role}_backchannel`),
    turns: points.filter((point) => point.event_type === `${role}_end_of_turn`),
    interruptions: points.filter((point) => point.event_type === `${role}_interruption`),
    segment_targets: preview[`${role}_segment_targets`],
    connection_targets: preview[`${role}_connection_targets`],
  };
}

function drawAnnotationTimeAxis(context, left, width, height) {
  context.strokeStyle = "#cbd3d1";
  context.fillStyle = "#657378";
  context.font = "11px sans-serif";
  for (let seconds = 0; seconds <= preview.input_duration_seconds; seconds += 2) {
    const ratio = seconds / preview.input_duration_seconds;
    const x = left + ratio * width;
    context.beginPath();
    context.moveTo(x, 24);
    context.lineTo(x, height - 24);
    context.stroke();
    context.fillText(`+${seconds}s`, x + 3, height - 17);
    context.fillText(formatDuration(preview.start_seconds + seconds), x + 3, height - 5);
  }
}

function drawBurnInOverlay(context, left, top, width, height, burnRatio) {
  const burnWidth = width * burnRatio;
  context.fillStyle = "rgba(117, 131, 136, 0.13)";
  context.fillRect(left, top - 16, burnWidth, height - top + 2);
  context.strokeStyle = "#879397";
  context.setLineDash([3, 3]);
  context.beginPath();
  context.moveTo(left + burnWidth, top - 16);
  context.lineTo(left + burnWidth, height - 18);
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = "#5d6b70";
  context.font = "11px sans-serif";
  context.fillText("4 s burn-in · all losses masked", left + 8, top - 5);
}

function drawWaveform(context, left, top, width, height) {
  const gain = preview.user_gain.default_gain;
  const displayScale = commonWaveformDisplayScale([
    { points: preview.user_waveform, gain },
  ]);
  drawWaveformPoints(
    context,
    preview.user_waveform,
    left,
    top,
    width,
    height,
    "#0057d9",
    gain * displayScale,
  );
}

function drawWaveformPoints(context, points, left, top, width, height, color, gain) {
  const middle = top + height / 2;
  const amplitudeScale = height * 0.43;
  context.fillStyle = "#f7f9f8";
  context.fillRect(left, top, width, height);
  context.strokeStyle = color;
  context.lineWidth = 2;
  points.forEach((point, index) => {
    const x = left + (index / Math.max(1, points.length - 1)) * width;
    context.beginPath();
    const maximumAmplitude = Math.min(1, point.maximum_amplitude * gain);
    const minimumAmplitude = Math.max(-1, point.minimum_amplitude * gain);
    context.moveTo(x, middle - maximumAmplitude * amplitudeScale);
    context.lineTo(x, middle - minimumAmplitude * amplitudeScale);
    context.stroke();
  });
}

function drawTargetFrames(context, field, validField, left, top, width, height) {
  const frameWidth = width / preview.frames.length;
  preview.frames.forEach((frame, index) => {
    const target = targetValue(frame, field, validField);
    if (!target.valid) {
      drawMaskedFrame(
        context,
        left + index * frameWidth,
        top,
        frameWidth,
        height,
        target.maskReason,
      );
      return;
    }
    const value = target.value;
    if (value === null) {
      return;
    }
    if (typeof value === "boolean") {
      context.fillStyle = "#7ebdb4";
      if (!value) {
        return;
      }
    } else {
      context.fillStyle =
        field === "assistant_has_floor_input"
          ? `rgba(190, 128, 20, ${0.08 + 0.92 * value})`
          : `rgba(20, 107, 99, ${0.08 + 0.92 * value})`;
    }
    if (value === true || typeof value === "number") {
      context.fillRect(left + index * frameWidth, top + 3, Math.max(1, frameWidth + 0.2), height - 6);
    }
  });
}

function drawMaskedFrame(context, left, top, width, height, maskReason) {
  if (
    maskReason === NO_AUXILIARY_ANNOTATION_REASON ||
    maskReason === OUTSIDE_USER_YIELD_CONTEXT_REASON
  ) {
    return;
  }
  const isBurnIn = maskReason === BURN_IN_REASON;
  context.fillStyle = isBurnIn ? "#e2e7e5" : "#f5e7bf";
  context.fillRect(left, top + 3, Math.max(1, width + 0.2), height - 6);
  if (width < 2.5) {
    return;
  }
  context.strokeStyle = isBurnIn ? "#c5cecb" : "#c39328";
  context.lineWidth = 0.6;
  context.beginPath();
  context.moveTo(left, top + height - 3);
  context.lineTo(left + Math.max(1, width), top + 3);
  context.stroke();
}

function targetValue(frame, field, validField) {
  if (field.startsWith("future_activity.")) {
    const target = frame.future_activity[Number(field.split(".")[1])];
    return {
      value: target?.occupancy ?? null,
      valid: target?.valid === true,
      maskReason: target?.mask_reason ?? null,
    };
  }
  if (field.startsWith("interaction_auxiliary.")) {
    const target = frame.interaction_auxiliary[field.split(".")[1]];
    return {
      value: target.target,
      valid: target.valid,
      maskReason: target.mask_reason,
    };
  }
  const maskReasonField =
    field === "user_has_floor_target"
      ? "user_has_floor_mask_reason"
      : field === "user_yield_target"
        ? "user_yield_mask_reason"
        : field === "assistant_backchannel_target"
          ? "assistant_backchannel_mask_reason"
        : null;
  return {
    value: frame[field] ?? null,
    valid: validField === undefined || frame[validField] === true,
    maskReason: maskReasonField === null ? null : frame[maskReasonField],
  };
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
      ["INPUT · p_assistant_has_floor", optionalProbability(frame.assistant_has_floor_input)],
      ["Candidate", booleanLabel(frame.candidate)],
      ["Candidate source", frame.candidate_source ?? "—"],
      ["Since user speech offset", optionalSeconds(frame.seconds_since_speech_offset)],
      ["p_user_has_floor target", maskedProbability(
        frame.user_has_floor_target,
        frame.user_has_floor_valid,
        frame.user_has_floor_mask_reason,
      )],
      ["p_user_yield ≤ 500 ms target", maskedProbability(
        frame.user_yield_target,
        frame.user_yield_valid,
        frame.user_yield_mask_reason,
      )],
      ["p_assistant_backchannel ≤ 200 ms target", maskedProbability(
        frame.assistant_backchannel_target,
        frame.assistant_backchannel_valid,
        frame.assistant_backchannel_mask_reason,
      )],
      ["Point: turn completion", auxiliaryProbability(frame, "turn_completion")],
      ["Span: continuation pause", auxiliaryProbability(frame, "continuation_pause")],
      ["Span: non-floor feedback", auxiliaryProbability(frame, "non_floor_feedback")],
      ["Point: floor take / interruption", auxiliaryProbability(frame, "floor_take")],
      ...frame.future_activity.map((target) => [
        `Future user activity ${target.start_milliseconds}–${target.end_milliseconds} ms`,
        maskedProbability(target.occupancy, target.valid, target.mask_reason),
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
    } else if (value.startsWith("MASKED")) {
      definition.className = "value-masked";
    }
    return [term, definition];
  });
}

function selectFrameAtEvent(event) {
  if (preview === null) {
    return;
  }
  const rect = timeline.getBoundingClientRect();
  const left = 245;
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
  setCanonicalPlaybackTime(playbackSeconds);
  renderSelectedFrame(frame);
  updateTimeReadout();
  drawTimeline();
  drawSourceAnnotationTimeline();
}

async function togglePlayback() {
  if (preview === null) {
    return;
  }
  if (!userAudio.paused) {
    pausePlayback();
    return;
  }
  const canonicalTime = canonicalPlaybackTime();
  if (canonicalTime < preview.start_seconds || canonicalTime >= preview.end_seconds) {
    setCanonicalPlaybackTime(preview.start_seconds);
  }
  await startPlayback();
}

function togglePlaybackFromSpace(event) {
  if (event.code !== "Space") {
    return;
  }
  event.preventDefault();
  if (event.repeat) {
    return;
  }
  void togglePlayback();
}

function pausePlayback() {
  userAudio.pause();
  assistantAudio.pause();
  playButton.textContent = playBothInput.checked ? "Play both tracks" : "Play user input";
}

function trackPlayback() {
  if (preview === null) {
    return;
  }
  playbackSeconds = canonicalPlaybackTime();
  if (playbackSeconds >= preview.end_seconds) {
    setCanonicalPlaybackTime(preview.end_seconds);
    playbackSeconds = preview.end_seconds;
    pausePlayback();
  } else if (playBothInput.checked && audioSeparationSeconds() > 0.08) {
    setCanonicalPlaybackTime(playbackSeconds);
  }
  updateTimeReadout();
  drawTimeline();
  drawSourceAnnotationTimeline();
}

function synchronizeAudioTracks() {
  setCanonicalPlaybackTime(canonicalPlaybackTime());
}

function canonicalPlaybackTime() {
  if (preview === null) {
    return 0;
  }
  const sourceTime =
    preview.user_side === "speaker1"
      ? userAudio.currentTime
      : assistantAudio.currentTime;
  const sourceStart =
    preview.user_side === "speaker1"
      ? preview.user_source_start_seconds
      : preview.assistant_source_start_seconds;
  return preview.start_seconds + sourceTime - sourceStart;
}

function setCanonicalPlaybackTime(canonicalSeconds) {
  if (preview === null) {
    return;
  }
  const relativeSeconds = canonicalSeconds - preview.start_seconds;
  userAudio.currentTime = preview.user_source_start_seconds + relativeSeconds;
  assistantAudio.currentTime =
    preview.assistant_source_start_seconds + relativeSeconds;
}

function audioSeparationSeconds() {
  if (preview === null) {
    return 0;
  }
  const userRelative = userAudio.currentTime - preview.user_source_start_seconds;
  const assistantRelative =
    assistantAudio.currentTime - preview.assistant_source_start_seconds;
  return Math.abs(userRelative - assistantRelative);
}

async function updatePlaybackMode() {
  if (!playBothInput.checked) {
    assistantAudio.pause();
    if (userAudio.paused) {
      playButton.textContent = "Play user input";
    }
    return;
  }
  if (userAudio.paused) {
    playButton.textContent = "Play both tracks";
    return;
  }
  synchronizeAudioTracks();
  try {
    await assistantAudio.play();
  } catch {
    setStatus("The second track was blocked; pause and press Play both tracks.", false);
  }
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

async function readJsonResponse(response) {
  const responseText = await response.text();
  try {
    return JSON.parse(responseText);
  } catch {
    const responseDescription = responseText.trim() || response.statusText || "Empty response";
    throw new Error(`Request failed (${response.status}): ${responseDescription}`);
  }
}

function errorMessage(payload, statusCode) {
  return payload !== null && typeof payload === "object" && typeof payload.detail === "string"
    ? payload.detail
    : `Request failed (${statusCode})`;
}

function sampleOptionLabel(sample) {
  return `${sample.external_id} · quality ${optionalScore(sample.quality_score)} · ${formatDuration(sample.represented_duration_seconds)} · ${sample.usable_event_count} events`;
}

function selectedMinimumQuality() {
  if (minimumQualityInput.value === "") {
    return null;
  }
  if (!minimumQualityInput.checkValidity()) {
    throw new Error("Minimum quality must be between 0 and 1.");
  }
  return minimumQualityInput.valueAsNumber;
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

function formatGain(gain) {
  if (gain.estimated_active_rms_dbfs === null) {
    return `${gain.default_gain.toFixed(2)}× · loudness unavailable`;
  }
  return (
    `${gain.default_gain.toFixed(2)}× · ` +
    `${gain.estimated_active_rms_dbfs.toFixed(1)} dBFS → ` +
    `${gain.target_active_rms_dbfs.toFixed(0)} dBFS`
  );
}

function maskedProbability(value, valid, maskReason) {
  if (valid) {
    return optionalProbability(value);
  }
  return maskReason === NO_AUXILIARY_ANNOTATION_REASON
    ? "NO AUXILIARY LABEL HERE"
    : `MASKED · ${prettyMaskReason(maskReason)}`;
}

function prettyMaskReason(maskReason) {
  if (maskReason === null) {
    return "unspecified";
  }
  return maskReason.replaceAll("_", " ");
}

function optionalScore(value) {
  return value === null ? "—" : value.toFixed(3);
}

function optionalInteger(value) {
  return value === null ? "—" : String(value);
}

function optionalDecimal(value) {
  return value === null ? "—" : value.toFixed(1);
}

function optionalSeconds(value) {
  return value === null ? "—" : `${value.toFixed(2)} s`;
}

function auxiliaryProbability(frame, field) {
  const target = frame.interaction_auxiliary[field];
  return maskedProbability(target.target, target.valid, target.mask_reason);
}

function percentage(count, total) {
  return total === 0 ? "0.0%" : `${((100 * count) / total).toFixed(1)}%`;
}

function supervisionCoverage(validFrames, supervisedFrames) {
  return `${validFrames.length} / ${supervisedFrames.length} (${percentage(validFrames.length, supervisedFrames.length)})`;
}

function formatDuration(totalSeconds) {
  const bounded = Math.max(0, totalSeconds);
  const minutes = Math.floor(bounded / 60);
  const seconds = bounded - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
}

function formatDateTime(value) {
  return new Date(value).toLocaleString();
}

datasetSelect.addEventListener("change", () => {
  void loadSamples();
});
sampleSelect.addEventListener("change", () => loadPreview(true));
userSideSelect.addEventListener("change", () => loadPreview(true));
loadButton.addEventListener("click", () => loadPreview(false));
randomButton.addEventListener("click", () => loadPreview(true));
nextRandomButton.addEventListener("click", loadNextPreparedSample);
minimumQualityInput.addEventListener("change", () => {
  resetReviewQueue();
  void loadSamples();
});
samplingModeSelect.addEventListener("change", () => {
  resetReviewQueue();
  void prepareNextReviewSample();
});
positionSlider.addEventListener("input", () => {
  const startSeconds = Number(positionSlider.value);
  startInput.value = startSeconds.toFixed(2);
  contextOverviewController.draw();
});
positionSlider.addEventListener("change", () => loadPreview(false));
startInput.addEventListener("input", () => {
  if (startInput.value === "" || !startInput.checkValidity()) {
    return;
  }
  positionSlider.value = startInput.value;
  contextOverviewController.draw();
});
startInput.addEventListener("change", () => loadPreview(false));
playButton.addEventListener("click", togglePlayback);
playBothInput.addEventListener("change", updatePlaybackMode);
userAudio.addEventListener("timeupdate", trackPlayback);
userAudio.addEventListener("ended", pausePlayback);
timeline.addEventListener("click", selectFrameAtEvent);
document.addEventListener("keydown", togglePlaybackFromSpace);
window.addEventListener("resize", () => {
  drawTimeline();
  drawSourceAnnotationTimeline();
  contextOverviewController.draw();
});

await loadDatasets();
await loadSamples();
