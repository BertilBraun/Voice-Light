const state = {
  datasets: [],
  samples: [],
  playbackController: null,
};

const annotationMetricDescriptions = new Map([
  ["Interactions", "Any speaker exchange counted as a turn-taking, backchannel, or interruption."],
  ["Turns", "A detected point where one speaker's contribution appears complete."],
  ["Turn takings", "A transition where the next transcript segment comes from the other speaker."],
  ["Pauses", "A meaningful silence within one speaker's ongoing contribution."],
  ["Backchannels", "A short acknowledgment such as 'yeah,' 'right,' or 'mhm' that supports the other speaker without taking the floor."],
  ["Interruptions", "An event where one speaker begins taking the floor before the other speaker has finished."],
  ["Useful events", "Any detected turn completion, pause, backchannel, or interruption that can serve as an annotation or training signal."],
]);

let nextMetricTooltipId = 1;

const elements = {
  status: document.querySelector("#status"),
  refreshButton: document.querySelector("#refresh-button"),
  datasetFilter: document.querySelector("#dataset-filter"),
  qualityMin: document.querySelector("#quality-min"),
  overlapMax: document.querySelector("#overlap-max"),
  flagFilter: document.querySelector("#flag-filter"),
  sampleList: document.querySelector("#sample-list"),
  sampleDetail: document.querySelector("#sample-detail"),
  conversationSummary: document.querySelector("#conversation-summary"),
};

elements.refreshButton.addEventListener("click", loadDashboard);
for (const element of [
  elements.datasetFilter,
  elements.qualityMin,
  elements.overlapMax,
  elements.flagFilter,
]) {
  element.addEventListener("change", loadSamples);
}

loadDashboard();

async function loadDashboard() {
  elements.status.textContent = "Loading database records";
  const datasetsPayload = await fetchJson("/api/dataset-dashboard/datasets");
  state.datasets = datasetsPayload.datasets;
  renderDatasetOptions();
  await loadSamples();
}

async function loadSamples() {
  const parameters = new URLSearchParams();
  const datasetId = elements.datasetFilter.value;
  if (datasetId) {
    parameters.set("dataset_id", datasetId);
  }
  if (elements.qualityMin.value) {
    parameters.set("quality_min", elements.qualityMin.value);
  }
  if (elements.overlapMax.value) {
    parameters.set("overlap_ratio_max", elements.overlapMax.value);
  }
  if (elements.flagFilter.value.trim()) {
    parameters.set("flag", elements.flagFilter.value.trim());
  }
  const samplesPayload = await fetchJson(`/api/dataset-dashboard/samples?${parameters}`);
  const summary = await fetchJson(`/api/dataset-dashboard/conversation-summary?${parameters}`);
  state.samples = samplesPayload.samples;
  elements.status.textContent = `${state.datasets.length} datasets, ${state.samples.length} samples`;
  renderSamples();
  renderConversationSummary(summary);
}

function renderConversationSummary(summary) {
  const section = createMetricSection("Conversation annotation overview", [
    ["Analyzed samples", summary.analyzed_sample_count, "integer"],
    ["Invalid samples", summary.invalid_sample_count, "integer"],
    ["Analyzed audio", summary.analyzed_duration_seconds / 3600, "hours"],
    ["Speech segments", summary.speech_segment_count, "integer"],
    ["Interactions", summary.interaction_count, "integer"],
    ["Turns", summary.turn_count, "integer"],
    ["Turn takings", summary.turn_taking_count, "integer"],
    ["Pauses", summary.pause_count, "integer"],
    ["Backchannels", summary.backchannel_count, "integer"],
    ["Interruptions", summary.interruption_count, "integer"],
    ["Useful events", summary.usable_event_count, "integer"],
  ]);
  elements.conversationSummary.replaceChildren(...section.childNodes);
}

function renderDatasetOptions() {
  elements.datasetFilter.replaceChildren();
  elements.datasetFilter.appendChild(new Option("All datasets", ""));
  for (const dataset of state.datasets) {
    elements.datasetFilter.appendChild(new Option(dataset.name, dataset.id));
  }
}

function renderSamples() {
  elements.sampleList.replaceChildren();
  for (const sample of state.samples) {
    const button = document.createElement("button");
    button.className = "sample-row";
    button.type = "button";
    button.addEventListener("click", () => renderSampleDetail(sample));

    const text = document.createElement("div");
    const title = document.createElement("div");
    title.className = "sample-id";
    title.textContent = sample.sample.external_id;
    const metrics = document.createElement("div");
    metrics.className = "metric-line";
    metrics.textContent = [
      formatSeconds(sample.sample.duration_seconds),
      `speech ${formatRatio(sample.latest_quality?.speech_ratio)}`,
      `overlap ${formatRatio(sample.latest_quality?.overlap_ratio)}`,
    ].join(" - ");
    text.append(title, metrics);

    const score = document.createElement("div");
    score.className = "score";
    score.textContent = formatRatio(sample.sample.quality_score);
    button.append(text, score);
    elements.sampleList.appendChild(button);
  }
}

function renderSampleDetail(sample) {
  state.playbackController?.destroy();

  const container = document.createElement("div");
  container.className = "detail-grid";

  const payload = sample.latest_quality?.payload || {};
  const heading = document.createElement("div");
  heading.className = "detail-heading";
  const titleGroup = document.createElement("div");
  const title = document.createElement("h2");
  title.textContent = sample.sample.external_id;
  const version = document.createElement("div");
  version.className = "muted";
  version.textContent = payload.metric_version || "No completed quality result";
  titleGroup.append(title, version);
  heading.append(titleGroup, createScoreBadge(payload.total_quality_score));
  container.appendChild(heading);

  container.appendChild(
    createMetricSection("Quality scores", [
      ["Overall", payload.total_quality_score, "score"],
      ["Calibrated", payload.calibrated_quality_score, "score"],
      ["Raw", payload.raw_quality_score, "score"],
      ["Interaction", payload.interaction_density?.quality_score, "score"],
      ["Timing", payload.timing_reliability?.quality_score, "score"],
      ["Audio", payload.audio_quality?.quality_score, "score"],
      ["Conversation", payload.conversation_annotation?.quality_score, "score"],
    ]),
  );

  const playbackController = createSynchronizedPlayback(sample);
  state.playbackController = playbackController;
  container.appendChild(playbackController.element);

  const interaction = payload.interaction_density || {};
  const conversation = payload.conversation_annotation || {};
  container.appendChild(
    createMetricSection("ASR conversation annotation", [
      ["Analyzed duration", conversation.analyzed_duration_seconds, "seconds"],
      ["Speech segments", conversation.speech_segment_count, "integer"],
      ["Interactions", conversation.interaction_count, "integer"],
      ["Turns", conversation.turn_count, "integer"],
      ["Turn takings", conversation.turn_taking_count, "integer"],
      ["Pauses", conversation.pause_count, "integer"],
      ["Backchannels", conversation.backchannel_count, "integer"],
      ["Interruptions", conversation.interruption_count, "integer"],
      ["Useful events", conversation.usable_event_count, "integer"],
      ["Events / hour", conversation.events_per_hour, "number"],
      ["Speaker balance", conversation.speaker_balance_score, "score"],
    ]),
  );

  container.appendChild(
    createMetricSection("Interaction", [
      ["Speech", interaction.speech_ratio, "percent"],
      ["Silence", interaction.silence_ratio, "percent"],
      ["Overlap", interaction.overlap_ratio, "percent"],
      ["Candidates / hour", interaction.usable_candidate_windows_per_hour, "number"],
      ["Turns / hour", interaction.turn_completions_per_hour, "number"],
      ["Responses / hour", interaction.start_responses_per_hour, "number"],
      ["Interruptions / hour", interaction.interruptions_per_hour, "number"],
      ["Backchannels / hour", interaction.backchannels_per_hour, "number"],
    ]),
  );

  const timing = payload.timing_reliability || {};
  container.appendChild(
    createMetricSection("Timing reliability", [
      ["Median segment", timing.median_segment_duration_seconds, "seconds"],
      ["Median turn gap", timing.median_turn_gap_seconds, "seconds"],
      ["Median pause", timing.median_pause_duration_seconds, "seconds"],
      ["Median overlap", timing.median_overlap_duration_seconds, "seconds"],
      ["Tiny fragments", timing.tiny_fragment_ratio, "percent"],
      ["Long segments", timing.long_segment_ratio, "percent"],
    ]),
  );

  container.appendChild(createAudioQualitySection(payload.audio_quality));

  const chips = document.createElement("div");
  chips.className = "chips";
  const flags = [...new Set([...(sample.sample.quality_flags || []), ...(payload.audio_quality?.flags || [])])];
  for (const flag of flags) {
    const chip = document.createElement("span");
    chip.className = "chip warning-chip";
    chip.textContent = flag;
    chips.appendChild(chip);
  }
  if (flags.length > 0) {
    container.appendChild(chips);
  }

  container.appendChild(createEventSection(payload.event_candidates || [], playbackController));

  const rawDetails = document.createElement("details");
  rawDetails.className = "raw-details";
  const rawSummary = document.createElement("summary");
  rawSummary.textContent = "Raw quality JSON";
  const qualityPayload = document.createElement("pre");
  qualityPayload.textContent = JSON.stringify(payload, null, 2);
  rawDetails.append(rawSummary, qualityPayload);
  container.appendChild(rawDetails);
  elements.sampleDetail.replaceChildren(container);
}

function createSynchronizedPlayback(sample) {
  const section = document.createElement("section");
  section.className = "quality-section playback-section";
  const heading = document.createElement("div");
  heading.className = "section-heading";
  const title = document.createElement("h3");
  title.textContent = "Synchronized playback";
  const status = document.createElement("span");
  status.className = "muted";
  status.textContent = "Loading audio";
  heading.append(title, status);

  const controls = document.createElement("div");
  controls.className = "playback-controls";
  const playButton = document.createElement("button");
  playButton.type = "button";
  playButton.className = "play-button";
  playButton.textContent = "Play both";
  const seek = document.createElement("input");
  seek.type = "range";
  seek.className = "seek";
  seek.min = "0";
  seek.max = "1000";
  seek.value = "0";
  seek.setAttribute("aria-label", "Playback position");
  const clock = document.createElement("span");
  clock.className = "playback-clock";
  clock.textContent = "0:00 / --:--";
  controls.append(playButton, seek, clock);

  const tracks = document.createElement("div");
  tracks.className = "track-grid";
  const trackPlayers = sample.tracks.map((track) => {
    const card = document.createElement("div");
    card.className = `track-card ${track.side}`;
    const trackTitle = document.createElement("strong");
    trackTitle.textContent = formatSpeaker(track.side);
    const metadata = document.createElement("span");
    metadata.className = "muted";
    metadata.textContent = `${formatSeconds(track.duration_seconds)} · ${track.sample_rate || "?"} Hz · ${track.channels || "?"} ch`;
    const waveformBox = document.createElement("div");
    waveformBox.className = "waveform-box";
    const canvas = document.createElement("canvas");
    canvas.className = "waveform-canvas";
    canvas.width = 1200;
    canvas.height = 100;
    canvas.setAttribute("aria-label", `${formatSpeaker(track.side)} full recording waveform`);
    const waveformStatus = document.createElement("span");
    waveformStatus.className = "waveform-status";
    waveformStatus.textContent = "Building full waveform…";
    waveformBox.append(canvas, waveformStatus);
    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.src = `/api/dataset-dashboard/audio/${sample.sample.id}/${track.side}`;
    card.append(trackTitle, metadata, waveformBox, audio);
    tracks.appendChild(card);
    const player = { audio, canvas, waveformStatus, baseWaveform: null };
    canvas.addEventListener("click", (event) => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1);
      seekTo(ratio * duration());
    });
    void loadTrackWaveform(sample.sample.id, track.side, player).then(update);
    return player;
  });
  const audioElements = trackPlayers.map((player) => player.audio);
  section.append(heading, controls, tracks);

  const master = audioElements[0];
  let destroyed = false;
  let seeking = false;

  function duration() {
    const durations = audioElements.map((audio) => audio.duration).filter(Number.isFinite);
    return durations.length === 0 ? 0 : Math.min(...durations);
  }

  function update() {
    if (destroyed || !master) {
      return;
    }
    const sharedDuration = duration();
    const currentTime = master.currentTime;
    if (!seeking && sharedDuration > 0) {
      seek.value = String(Math.round((currentTime / sharedDuration) * 1000));
    }
    clock.textContent = `${formatClock(currentTime)} / ${sharedDuration > 0 ? formatClock(sharedDuration) : "--:--"}`;
    playButton.textContent = master.paused ? "Play both" : "Pause both";
    status.textContent = audioElements.length === 2 ? "Speaker 1 + Speaker 2" : `${audioElements.length} track(s)`;
    for (const player of trackPlayers) {
      const trackDuration = player.audio.duration;
      const progress = Number.isFinite(trackDuration) && trackDuration > 0 ? currentTime / trackDuration : 0;
      renderWaveformProgress(player, progress);
    }
    if (!seeking && !master.paused) {
      for (const audio of audioElements.slice(1)) {
        if (Math.abs(audio.currentTime - currentTime) > 0.08) {
          audio.currentTime = currentTime;
        }
      }
    }
  }

  async function play() {
    if (!master) {
      return;
    }
    const startTime = master.currentTime;
    for (const audio of audioElements) {
      audio.currentTime = startTime;
    }
    const results = await Promise.allSettled(audioElements.map((audio) => audio.play()));
    if (results.some((result) => result.status === "rejected")) {
      audioElements.forEach((audio) => audio.pause());
      status.textContent = "Playback could not start";
    }
    update();
  }

  function pause() {
    audioElements.forEach((audio) => audio.pause());
    update();
  }

  function seekTo(seconds) {
    const sharedDuration = duration();
    const nextTime = sharedDuration > 0 ? Math.min(Math.max(seconds, 0), sharedDuration) : Math.max(seconds, 0);
    for (const audio of audioElements) {
      audio.currentTime = nextTime;
    }
    update();
  }

  playButton.addEventListener("click", () => {
    if (master?.paused !== false) {
      void play();
    } else {
      pause();
    }
  });
  seek.addEventListener("input", () => {
    seeking = true;
    seekTo((Number(seek.value) / 1000) * duration());
  });
  seek.addEventListener("change", () => {
    seeking = false;
    seekTo((Number(seek.value) / 1000) * duration());
  });
  for (const audio of audioElements) {
    audio.addEventListener("loadedmetadata", update);
    audio.addEventListener("ended", pause);
  }
  master?.addEventListener("timeupdate", update);

  return {
    element: section,
    seekTo,
    destroy() {
      destroyed = true;
      audioElements.forEach((audio) => {
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
      });
    },
  };
}

async function loadTrackWaveform(sampleId, side, player) {
  try {
    const waveform = await fetchJson(`/api/dataset-dashboard/waveform/${sampleId}/${side}?points=1200`);
    player.baseWaveform = drawWaveformEnvelope(player.canvas, waveform.points);
    player.waveformStatus.textContent = `Full recording · ${formatClock(waveform.duration_seconds)}`;
  } catch (error) {
    player.waveformStatus.textContent = error instanceof Error ? error.message : "Waveform unavailable";
  }
}

function drawWaveformEnvelope(canvas, points) {
  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }
  const width = canvas.width;
  const height = canvas.height;
  const midpoint = height / 2;
  context.fillStyle = "#f8fafc";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#d7dde5";
  context.beginPath();
  context.moveTo(0, midpoint + 0.5);
  context.lineTo(width, midpoint + 0.5);
  context.stroke();
  context.strokeStyle = "#386d9d";
  context.lineWidth = 1;
  context.beginPath();
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const x = (index / Math.max(points.length - 1, 1)) * width;
    context.moveTo(x, midpoint - point.maximum_amplitude * midpoint * 0.9);
    context.lineTo(x, midpoint - point.minimum_amplitude * midpoint * 0.9);
  }
  context.stroke();
  return context.getImageData(0, 0, width, height);
}

function renderWaveformProgress(player, progress) {
  if (!player.baseWaveform) {
    return;
  }
  const context = player.canvas.getContext("2d");
  if (!context) {
    return;
  }
  context.putImageData(player.baseWaveform, 0, 0);
  const position = Math.min(Math.max(progress, 0), 1) * player.canvas.width;
  context.fillStyle = "rgba(29, 79, 122, 0.14)";
  context.fillRect(0, 0, position, player.canvas.height);
  context.strokeStyle = "#b45309";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(position, 0);
  context.lineTo(position, player.canvas.height);
  context.stroke();
}

function createMetricSection(title, metrics) {
  const section = document.createElement("section");
  section.className = "quality-section";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const grid = document.createElement("div");
  grid.className = "metric-grid";
  for (const [label, value, format] of metrics) {
    const metric = document.createElement("div");
    metric.className = "metric-card";
    const metricValue = document.createElement("strong");
    metricValue.textContent = formatMetric(value, format);
    metric.append(createMetricLabel(label), metricValue);
    grid.appendChild(metric);
  }
  section.append(heading, grid);
  return section;
}

function createMetricLabel(label) {
  const labelRow = document.createElement("div");
  labelRow.className = "metric-label-row";
  const labelText = document.createElement("span");
  labelText.className = "metric-label-text";
  labelText.textContent = label;
  labelRow.appendChild(labelText);

  const description = annotationMetricDescriptions.get(label);
  if (!description) {
    return labelRow;
  }
  const tooltipId = `metric-tooltip-${nextMetricTooltipId}`;
  nextMetricTooltipId += 1;
  const help = document.createElement("button");
  help.type = "button";
  help.className = "metric-help";
  help.textContent = "?";
  help.setAttribute("aria-label", `Explain ${label}`);
  help.setAttribute("aria-describedby", tooltipId);
  help.title = description;
  const tooltip = document.createElement("span");
  tooltip.id = tooltipId;
  tooltip.className = "metric-tooltip";
  tooltip.role = "tooltip";
  tooltip.textContent = description;
  labelRow.append(help, tooltip);
  return labelRow;
}

function createAudioQualitySection(audioQuality) {
  const section = document.createElement("section");
  section.className = "quality-section";
  const heading = document.createElement("h3");
  heading.textContent = "Audio quality";
  section.appendChild(heading);
  if (!audioQuality) {
    section.appendChild(createEmptyMessage("No audio quality result"));
    return section;
  }

  const overview = createMetricSection("", [
    ["Duration gap", audioQuality.duration_gap_seconds, "seconds"],
    ["Duration mismatch", audioQuality.duration_gap_ratio, "percent"],
    ["Track correlation", audioQuality.track_correlation, "number"],
    ["Envelope correlation", audioQuality.energy_envelope_correlation, "number"],
    ["Speaker 1 leakage", audioQuality.speaker1_leakage_db, "db"],
    ["Speaker 2 leakage", audioQuality.speaker2_leakage_db, "db"],
  ]);
  overview.className = "audio-overview";
  section.appendChild(overview.querySelector(".metric-grid"));

  const trackGrid = document.createElement("div");
  trackGrid.className = "audio-quality-tracks";
  for (const [side, track] of [["Speaker 1", audioQuality.speaker1], ["Speaker 2", audioQuality.speaker2]]) {
    if (!track) {
      continue;
    }
    const card = document.createElement("div");
    card.className = "audio-quality-card";
    const title = document.createElement("strong");
    title.textContent = side;
    const details = document.createElement("dl");
    for (const [label, value] of [
      ["Score", formatMetric(track.quality_score, "score")],
      ["RMS", formatMetric(track.rms_dbfs, "db")],
      ["Peak", formatMetric(track.peak_amplitude, "number")],
      ["Clipping", formatMetric(track.clipping_ratio, "percent")],
      ["Near zero", formatMetric(track.near_zero_ratio, "percent")],
      ["Speech", formatMetric(track.speech_ratio, "percent")],
    ]) {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = value;
      details.append(term, description);
    }
    card.append(title, details);
    trackGrid.appendChild(card);
  }
  section.appendChild(trackGrid);
  return section;
}

function createEventSection(events, playbackController) {
  const section = document.createElement("section");
  section.className = "quality-section event-section";
  const heading = document.createElement("div");
  heading.className = "section-heading";
  const title = document.createElement("h3");
  title.textContent = "Event candidates";
  const total = document.createElement("span");
  total.className = "muted";
  total.textContent = `${events.length} stored candidates`;
  heading.append(title, total);
  section.appendChild(heading);
  if (events.length === 0) {
    section.appendChild(createEmptyMessage("No event candidates detected"));
    return section;
  }

  const counts = new Map();
  for (const event of events) {
    counts.set(event.event_type, (counts.get(event.event_type) || 0) + 1);
  }
  const chips = document.createElement("div");
  chips.className = "chips event-counts";
  for (const [eventType, count] of [...counts.entries()].sort()) {
    const chip = document.createElement("span");
    chip.className = `chip event-${eventType}`;
    chip.textContent = `${formatEventType(eventType)} ${count}`;
    chips.appendChild(chip);
  }
  section.appendChild(chips);

  const eventDetails = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Browse and seek to candidates";
  const list = document.createElement("div");
  list.className = "event-list";
  for (const event of events) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "event-row";
    row.addEventListener("click", () => playbackController.seekTo(event.start_seconds));
    const type = document.createElement("strong");
    type.textContent = formatEventType(event.event_type);
    const speakers = document.createElement("span");
    speakers.textContent = event.secondary_speaker
      ? `${formatSpeaker(event.primary_speaker)} → ${formatSpeaker(event.secondary_speaker)}`
      : formatSpeaker(event.primary_speaker);
    const timing = document.createElement("span");
    timing.textContent = `${formatClock(event.start_seconds)} · ${formatEventDuration(event)}`;
    row.append(type, speakers, timing);
    list.appendChild(row);
  }
  eventDetails.append(summary, list);
  section.appendChild(eventDetails);
  return section;
}

function createScoreBadge(value) {
  const badge = document.createElement("div");
  badge.className = `score-badge ${scoreClass(value)}`;
  badge.textContent = formatMetric(value, "score");
  return badge;
}

function createEmptyMessage(message) {
  const empty = document.createElement("div");
  empty.className = "detail-empty";
  empty.textContent = message;
  return empty;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message);
  }
  return response.json();
}

function formatSeconds(value) {
  if (value === null || value === undefined) {
    return "duration ?";
  }
  return `${Number(value).toFixed(1)}s`;
}

function formatRatio(value) {
  if (value === null || value === undefined) {
    return "?";
  }
  return Number(value).toFixed(2);
}

function formatMetric(value, format) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "—";
  }
  const number = Number(value);
  switch (format) {
    case "score":
      return number.toFixed(2);
    case "percent":
      return `${(number * 100).toFixed(1)}%`;
    case "seconds":
      return `${number.toFixed(2)} s`;
    case "db":
      return `${number.toFixed(1)} dB`;
    case "integer":
      return String(Math.round(number));
    case "hours":
      return `${number.toFixed(2)} h`;
    default:
      return number.toFixed(2);
  }
}

function formatClock(value) {
  if (!Number.isFinite(Number(value))) {
    return "--:--";
  }
  const totalSeconds = Math.max(0, Math.floor(Number(value)));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatSpeaker(side) {
  if (side === "speaker1") {
    return "Speaker 1";
  }
  if (side === "speaker2") {
    return "Speaker 2";
  }
  return side || "Unknown speaker";
}

function formatEventType(eventType) {
  return String(eventType)
    .split("_")
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(" ");
}

function formatEventDuration(event) {
  const duration = event.gap_seconds ?? event.overlap_seconds ?? event.end_seconds - event.start_seconds;
  return `${Number(duration).toFixed(2)} s`;
}

function scoreClass(value) {
  if (!Number.isFinite(Number(value))) {
    return "score-unknown";
  }
  if (Number(value) >= 0.75) {
    return "score-good";
  }
  if (Number(value) >= 0.45) {
    return "score-medium";
  }
  return "score-bad";
}
